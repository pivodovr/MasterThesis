import json
import statistics
import sys
import multiprocessing
from absl import logging
import copy

import numpy as np

import tfne
from tfne.algorithms.codeepneat.codeepneat import CoDeepNEAT
from tfne.algorithms.lamacodeepneat._lamacodeepneat_selection_bp import LamaCoDeepNEATSelectionBP
from tfne.algorithms.lamacodeepneat._lamacodeepneat_selection_mod import LamaCoDeepNEATSelectionMOD
from tfne.encodings.lamacodeepneat.lamacodeepneat_encoding import LamaCoDeepNEATEncoding
from tfne.populations.lamacodeepneat.lamacodeepneat_population import LamaCoDeepNEATPopulation
from tfne.algorithms.lamacodeepneat._lamacodeepneat_evolution_bp import LamaCoDeepNEATEvolutionBP
from tfne.algorithms.lamacodeepneat._lamacodeepneat_evolution_mod import LamaCoDeepNEATEvolutionMOD
from tfne.algorithms.lamacodeepneat.nsgaII import *


class LamaCoDeepNEAT(CoDeepNEAT,
                     LamaCoDeepNEATSelectionBP,
                     LamaCoDeepNEATSelectionMOD,
                     LamaCoDeepNEATEvolutionBP,
                     LamaCoDeepNEATEvolutionMOD):
    """
    LamaCoDeepNEAT inherits from CoDeepNEAT.
    Following methods are overwritten:
    * __init__
    * evaluate_population
    * evolve_population
    """

    def __init__(self, config, initial_state_file_path=None):
        """
        Initialize the LamaCDN algorithm by processing and sanity checking the supplied configuration, which saves
        all algorithm config parameters as instance variables. Then initialize the CoDeepNEAT encoding and population.
        Alternatively, if a backup-state is supplied, reinitialize the encoding and population with the state present
        in that backup.
        @param config: ConfigParser instance holding all documentation specified sections of the CoDeepNEAT algorithm
        @param initial_state_file_path: string file path to a state backup that is to be resumed
        of fit/sizes of genomes they are part of, false if you want to select parents from all mods/bps
        """
        # Register and process the supplied configuration
        super().__init__(config, initial_state_file_path)
        self.config = config
        self._process_config()
        self._sanity_check_config()

        # Declare variables of environment shapes to which the created genomes have to adhere to
        self.input_shape = None
        self.output_shape = None

        # If an initial state of the evolution was supplied, load and recreate this state for the algorithm as well as
        # its dependencies
        if initial_state_file_path is not None:
            # Load the backed up state for the algorithm from file
            with open(initial_state_file_path) as saved_state_file:
                saved_state = json.load(saved_state_file)

            # Initialize and register an associated CoDeepNEAT encoding and population outfitted with the saved state
            self.enc = tfne.deserialization.load_encoding(serialized_encoding=saved_state['encoding'],
                                                          dtype=self.dtype)
            self.pop = tfne.deserialization.load_population(serialized_population=saved_state['population'],
                                                            dtype=self.dtype,
                                                            module_config_params=self.available_mod_params)
        else:
            # Initialize and register a blank associated CoDeepNEAT encoding and population
            self.enc = LamaCoDeepNEATEncoding(dtype=self.dtype)
            self.pop = LamaCoDeepNEATPopulation()

    def evaluate_population(self, environment, save_dir_path) -> (int, (float, int)):
        """
        Evaluate the population by building the specified amount of genomes from each blueprint, all having randomly
        assigned specific modules for the inherent blueprint module species. Set the evaluated fitness of each blueprint
        and each module as the average fitness achieved by all genomes in which the respective member was invovled in.
        Return the generational counter as well as the achieved fitness of the best genome.
        Implementation is almost the same as in CoDeepNEAT. The difference is in support for genome weights inheritance.
        @param environment: instance of the evaluation environment
        @return: tuple of generation counter and best fitness and size achieved by best genome
        """
        # Initialize population evaluation progress bar. Print notice of evaluation start
        genome_pop_size = len(self.pop.blueprints) * self.genomes_per_bp
        genome_eval_counter = 0
        genome_eval_counter_div = round(genome_pop_size / 40.0, 4)
        print("\nEvaluating {} genomes in generation {}...".format(genome_pop_size, self.pop.generation_counter))
        print_str = "\r[{:40}] {}/{} Genomes".format("", genome_eval_counter, genome_pop_size)
        sys.stdout.write(print_str + '\n')

        # Create list of all generated genomes in this generation (genome population).
        generated_genomes = []

        # Evaluate each blueprint independent of its species by building 'genomes_per_bp' genomes and averaging out
        # and assigning the resulting fitness
        for blueprint in self.pop.blueprints.values():
            # Get the species ids of all species present in the blueprint currently evaluated
            bp_module_species = blueprint.get_species()

            for _ in range(self.genomes_per_bp):
                # Assemble genome by first uniform randomly choosing a specific module from the species that the
                # blueprint nodes are referring to.
                bp_assigned_modules = dict()
                for i in bp_module_species:
                    chosen_module_id = random.choice(self.pop.mod_species[i])
                    bp_assigned_modules[i] = self.pop.modules[chosen_module_id]

                try:
                    # Create genome, using the specific blueprint, a dict of modules for each species, the configured
                    # output layers and input shape as well as the current generation
                    genome_id, genome = self.enc.create_genome(blueprint,
                                                               bp_assigned_modules,
                                                               self.output_layers,
                                                               self.input_shape,
                                                               self.pop.generation_counter)

                except ValueError:
                    # Catching build value error, occuring when the supplied layers and parameters do not result in a
                    # valid TF model. See warning string.
                    bp_id = blueprint.get_id()
                    mod_spec_to_id = dict()
                    for spec, mod in bp_assigned_modules.items():
                        mod_spec_to_id[spec] = mod.get_id()
                    logging.warning(
                        f"LamaCoDeepNEAT tried combining the Blueprint ID {bp_id} with the module assignment "
                        f"{mod_spec_to_id}, resulting in an invalid neural network model. Population in this "
                        f"generation is smaller.")

                else:
                    # Append generated genome to list of all generated genomes
                    # and link genome with its origin blueprint.
                    generated_genomes.append(genome)

        # Evaluate in parallel all generated genomes.
        genomes_pool = multiprocessing.pool.ThreadPool(multiprocessing.cpu_count())  # multiprocessing.cpu_count()
        genomes_fitnesses_flopss = genomes_pool.map(environment.eval_genome_fitness, generated_genomes)
        genome_eval_counter = 0

        bp_fitnesses_flops_in_genomes = dict()

        # Create container containing module and list of fitness of genomes which were created from given module.
        # {mod_id : [fitness]}
        mod_fitnesses_weights_in_genomes = dict()

        # Add evaluated genome fitnesses and sizes to all objects
        for genome, genome_fitness_flops in zip(generated_genomes, genomes_fitnesses_flopss):
            # Set fitness and size
            genome_fitness = genome_fitness_flops[0]
            genome_flops = genome_fitness_flops[1]
            genome.set_fitness(genome_fitness)
            genome.set_flops(genome_flops)
            genome_weights = dict()

            # Weights inheritance
            # Add weights to modules used for the creation of the genome
            layer_index = 0
            for topology_level in genome.blueprint.graph_topology[1:]:
                for node in topology_level:
                    curr_module = genome.bp_assigned_modules[genome.blueprint.node_species[node]]
                    curr_module_id = curr_module.get_id()
                    layer = genome.get_model().layers[layer_index]
                    # Try to pass weight from genome model layer back to used modules.
                    weights_inherited = False
                    # Until weights are inherited
                    while weights_inherited is False:
                        # Go to conv2d layer in genome
                        while layer.__class__.__name__ != "Conv2D" and layer_index < len(genome.get_model().layers):
                            layer_index += 1
                            layer = genome.get_model().layers[layer_index]
                        if layer_index < len(genome.get_model().layers):
                            layer_weights_shape = layer.get_weights()[0].shape
                            # Check if layer and module shapes are the same
                            # If it is true, pass the weights and go to next module
                            # If it is false, continue at looking for the right layer
                            if layer_weights_shape[0] == curr_module.kernel_size and \
                                    layer_weights_shape[1] == curr_module.kernel_size and \
                                    layer_weights_shape[3] == curr_module.filters:
                                weights_inherited = True
                                if curr_module_id not in genome_weights.keys():
                                    genome_weights[curr_module_id] = []
                                if weights_not_nan(layer.get_weights()):
                                    genome_weights[curr_module_id].append(np.copy(layer.get_weights()))
                                else:
                                    genome_weights[curr_module_id].append(None)
                                layer_index += 1
                        # If we looked through the whole genome model without right layer, pass to module no weights,
                        # which would not reset the weights (the old weights will stay)
                        else:
                            weights_inherited = True
                            if curr_module_id not in genome_weights.keys():
                                genome_weights[curr_module_id] = []
                            genome_weights[curr_module_id].append([])

            # Print population evaluation
            genome_eval_counter += 1
            genome_id = genome.get_id()
            progress_mult = int(round(genome_eval_counter / genome_eval_counter_div, 4))
            print_str = "\r[{:40}] {}/{} Genomes | Genome ID {} achieved fitness of {} and FLOPs {}".format(
                "=" * progress_mult,
                genome_eval_counter,
                genome_pop_size,
                genome_id,
                genome_fitness,
                genome_flops
            )
            sys.stdout.write(print_str + '\n')

            # Add newline after status update when debugging
            if logging.level_debug():
                print("")

            # Assign fitness and flops to blueprints
            assigned_bp_id = genome.blueprint.get_id()
            if assigned_bp_id not in bp_fitnesses_flops_in_genomes.keys():
                bp_fitnesses_flops_in_genomes[assigned_bp_id] = []
            bp_fitnesses_flops_in_genomes[assigned_bp_id].append((genome_fitness, genome_flops))

            # To modules fitness and weights
            for assigned_mod in genome.bp_assigned_modules.values():
                module_id = assigned_mod.get_id()
                for weights in genome_weights[module_id]:
                    if weights is not None:
                        if module_id not in mod_fitnesses_weights_in_genomes.keys():
                            mod_fitnesses_weights_in_genomes[module_id] = []
                        mod_fitnesses_weights_in_genomes[module_id].append((genome_fitness, weights))

            # Register genome as new best if it exhibits better fitness than the previous best
            if self.pop.best_fitness is None or self.pop.best_fitness == 0 or genome_fitness < self.pop.best_fitness:
                self.pop.best_genome = genome
                self.pop.best_fitness = genome_fitness

        # Reset models, counters, layers, etc including in the GPU to avoid memory clutter from old models as
        # most likely only limited gpu memory is available
        tf.keras.backend.clear_session()

        # MOLamaCDN cloning phase
        for bp_id, bp_pair_list in bp_fitnesses_flops_in_genomes.items():
            # Find species of the current blueprint
            for spec_id, spec_bp_ids in self.pop.bp_species.items():
                if bp_id in spec_bp_ids:
                    bp_species_id = spec_id
                    break
            # Create new individual blueprints as copy of current blueprint with fitness and size gained
            # in each genome. Put them in the same species.
            for bp_pair in bp_pair_list:
                self.enc.bp_id_counter += 1
                self.pop.blueprints[self.enc.bp_id_counter] = copy.deepcopy(self.pop.blueprints[bp_id])
                self.pop.blueprints[self.enc.bp_id_counter].set_fitness(bp_pair[0])
                self.pop.blueprints[self.enc.bp_id_counter].set_flops(bp_pair[1])
                self.pop.blueprints[self.enc.bp_id_counter].set_siblings(len(bp_pair_list))
                self.pop.bp_species[bp_species_id].append(self.enc.bp_id_counter)
            # Delete the current blueprint from population and its species
            del self.pop.blueprints[bp_id]
            self.pop.bp_species[bp_species_id].remove(bp_id)

        for mod_id, mod_pair_list in mod_fitnesses_weights_in_genomes.items():
            # Find species of the current module
            for spec_id, spec_mod_ids in self.pop.mod_species.items():
                if mod_id in spec_mod_ids:
                    mod_species_id = spec_id
                    break
            # Create new individual module as copy of current module with fitness and size gained
            # in each genome. Put them in the same species.
            for mod_pair in mod_pair_list:
                self.enc.mod_id_counter += 1
                self.pop.modules[self.enc.mod_id_counter] = copy.deepcopy(self.pop.modules[mod_id])
                self.pop.modules[self.enc.mod_id_counter].set_fitness(mod_pair[0])
                self.pop.modules[self.enc.mod_id_counter].set_weights(mod_pair[1])
                self.pop.modules[self.enc.mod_id_counter].set_siblings(len(mod_pair_list))
                self.pop.mod_species[mod_species_id].append(self.enc.mod_id_counter)
            # Delete the current module from population and its species
            del self.pop.modules[mod_id]
            self.pop.mod_species[mod_species_id].remove(mod_id)

        # Calculate average fitness of each module species and add to pop.mod_species_fitness_history
        for spec_id, spec_mod_ids in self.pop.mod_species.items():
            spec_fitness_list = [self.pop.modules[mod_id].get_fitness() for mod_id in spec_mod_ids]
            spec_avg_fitness = round(statistics.mean(spec_fitness_list), 4)
            if spec_id in self.pop.mod_species_fitness_history:
                self.pop.mod_species_fitness_history[spec_id][self.pop.generation_counter] = spec_avg_fitness
            else:
                self.pop.mod_species_fitness_history[spec_id] = {self.pop.generation_counter: spec_avg_fitness}

        # Calculate average fitness of each blueprint species and add to pop.bp_species_fitness_history
        for spec_id, spec_bp_ids in self.pop.bp_species.items():
            spec_fitness_list = [self.pop.blueprints[bp_id].get_fitness() for bp_id in spec_bp_ids]
            spec_avg_fitness = round(statistics.mean(spec_fitness_list), 4)
            if spec_id in self.pop.bp_species_fitness_history:
                self.pop.bp_species_fitness_history[spec_id][self.pop.generation_counter] = spec_avg_fitness
            else:
                self.pop.bp_species_fitness_history[spec_id] = {self.pop.generation_counter: spec_avg_fitness}

        return self.pop.generation_counter, self.pop.best_fitness

    def evolve_population(self, save_dir_path) -> bool:
        """
        Evolve the population according to the CoDeepNEAT algorithm by first selecting all modules and blueprints, which
        eliminates low performing members and species and determines members elligible for being parents of offspring.
        Then evolve the module population by creating mutations or crossovers of elligible parents. Evolve the blueprint
        population by adding nodes or connections, removing nodes or connections, mutating module species or optimizers,
        etc. Subsequently speciate the module and blueprint population according to the chosen speciation method, which
        clusters the modules and blueprints according to their similarity.
        @return: bool flag, indicating ig population went extinct during evolution
        """
        #### Select Modules ####
        if self.mod_spec_type == 'basic':
            mod_spec_offspring, mod_spec_parents, mod_spec_extinct = self._select_modules_basic()
        elif self.mod_spec_type == 'param-distance-fixed':
            mod_spec_offspring, mod_spec_parents, mod_spec_extinct = self._select_modules_param_distance_fixed()
        elif self.mod_spec_type == 'param-distance-dynamic':
            mod_spec_offspring, mod_spec_parents, mod_spec_extinct = self._select_modules_param_distance_dynamic()
        elif self.mod_spec_type == 'multiobjective':
            mod_spec_offspring, mod_spec_parents = self._select_modules_multiobjective()
        else:
            raise RuntimeError(f"Module speciation type '{self.mod_spec_type}' not yet implemented")

        # If population went extinct abort evolution and return True
        if len(self.pop.mod_species) == 0:
            return True

        #### Select Blueprints ####
        if self.bp_spec_type == 'basic':
            bp_spec_offspring, bp_spec_parents = self._select_blueprints_basic()
        elif self.bp_spec_type == 'gene-overlap-fixed':
            bp_spec_offspring, bp_spec_parents = self._select_blueprints_gene_overlap_fixed()
        elif self.bp_spec_type == 'gene-overlap-dynamic':
            bp_spec_offspring, bp_spec_parents = self._select_blueprints_gene_overlap_dynamic()
        elif self.bp_spec_type == 'multiobjective':
            bp_spec_offspring, bp_spec_parents = self._select_blueprints_multiobjective()
        else:
            raise RuntimeError(f"Blueprint speciation type '{self.bp_spec_type}' not yet implemented")

        # If population went extinct abort evolution and return True
        if len(self.pop.bp_species) == 0:
            return True

        #### Evolve Modules ####
        new_module_ids = self._evolve_modules(mod_spec_offspring, mod_spec_parents, save_dir_path)

        #### Evolve Blueprints ####
        new_bp_ids, bp_spec_parents = self._evolve_blueprints(bp_spec_offspring, bp_spec_parents, save_dir_path)

        #### Speciate Modules ####
        if self.mod_spec_type == 'basic':
            self._speciate_modules_basic(mod_spec_parents, new_module_ids)
        elif self.mod_spec_type == 'param-distance-fixed':
            self._speciate_modules_param_distance_fixed(mod_spec_parents, new_module_ids)
        elif self.mod_spec_type == 'param-distance-dynamic' or self.mod_spec_type == 'multiobjective':
            self._speciate_modules_param_distance_dynamic(mod_spec_parents, new_module_ids)
        else:
            raise RuntimeError(f"Module speciation type '{self.mod_spec_type}' not yet implemented")

        #### Speciate Blueprints ####
        if self.bp_spec_type == 'basic' or self.bp_spec_type == 'multiobjective':
            self._speciate_blueprints_basic(bp_spec_parents, new_bp_ids)
        elif self.bp_spec_type == 'gene-overlap-fixed':
            self._speciate_blueprints_gene_overlap_fixed(bp_spec_parents, new_bp_ids)
        elif self.bp_spec_type == 'gene-overlap-dynamic':
            self._speciate_blueprints_gene_overlap_dynamic(bp_spec_parents, new_bp_ids)
        else:
            raise RuntimeError(f"Blueprint speciation type '{self.bp_spec_type}' not yet implemented")

        #### Return ####
        # Reset individuals attributes needed for multiobjective sort for next generation
        for bp in self.pop.blueprints.values():
            bp.dominates = []
            bp.rank = -1
            bp.dom_counter = 0
            bp.distance = 0
        for mod in self.pop.modules.values():
            mod.dominates = []
            mod.rank = -1
            mod.dom_counter = 0
            mod.distance = 0

        # Adjust generation counter and return False, signalling that the population has not gone extinct
        self.pop.generation_counter += 1
        return False


def weights_not_nan(weights):
    """
    Check, whether the network weights contains NaN value (not a number).
    """
    return not (np.all(np.isnan(weights[0])) and np.all(np.isnan(weights[1])))
