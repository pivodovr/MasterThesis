import json
import random
import statistics
import sys
import multiprocessing
from absl import logging

import numpy as np
import tensorflow as tf

import tfne
from tfne.algorithms.codeepneat.codeepneat import CoDeepNEAT
from tfne.encodings.lamacodeepneat.lamacodeepneat_encoding import LamaCoDeepNEATEncoding


class LamaCoDeepNEAT(CoDeepNEAT):
    """
    The LamaCoDeepNEAT inherits from the CoDeepNEAT.
    Following methods are overwritten:
    * __init__
    * evaluate_population
    """

    def __init__(self, config, initial_state_file_path=None):
        """
        Initialize the LamaCoDeepNEAT algorithm by processing and sanity checking the supplied configuration, which
        saves all algorithm config parameters as instance variables. Then initialize the LamaCoDeepNEAT encoding
        and population. Alternatively, if a backup-state is supplied, reinitialize the encoding and population with the
        state present in that backup.
        @param config: ConfigParser instance holding all documentation specified sections of the LamaCDN algorithm
        @param initial_state_file_path: string file path to a state backup that is to be resumed
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

            # Initialize and register an associated LamaCoDeepNEAT encoding and population outfitted with the saved
            # state
            self.enc = tfne.deserialization.load_encoding(serialized_encoding=saved_state['encoding'],
                                                          dtype=self.dtype)
            self.pop = tfne.deserialization.load_population(serialized_population=saved_state['population'],
                                                            dtype=self.dtype,
                                                            module_config_params=self.available_mod_params)
        else:
            # Initialize and register a blank associated LamaCoDeepNEAT encoding and CoDeepNEAT population
            self.enc = LamaCoDeepNEATEncoding(dtype=self.dtype)
            self.pop = tfne.populations.CoDeepNEATPopulation()

    def evaluate_population(self, environment, save_dir_path) -> (int, float):
        """
        Evaluate the population by building the specified amount of genomes from each blueprint, all having randomly
        assigned specific modules for the inherent blueprint module species. Set the evaluated fitness of each blueprint
        and each module as the average fitness achieved by all genomes in which the respective member was invovled in.
        Return the generational counter as well as the achieved fitness of the best genome.
        Implementation is almost the same as in CoDeepNEAT. The difference is in support for genome weights inheritance.
        @param environment: instance of the evaluation environment
        @return: tuple of generation counter and best fitness achieved by best genome
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
                        f"{mod_spec_to_id}, resulting in an invalid neural network model. Setting genome "
                        f"fitness to 0.")

                else:
                    # Append generated genome to list of all generated genomes
                    # and link genome with its origin blueprint.
                    generated_genomes.append(genome)

        # Evaluate in parallel all generated genomes.
        genomes_pool = multiprocessing.pool.ThreadPool(multiprocessing.cpu_count())
        genomes_fitnesses = genomes_pool.map(environment.eval_genome_fitness, generated_genomes)
        genome_eval_counter = 0

        # Create container containing blueprint and list of fitness of genomes which were created from given bp.
        # {bp_id : [fitness]}
        bp_fitnesses_in_genomes = dict()

        # Create container containing module and list of fitness of genomes which were created from given module.
        # {mod_id : [fitness]}
        mod_fitnesses_in_genomes = dict()

        # Create container collecting the weights of the genomes that involve specific modules. Calculate the average
        # weights of the genomes in which a module is involved in later and assign it as the module's weights
        # {mod_id : [[weights, genome fitness]]}
        mod_weights_in_genomes = dict()

        # Add evaluated genome fitnesses to all objects
        for genome, genome_fitness in zip(generated_genomes, genomes_fitnesses):
            # Set fitness
            genome.set_fitness(genome_fitness)

            # Weights inheritance
            # Add weights to modules used for the creation of the genome
            layer_index = 0
            for topology_level in genome.blueprint.graph_topology[1:]:
                for node in topology_level:
                    curr_module = genome.bp_assigned_modules[genome.blueprint.node_species[node]]
                    curr_module_id = curr_module.get_id()
                    layer = genome.get_model().layers[layer_index]
                    # Go to conv2d layer in genome
                    while layer.__class__.__name__ != "Conv2D" and layer_index < len(genome.get_model().layers):
                        layer_index += 1
                        layer = genome.get_model().layers[layer_index]
                    # Check if layer and module shapes are the same
                    # If it is true, pass the weights and go to next module
                    # If it is false, continue at looking for the right layer
                    if layer_index < len(genome.get_model().layers):
                        layer_weights_shape = layer.get_weights()[0].shape
                        if layer_weights_shape[0] == curr_module.kernel_size and \
                                layer_weights_shape[1] == curr_module.kernel_size and \
                                layer_weights_shape[3] == curr_module.filters:
                            # Resulting module weights are from genome with highest fitness
                            if (curr_module_id not in mod_weights_in_genomes.keys() or
                                mod_weights_in_genomes[curr_module_id][1] <= genome_fitness) \
                                    and weights_not_nan(layer.get_weights()):
                                mod_weights_in_genomes[curr_module_id] = [np.copy(layer.get_weights()), genome_fitness]
                            layer_index += 1

            # Print population evaluation
            genome_eval_counter += 1
            genome_id = genome.get_id()
            progress_mult = int(round(genome_eval_counter / genome_eval_counter_div, 4))
            print_str = "\r[{:40}] {}/{} Genomes | Genome ID {} achieved fitness of {}".format(
                "=" * progress_mult,
                genome_eval_counter,
                genome_pop_size,
                genome_id,
                genome_fitness
            )
            sys.stdout.write(print_str + '\n')

            # Add newline after status update when debugging
            if logging.level_debug():
                print("")

            # Assign the genome fitness and train time to the blueprint
            # and all modules used for the creation of the genome.
            # To blueprint
            assigned_bp_id = genome.blueprint.get_id()
            if assigned_bp_id not in bp_fitnesses_in_genomes.keys():
                bp_fitnesses_in_genomes[assigned_bp_id] = []
            bp_fitnesses_in_genomes[assigned_bp_id].append(genome_fitness)

            # To modules
            for assigned_mod in genome.bp_assigned_modules.values():
                module_id = assigned_mod.get_id()
                if module_id not in mod_fitnesses_in_genomes.keys():
                    mod_fitnesses_in_genomes[module_id] = []
                mod_fitnesses_in_genomes[module_id].append(genome_fitness)

            # Register genome as new best if it exhibits better fitness than the previous best
            if self.pop.best_fitness is None or self.pop.best_fitness == 0 or genome_fitness > self.pop.best_fitness:
                self.pop.best_genome = genome
                self.pop.best_fitness = genome_fitness

        # Reset models, counters, layers, etc including in the GPU to avoid memory clutter from old models as
        # most likely only limited gpu memory is available
        tf.keras.backend.clear_session()

        # Average out collected fitness of genomes the blueprint was invovled in. Then assign that average fitness
        # to the blueprint
        for bp_id, bp_fitness_list in bp_fitnesses_in_genomes.items():
            bp_genome_fitness_avg = round(statistics.mean(bp_fitness_list), 4)
            self.pop.blueprints[bp_id].set_fitness(bp_genome_fitness_avg)

        # Average out collected fitness of genomes each module was invovled in. Then assign that average fitness to the
        # module
        for mod_id, mod_fitness_list in mod_fitnesses_in_genomes.items():
            mod_genome_fitness_avg = round(statistics.mean(mod_fitness_list), 4)
            self.pop.modules[mod_id].set_fitness(mod_genome_fitness_avg)

        # Set weights from genome with the highest fitness to the
        # module.
        for mod_id, mod_weights in mod_weights_in_genomes.items():
            self.pop.modules[mod_id].set_weights(mod_weights[0])

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


def weights_not_nan(weights):
    """
    Check, whether the network weights contains NaN value (not a number).
    """
    return not (np.all(np.isnan(weights[0])) and np.all(np.isnan(weights[1])))
