from tfne.algorithms.codeepneat._codeepneat_selection_mod import CoDeepNEATSelectionMOD

from typing import Union


class LamaCoDeepNEATSelectionMOD(CoDeepNEATSelectionMOD):
    """
    LamaCoDeepNEATSelectionMOD inherits from CoDeepNEATSelectionMOD.
    Following methods are new:
    * _select_blueprints_multiobjective
    """

    def _select_modules_multiobjective(self) -> ({Union[int, str]: int}, {int: int}):
        """"""
        ### Rebase Species Representative ###
        # Rechoose the representative of each species as the best and existing module of the species
        # representatives. Begin the rebasing of species representatives from the oldest to the newest species.
        all_spec_repr_ids = set(self.pop.mod_species_repr.values())
        for spec_id, spec_repr_id in self.pop.mod_species_repr.items():
            # Determine the module ids of all other species representatives and create a sorted list of the modules
            # in the current species according to their fitness
            other_spec_repr_ids = all_spec_repr_ids - {spec_repr_id}

            # Traverse each module id in the sorted module id list beginning with the best. Determine the distance
            # to other species representative module ids and if the distance to all other species representatives is
            # higher than the specified minimum distance for a new species, set the module as the new
            # representative.
            spec_mod_ids_sorted = sorted(self.pop.mod_species[spec_id],
                                         key=lambda x: (self.pop.modules[x].get_rank(),
                                                        -self.pop.modules[x].get_crowd_distance()))
            for mod_id in spec_mod_ids_sorted:
                if mod_id == spec_repr_id:
                    # Best species module already representative. Abort search.
                    break
                module = self.pop.modules[mod_id]
                distance_to_other_spec_repr = [module.get_distance(self.pop.modules[other_mod_id])
                                               for other_mod_id in other_spec_repr_ids
                                               if other_mod_id in self.pop.modules.keys()]
                if all(distance >= self.mod_spec_distance for distance in distance_to_other_spec_repr) or \
                        spec_repr_id not in self.pop.modules.keys():
                    # New best species representative found. Set as representative and abort search
                    self.pop.mod_species_repr[spec_id] = mod_id
                    break

        ### Generational Parent Determination ###
        # Determine whole population of modules as potential parents as long as NSGA-II selection works differently.
        # We ignore reproduction threshold parameter and we implement elitism later.
        mod_spec_parents = self.pop.mod_species.copy()

        #### Offspring Size Calculation ####
        # Determine the amount of offspring for each species as well as the amount of reinitialized modules, in case
        # this option is activated. Each species is assigned offspring according to its species share of the total
        # fitness, though minimum offspring constraints are considered. Preprocess by determining the sum of all
        # average fitness and removing the extinct species from the species order
        total_avg_fitness = 0
        for fitness_history in self.pop.mod_species_fitness_history.values():
            total_avg_fitness += fitness_history[self.pop.generation_counter]

        # Determine the amount of offspring to be reinitialized as the fitness share of the total fitness by the extinct
        # species
        mod_spec_offspring = dict()
        available_mod_pop = self.mod_pop_size

        # Work through each species in order (from least to most fit) and determine the intended size as the species
        # fitness share of the total fitness of the remaining species, applied to the remaining population slots.
        # Assign offspring under the consideration of the minimal offspring constraint and then decrease the total
        # fitness and the remaining population slots.
        mod_species_ordered = sorted(self.pop.mod_species.keys(),
                                     key=lambda x: self.pop.mod_species_fitness_history[x][self.pop.generation_counter],
                                     reverse=True)
        for spec_id in mod_species_ordered:
            spec_fitness = self.pop.mod_species_fitness_history[spec_id][self.pop.generation_counter]
            spec_fitness_share = spec_fitness / total_avg_fitness
            spec_intended_size = int(round(spec_fitness_share * available_mod_pop))

            mod_spec_offspring[spec_id] = spec_intended_size
            available_mod_pop -= spec_intended_size
            total_avg_fitness -= spec_fitness

        # Return
        # mod_spec_offspring {int: int} associating species id with amount of offspring
        # mod_spec_parents {int: [int]} associating species id with list of potential parent ids for species
        return mod_spec_offspring, mod_spec_parents
