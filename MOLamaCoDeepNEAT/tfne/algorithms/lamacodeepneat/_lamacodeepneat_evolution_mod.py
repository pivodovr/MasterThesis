from tfne.algorithms.codeepneat._codeepneat_evolution_mod import CoDeepNEATEvolutionMOD
from tfne.algorithms.lamacodeepneat.nsgaII import *


class LamaCoDeepNEATEvolutionMOD(CoDeepNEATEvolutionMOD):
    """
    LamaCoDeepNEATEvolutionMOD inherits from CoDeepNEATEvolutionMOD.
    Following methods are overwritten:
    * _evolve_modules
    """

    def _evolve_modules(self, mod_spec_offspring, mod_spec_parents, save_dir_path) -> [int]:
        """"""
        # Create container for new modules that will be speciated in a later function
        new_module_ids = list()
        new_population = dict()

        #### Evolve Modules ####
        # Traverse through each species and create according amount of offspring as determined prior during selection
        for spec_id, species_offspring in mod_spec_offspring.items():
            if spec_id == 'reinit':
                continue

            # Sort parents in NSGA-II way.
            fronts = fast_non_dominated_sort(mod_spec_parents[spec_id], self.pop.modules)

            # Move x best parents straight to new population - here is implemented elitism. Then use them for mating.
            # Create container for best parents who will be part of new population and will mate.
            mating_pool_ids = list()
            used_ids = set()
            x = species_offspring // 2
            if x == 0:
                x = 1
            for front in fronts:
                if x == 0:
                    break
                if len(front) <= x:
                    # Compute crowding distance for later usage
                    # and append all modules from front to mating pool.
                    crowding_distance(front, self.pop.modules)
                    for fid in front:
                        if self.pop.modules[fid].get_id() not in used_ids:
                            mating_pool_ids.append(fid)
                            used_ids.add(self.pop.modules[fid].get_id())
                            self.pop.modules[fid].set_id(fid)
                            x -= 1
                        else:
                            del self.pop.modules[fid]
                            self.pop.mod_species[spec_id].remove(fid)
                            front.remove(fid)
                else:
                    # Select those with higher number of siblings, secondary use crowding distance
                    # Order module ids from the biggest num of siblings to the smallest
                    crowding_distance(front, self.pop.modules)
                    mod_siblings_dist = [[modid,
                                         self.pop.modules[modid].get_siblings(),
                                         self.pop.modules[modid].get_crowd_distance()] for modid in front]
                    sorted_mod_siblings_dist = sorted(mod_siblings_dist, key=lambda l: (l[1], l[2]), reverse=True)
                    # Take x ordered individuals from the last front
                    for mod_sib_dist in sorted_mod_siblings_dist:
                        if x <= 0:
                            break
                        # Check if it's not already in mating pool
                        fid = mod_sib_dist[0]
                        if self.pop.modules[fid].get_id() not in used_ids:
                            mating_pool_ids.append(fid)
                            used_ids.add(self.pop.modules[fid].get_id())
                            self.pop.modules[fid].set_id(fid)
                            x -= 1
                        # If one of the condition is false, then remove module completely and continue
                        else:
                            del self.pop.modules[fid]
                            self.pop.mod_species[spec_id].remove(fid)
                            sorted_mod_siblings_dist.remove(mod_sib_dist)

            # Save modules from mating pool into new population.
            for pid in mating_pool_ids:
                new_population[pid] = self.pop.modules[pid]

            for _ in range(species_offspring - species_offspring // 2):
                # Choose randomly between mutation or crossover of module
                if random.random() < self.mod_mutation_prob:
                    ## Create new module through mutation ##
                    # Determine random maximum degree of mutation > 0 and randomly choose a parent module from the
                    # remaining modules of the current species. Create a mutation by letting the module internal
                    # function take care of this.
                    max_degree_of_mutation = random.uniform(1e-323, self.mod_max_mutation)
                    if len(mating_pool_ids) <= 1:
                        parent_module = self.pop.modules[random.choice(mating_pool_ids)]
                    else:
                        # Tournament selection with comparing rank, secondary crowding distance.
                        parent_module = tournament_selection(mating_pool_ids, self.pop.modules)
                    new_mod_id, new_mod = self.enc.create_mutated_module(parent_module, max_degree_of_mutation)

                else:  # random.random() < self.mod_mutation_prob + self.mod_crossover_prob
                    ## Create new module through crossover ##
                    # Determine if species has at least 2 modules as required for crossover
                    if len(mating_pool_ids) >= 2:
                        # Determine the 2 parent modules used for crossover
                        if len(mating_pool_ids) > 2:
                            parent_module_1 = tournament_selection(mating_pool_ids, self.pop.modules)
                            mating_pool_ids_smaller = mating_pool_ids.copy()
                            mating_pool_ids_smaller.remove(parent_module_1.get_id())
                            parent_module_2 = tournament_selection(mating_pool_ids_smaller, self.pop.modules)

                        if len(mating_pool_ids) == 2:
                            parent_module_1 = self.pop.modules[mating_pool_ids[0]]
                            parent_module_2 = self.pop.modules[mating_pool_ids[1]]
                        # Randomly determine the maximum degree of mutation > 0 and let the modules internal function
                        # create a crossover
                        max_degree_of_mutation = random.uniform(1e-323, self.mod_max_mutation)
                        new_mod_id, new_mod = self.enc.create_crossover_module(parent_module_1,
                                                                               parent_module_2,
                                                                               max_degree_of_mutation)

                    else:
                        # As species does not have enough modules for crossover, perform a mutation on the remaining
                        # module
                        max_degree_of_mutation = random.uniform(1e-323, self.mod_max_mutation)
                        parent_module = self.pop.modules[mating_pool_ids[0]]
                        new_mod_id, new_mod = self.enc.create_mutated_module(parent_module, max_degree_of_mutation)

                # Add newly created module to the module container and to the list of modules that have to be speciated
                new_module_ids.append(new_mod_id)
                new_population[new_mod_id] = new_mod

        # Delete an old population and save the new one.
        self.pop.modules = new_population.copy()

        # Update self.pop.mod_species
        for specie, modules in self.pop.mod_species.items():
            self.pop.mod_species[specie] = [mod for mod in modules if mod in self.pop.modules.keys()]


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



        #### Reinitialize Modules ####
        if 'reinit' in mod_spec_offspring:
            # Initialize predetermined number of new modules as species went extinct and reinitialization is activated
            for i in range(mod_spec_offspring['reinit']):
                # Decide on for which species a new module is added (uniformly distributed)
                chosen_species = i % len(self.available_modules)

                # Determine type and the associated config parameters of chosen species and initialize a module with it
                mod_type = self.available_modules[chosen_species]
                mod_config_params = self.available_mod_params[mod_type]
                new_mod_id, new_mod = self.enc.create_initial_module(mod_type=mod_type,
                                                                     config_params=mod_config_params)

                # Add newly created module to the module container and to the list of modules that have to be speciated
                self.pop.modules[new_mod_id] = new_mod
                new_module_ids.append(new_mod_id)

        # Return the list of new module ids for later speciation
        return new_module_ids
