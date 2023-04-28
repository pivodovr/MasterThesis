from tfne.algorithms.codeepneat._codeepneat_evolution_bp import CoDeepNEATEvolutionBP
from tfne.algorithms.lamacodeepneat.nsgaII import *


class LamaCoDeepNEATEvolutionBP(CoDeepNEATEvolutionBP):
    """
    LamaCoDeepNEATEvolutionBP inherits from CoDeepNEATEvolutionBP.
    Following methods are overwritten:
    * _evolve_blueprints
    """
    def _evolve_blueprints(self, bp_spec_offspring, bp_spec_parents, save_dir_path) -> [int]:
        #### Evolve Blueprints ####
        # Create container for new blueprints that will be speciated in a later function
        new_blueprint_ids = list()
        new_population = dict()

        # Calculate the brackets for a random float to fall into in order to choose a specific evolutionary method
        bp_mutation_add_node_bracket = self.bp_mutation_add_conn_prob + self.bp_mutation_add_node_prob
        bp_mutation_rem_conn_bracket = bp_mutation_add_node_bracket + self.bp_mutation_rem_conn_prob
        bp_mutation_rem_node_bracket = bp_mutation_rem_conn_bracket + self.bp_mutation_rem_node_prob
        bp_mutation_node_spec_bracket = bp_mutation_rem_node_bracket + self.bp_mutation_node_spec_prob
        bp_mutation_optimizer_bracket = bp_mutation_node_spec_bracket + self.bp_mutation_optimizer_prob

        # Traverse through each species and create according amount of offspring as determined prior during selection
        for spec_id, species_offspring in bp_spec_offspring.items():
            if spec_id == 'reinit':
                continue

            # Sort parents in NSGA-II way.
            fronts = fast_non_dominated_sort(bp_spec_parents[spec_id], self.pop.blueprints)

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
                    crowding_distance(front, self.pop.blueprints)
                    # Append all blueprints from front to mating pool.
                    for fid in front:
                        # Check if it's not already in mating pool
                        if self.pop.blueprints[fid].get_id() not in used_ids:
                            mating_pool_ids.append(fid)
                            used_ids.add(self.pop.blueprints[fid].get_id())
                            self.pop.blueprints[fid].set_id(fid)
                            x -= 1
                        # If one of the condition is false, then remove blueprint completely and continue
                        else:
                            del self.pop.blueprints[fid]
                            self.pop.bp_species[spec_id].remove(fid)
                            front.remove(fid)
                else:
                    # Compute crowding distance as third condition
                    crowding_distance(front, self.pop.blueprints)
                    # Order module ids from the biggest num of siblings to the smallest, then by crowding distance
                    bp_siblings_dist = [[bpid,
                                        self.pop.blueprints[bpid].get_siblings(),
                                        self.pop.blueprints[bpid].get_crowd_distance()] for bpid in front]
                    sorted_bp_siblings_dist = sorted(bp_siblings_dist, key=lambda l: (l[1], l[2]), reverse=True)
                    # Take x ordered individuals from the last front
                    for bp_sib_dist in sorted_bp_siblings_dist:
                        if x <= 0:
                            break
                        # Check if it's not already in mating pool
                        fid = bp_sib_dist[0]
                        if self.pop.blueprints[fid].get_id() not in used_ids:
                            mating_pool_ids.append(fid)
                            used_ids.add(self.pop.blueprints[fid].get_id())
                            self.pop.blueprints[fid].set_id(fid)
                            x -= 1
                        # If one of the condition is false, then remove blueprint completely and continue
                        else:
                            del self.pop.blueprints[fid]
                            self.pop.bp_species[spec_id].remove(fid)
                            sorted_bp_siblings_dist.remove(bp_sib_dist)

            # Extinction of this blueprint specie
            if len(mating_pool_ids) == 0:
                for bp_id in self.pop.bp_species[spec_id]:
                    del self.pop.blueprints[bp_id]
                del self.pop.bp_species[spec_id]
                del self.pop.bp_species_repr[spec_id]
                del self.pop.bp_species_fitness_history[spec_id]

            # Save blueprints from mating pool into new population.
            for pid in mating_pool_ids:
                new_population[pid] = self.pop.blueprints[pid]

            for _ in range(species_offspring - species_offspring // 2):
                # Choose random float value determining specific evolutionary method to evolve the chosen blueprint.
                random_choice = random.random()
                # Choose random parents for binary tournament selection.
                if len(mating_pool_ids) <= 1:
                    parent_blueprint = self.pop.blueprints[mating_pool_ids[0]]
                else:
                    # Tournament selection with comparing rank, secondary crowding distance.
                    parent_blueprint = tournament_selection(mating_pool_ids, self.pop.blueprints)

                # If randomly chosen parent for mutation contains extinct species, force node module species mutation
                if random_choice < self.bp_mutation_add_conn_prob:
                    ## Create new blueprint by adding connection ##
                    max_degree_of_mutation = random.uniform(1e-323, self.bp_max_mutation)
                    new_bp_id, new_bp = self._create_mutated_blueprint_add_conn(parent_blueprint,
                                                                                max_degree_of_mutation)

                elif random_choice < bp_mutation_add_node_bracket:
                    ## Create new blueprint by adding node ##
                    max_degree_of_mutation = random.uniform(1e-323, self.bp_max_mutation)
                    new_bp_id, new_bp = self._create_mutated_blueprint_add_node(parent_blueprint,
                                                                                max_degree_of_mutation)

                elif random_choice < bp_mutation_rem_conn_bracket:
                    ## Create new blueprint by removing connection ##
                    max_degree_of_mutation = random.uniform(1e-323, self.bp_max_mutation)
                    new_bp_id, new_bp = self._create_mutated_blueprint_rem_conn(parent_blueprint,
                                                                                max_degree_of_mutation)

                elif random_choice < bp_mutation_rem_node_bracket:
                    ## Create new blueprint by removing node ##
                    max_degree_of_mutation = random.uniform(1e-323, self.bp_max_mutation)
                    new_bp_id, new_bp = self._create_mutated_blueprint_rem_node(parent_blueprint,
                                                                                max_degree_of_mutation)

                elif random_choice < bp_mutation_node_spec_bracket:
                    ## Create new blueprint by mutating species in nodes ##
                    max_degree_of_mutation = random.uniform(1e-323, self.bp_max_mutation)
                    new_bp_id, new_bp = self._create_mutated_blueprint_node_spec(parent_blueprint,
                                                                                 max_degree_of_mutation,
                                                                                 {})
                    print(new_bp)

                elif random_choice < bp_mutation_optimizer_bracket:
                    ## Create new blueprint by mutating the associated optimizer ##
                    new_bp_id, new_bp = self._create_mutated_blueprint_optimizer(parent_blueprint)

                else:  # random_choice < bp_crossover_bracket:
                    ## Create new blueprint through crossover ##
                    # Try tournament selecting another parent blueprint and checking that other parnet bp for extinct
                    # node module species. If species has only 1 member or other blueprint has extinct node module
                    # species fail and create a new blueprint by adding a node to the original parent blueprint.
                    if len(mating_pool_ids) <= 1:
                        max_degree_of_mutation = random.uniform(1e-323, self.bp_max_mutation)
                        new_bp_id, new_bp = self._create_mutated_blueprint_add_node(parent_blueprint,
                                                                                    max_degree_of_mutation)
                    else:
                        # Tournament selection with comparing rank, secondary crowding distance.
                        mating_pool_ids_smaller = mating_pool_ids.copy()
                        mating_pool_ids_smaller.remove(parent_blueprint.get_id())
                        if len(mating_pool_ids_smaller) == 1:
                            other_bp = self.pop.blueprints[mating_pool_ids[0]]
                        else:
                            other_bp = tournament_selection(mating_pool_ids_smaller, self.pop.blueprints)

                        # Create crossover blueprint if second valid blueprint was found
                        new_bp_id, new_bp = self._create_crossed_over_blueprint(parent_blueprint,
                                                                                other_bp)

                # Add newly created blueprint to the new population and to the list of bps that have to be speciated
                new_blueprint_ids.append(new_bp_id)
                new_population[new_bp_id] = new_bp

        # Delete an old population and save the new one.
        self.pop.blueprints = new_population.copy()

        # Update self.pop.mod_species
        for specie, blueprints in self.pop.bp_species.items():
            self.pop.bp_species[specie] = [bp for bp in blueprints if bp in self.pop.blueprints.keys()]


        ### Rebase Species Representative ###
        # Rechoose the representative of each species as the best and existing blueprint of the species
        # representatives. Begin the rebasing of species representatives from the oldest to the newest species.
        all_spec_repr_ids = set(self.pop.bp_species_repr.values())
        for spec_id, spec_repr_id in self.pop.bp_species_repr.items():
            # Determine the blueprint ids of all other species representatives and create a sorted list of the
            # blueprints in the current species according to their fitness
            other_spec_repr_ids = all_spec_repr_ids - {spec_repr_id}

            # Find the best representative blueprint with lowest rank (secondary with lowest crowding distance)
            spec_bp_ids_sorted = sorted(self.pop.bp_species[spec_id],
                                        key=lambda x: (self.pop.blueprints[x].get_rank(),
                                                       -self.pop.blueprints[x].get_crowd_distance()))
            for bp_id in spec_bp_ids_sorted:
                if bp_id == spec_repr_id:
                    # Best species blueprint already representative. Abort search.
                    break
                blueprint = self.pop.blueprints[bp_id]
                distance_to_other_spec_repr = [blueprint.calculate_gene_distance(self.pop.blueprints[other_bp_id])
                                               for other_bp_id in other_spec_repr_ids if
                                               other_bp_id in self.pop.blueprints.keys()]

                if all(distance >= self.bp_spec_distance for distance in distance_to_other_spec_repr) or \
                        spec_repr_id not in self.pop.blueprints.keys():
                    # New best species representative found. Set as representative and abort search
                    self.pop.bp_species_repr[spec_id] = bp_id
                    break

        #### Reinitialize Blueprints ####
        # Initialize predetermined number of new blueprints as species went extinct and reinitialization is activated
        if 'reinit' in bp_spec_offspring:
            available_mod_species = tuple(self.pop.mod_species.keys())
            for _ in range(bp_spec_offspring['reinit']):
                # Determine the module species of the initial (and only) node
                initial_node_species = random.choice(available_mod_species)

                # Initialize a new blueprint with minimal graph only using initial node species
                new_bp_id, new_bp = self._create_initial_blueprint(initial_node_species)

                # Add newly created blueprint to the bp container and to the list of bps that have to be speciated
                self.pop.blueprints[new_bp_id] = new_bp
                new_blueprint_ids.append(new_bp_id)

        # Return the list of new blueprint ids for later speciation as well as the updated list of blueprint parents
        # for each species
        return new_blueprint_ids, bp_spec_parents
