import statistics

from tfne.populations.codeepneat.codeepneat_population import CoDeepNEATPopulation

class LamaCoDeepNEATPopulation(CoDeepNEATPopulation):
    def summarize_population(self):
        """
        Prints the current state of all LamaCoDeepNEAT population variables to stdout in a formatted and clear manner
        """
        # Determine average fitness of all blueprints
        bp_fitness_list = [self.blueprints[bp_id].get_fitness() for bp_id in self.blueprints]
        blueprints_avg_fitness = round(statistics.mean(bp_fitness_list), 4)

        # Determine best id of each blueprint species
        bp_species_best_id = dict()
        for spec_id, spec_bp_ids in self.bp_species.items():
            spec_bp_ids_sorted = sorted(spec_bp_ids, key=lambda x: self.blueprints[x].get_fitness())
            bp_species_best_id[spec_id] = spec_bp_ids_sorted[0]

        # Determine average fitness of all modules
        mod_fitness_list = [self.modules[mod_id].get_fitness() for mod_id in self.modules]
        modules_avg_fitness = round(statistics.mean(mod_fitness_list), 4)

        # Determine best id of each module species
        mod_species_best_id = dict()
        for spec_id, spec_mod_ids in self.mod_species.items():
            spec_mod_ids_sorted = sorted(spec_mod_ids, key=lambda x: self.modules[x].get_fitness())
            mod_species_best_id[spec_id] = spec_mod_ids_sorted[0]

        # Print summary header
        print("\n\n\n\033[1m{}  Population Summary  {}\n\n"
              "Generation: {:>4}  ||  Best Genome Fitness: {:>8}  ||  Avg Blueprint Fitness: {:>8}  ||  "
              "Avg Module Fitness: {:>8}\033[0m\n"
              "Best Genome: {}\n"
              .format('#' * 60,
                      '#' * 60,
                      self.generation_counter,
                      self.best_fitness,
                      blueprints_avg_fitness,
                      modules_avg_fitness,
                      self.best_genome))

        # Print summary of blueprint species
        print("\033[1mBlueprint Species       || Blueprint Species Avg Fitness       || Blueprint Species Size\033[0m")
        for spec_id, spec_fitness_hisotry in self.bp_species_fitness_history.items():
            print("{:>6}                  || {:>8}                            || {:>8}"
                  .format(spec_id,
                          spec_fitness_hisotry[self.generation_counter],
                          len(self.bp_species[spec_id])))
            print(f"Best BP of Species {spec_id}    || {self.blueprints[bp_species_best_id[spec_id]]}")

        # Print summary of module species
        print("\n\033[1mModule Species          || Module Species Avg Fitness          || Module Species Size\033[0m")
        for spec_id, spec_fitness_hisotry in self.mod_species_fitness_history.items():
            print("{:>6}                  || {:>8}                            || {:>8}"
                  .format(spec_id,
                          spec_fitness_hisotry[self.generation_counter],
                          len(self.mod_species[spec_id])))
            print(f"Best Mod of Species {spec_id}   || {self.modules[mod_species_best_id[spec_id]]}")

        # Print summary footer
        print("\n\033[1m" + '#' * 142 + "\033[0m\n")