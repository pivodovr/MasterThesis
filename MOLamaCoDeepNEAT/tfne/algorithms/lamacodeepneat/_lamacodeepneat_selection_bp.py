from tfne.algorithms.codeepneat._codeepneat_selection_bp import CoDeepNEATSelectionBP

from typing import Union


class LamaCoDeepNEATSelectionBP(CoDeepNEATSelectionBP):
    """
    LamaCoDeepNEATSelectionBP inherits from CoDeepNEATSelectionBP.
    Following methods are new:
    * _select_blueprints_multiobjective
    """
    def _select_blueprints_multiobjective(self) -> ({Union[int, str]: int}, {int: int}):
        ### Generational Parent Determination ###
        # Determine whole population of blueprints as potential parents as long as NSGA-II selection works differently.
        # We ignore reproduction threshold parameter and we implement elitism later.
        bp_spec_parents = self.pop.bp_species.copy()

        #### Offspring Size Calculation ####
        # Determine the amount of offspring for the blueprint species as the intended population size.
        bp_spec_offspring = {1: self.bp_pop_size}

        # Return
        # bp_spec_offspring {int: int} associating species id with amount of offspring
        # bp_spec_parents {int: [int]} associating species id with list of potential parent ids for species
        return bp_spec_offspring, bp_spec_parents





