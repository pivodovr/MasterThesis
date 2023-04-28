from tfne.encodings.codeepneat.codeepneat_encoding import CoDeepNEATEncoding
from tfne.encodings.lamacodeepneat.lamacodeepneat_genome import LamaCoDeepNEATGenome
from tfne.encodings.codeepneat.modules.codeepneat_module_base import CoDeepNEATModuleBase
from tfne.encodings.lamacodeepneat.modules.lamacodeepneat_module_association import LAMAMODULES
from tfne.encodings.lamacodeepneat.lamacodeepneat_blueprint import LamaCoDeepNEATBlueprint


class LamaCoDeepNEATEncoding(CoDeepNEATEncoding):
    """
    LamaCoDeepNEATEncoding inherits from CoDeepNEATEncoding.
    Following methods are overwritten:
    * create_genome
    * create_initial_module
    """

    def create_genome(self,
                      blueprint,
                      bp_assigned_modules,
                      output_layers,
                      input_shape,
                      generation) -> (int, LamaCoDeepNEATGenome):
        """
        Create genome by incrementing genome counter and passing supplied genotype along
        @param blueprint: CoDeepNEAT blueprint instance
        @param bp_assigned_modules: dict associating each BP species with a CoDeepNEAT module instance
        @param output_layers: string of TF deserializable layers serving as output
        @param input_shape: int-tuple specifying the input shape the genome model has to adhere to
        @param generation: int, specifying the evolution generation at which the genome was created
        @return: int of genome ID and newly created LamaCoDeepNEAT genome instance
        """
        self.genome_id_counter += 1
        # Genome genotype: (blueprint, bp_assigned_modules, output_layers)
        return self.genome_id_counter, LamaCoDeepNEATGenome(genome_id=self.genome_id_counter,
                                                            blueprint=blueprint,
                                                            bp_assigned_modules=bp_assigned_modules,
                                                            output_layers=output_layers,
                                                            input_shape=input_shape,
                                                            dtype=self.dtype,
                                                            origin_generation=generation)

    def create_initial_module(self, mod_type, config_params) -> (int, CoDeepNEATModuleBase):
        """
        Create an initial module by incrementing module ID counter and supplying initial parent_mutation
        @param mod_type: string of the module type that is to be created
        @param config_params: dict of the module parameter range supplied via config
        @return: int of module ID counter and initialized module instance
        """
        # Determine module ID and set the parent mutation to 'init' notification
        self.mod_id_counter += 1
        parent_mutation = {'parent_id': None,
                           'mutation': 'init'}

        return self.mod_id_counter, LAMAMODULES[mod_type](config_params=config_params,
                                                          module_id=self.mod_id_counter,
                                                          parent_mutation=parent_mutation,
                                                          dtype=self.dtype,
                                                          self_initialization_flag=True)

    def create_blueprint(self,
                         parent_mutation,
                         blueprint_graph,
                         optimizer_factory) -> (int, LamaCoDeepNEATBlueprint):
        """
        Create blueprint by incrementing blueprint counter and passing Blueprint parameters along
        @param parent_mutation: dict summarizing the parent mutation for the BP
        @param blueprint_graph: dict of the blueprint graph, associating graph gene ID with graph gene, being either
                                a BP graph node or a BP graph connection.
        @param optimizer_factory: instance of a configured optimizer factory that produces configured TF optimizers
        @return: int of blueprint ID and newly created BP instance
        """
        self.bp_id_counter += 1
        return self.bp_id_counter, LamaCoDeepNEATBlueprint(blueprint_id=self.bp_id_counter,
                                                           parent_mutation=parent_mutation,
                                                           blueprint_graph=blueprint_graph,
                                                           optimizer_factory=optimizer_factory)
