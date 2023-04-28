from tfne.encodings.codeepneat.codeepneat_genome import CoDeepNEATGenome
from tfne.encodings.lamacodeepneat.lamacodeepneat_model import LamaCoDeepNEATModel
import numpy as np
from tensorflow.python.keras.utils.layer_utils import count_params
from tfne.algorithms.lamacodeepneat.nsgaII import get_flops

class LamaCoDeepNEATGenome(CoDeepNEATGenome, LamaCoDeepNEATModel):
    """
    LamaCoDeepNEATGenome inherits from CoDeepNEATGenome.
    Following methods are overwritten:
    * __init__
    * serialize
    Following methods are unique:
    * compute_flops
    * set_flops
    * get_flops
    * set_weights
    * get_weights
    """

    def __init__(self, genome_id, blueprint, bp_assigned_modules, output_layers, input_shape, dtype, origin_generation):
        """
        Create CoDeepNEAT genome by saving the associated genotype parameters as well as additional information like
        dtype and origin generation. Then create TF model from genotype.
        @param genome_id: int of unique genome ID
        @param blueprint: CoDeepNEAT blueprint instance
        @param bp_assigned_modules: dict associating each BP species with a CoDeepNEAT module instance
        @param output_layers: string of TF deserializable layers serving as output
        @param input_shape: int-tuple specifying the input shape the genome model has to adhere to
        @param dtype: string of TF dtype
        @param origin_generation: int, specifying the evolution generation at which the genome was created
        """
        # Register parameters
        self.genome_id = genome_id
        self.input_shape = input_shape
        self.dtype = dtype
        self.origin_generation = origin_generation

        # Register genotype
        self.blueprint = blueprint
        self.bp_assigned_modules = bp_assigned_modules
        self.output_layers = output_layers

        # Initialize internal variables
        self.fitness = None
        self.flops = None
        self.dominates = []
        self.rank = -1
        self.dom_counter = 0

        # Create optimizer and model
        self.model = None
        self.optimizer = self.blueprint.create_optimizer()
        self._create_model()

        # Create weights attribute needed for LamaCoDeepNEAT
        self.weights = np.copy(self.model.get_weights())

    def _create_model(self):
        return LamaCoDeepNEATModel._create_model(self)

    def serialize(self) -> dict:
        """
        @return: serialized constructor variables of the genome as json compatible dict
        """
        # Serialize the assignment of modules to the bp species for json output
        serialized_bp_assigned_mods = dict()
        for spec, assigned_mod in self.bp_assigned_modules.items():
            serialized_bp_assigned_mods[spec] = assigned_mod.serialize()

        # Use the serialized mod to bp assignment to create a serialization of the whole genome
        serialized_genome = {
            'genome_type': 'LamaCoDeepNEAT',
            'genome_id': self.genome_id,
            'fitness': self.fitness,
            'blueprint': self.blueprint.serialize(),
            'bp_assigned_modules': serialized_bp_assigned_mods,
            'output_layers': self.output_layers,
            'input_shape': self.input_shape,
            'dtype': self.dtype,
            'origin_generation': self.origin_generation
        }

        return serialized_genome

    def compute_flops(self):
        """
        @return: flops of genome model
        """
        return get_flops(self.model)

    def get_flops(self):
        return self.flops

    def set_flops(self, flops):
        self.flops = flops

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights





