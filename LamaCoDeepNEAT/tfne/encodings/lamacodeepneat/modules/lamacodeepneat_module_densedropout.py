from tfne.encodings.codeepneat.modules.codeepneat_module_densedropout import CoDeepNEATModuleDenseDropout
from tfne.helper_functions import round_with_step
import tensorflow as tf
import numpy as np
import math
import random


class LamaCoDeepNEATModuleDenseDropout(CoDeepNEATModuleDenseDropout):
    """
        This module inherits from Dense CoDeepNEATModule module encapsulating a Dense layer followed by
        an optional Dropout layer.
        Overriden methods are changed for supporting weight inheritance.
        Overriden methods:
        * __init__
        * create_module_layers
        * create_mutation
        * create_crossover
        * serialize
        """

    def __init__(self,
                 config_params,
                 module_id,
                 parent_mutation,
                 dtype,
                 weights=None,
                 merge_method=None,
                 units=None,
                 activation=None,
                 kernel_init=None,
                 bias_init=None,
                 dropout_flag=None,
                 dropout_rate=None,
                 self_initialization_flag=False):
        # Register the implementation specifics by calling parent class
        super().__init__(config_params, module_id, parent_mutation, dtype)

        # Register the module parameters
        self.merge_method = merge_method
        self.units = units
        self.activation = activation
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.dropout_flag = dropout_flag
        self.dropout_rate = dropout_rate
        self.weights = weights

        # If self initialization flag is provided, initialize the module parameters as they are currently set to None
        if self_initialization_flag:
            self._initialize()

    def set_weights(self, weights):
        self.weights = weights

    def get_weights(self):
        return self.weights

    def create_module_layers(self) -> (tf.keras.layers.Layer, ...):
        """
        Instantiate TF layers with their respective configuration that are represented by the current module
        configuration. Return the instantiated module layers in their respective order as a tuple.
        @return: tuple of instantiated TF layers represented by the module configuration.
        """
        # Create the basic keras Dense layer, needed in all variants of the module
        dense_layer = tf.keras.layers.Dense(units=self.units,
                                            activation=self.activation,
                                            kernel_initializer=self.kernel_init,
                                            bias_initializer=self.bias_init,
                                            dtype=self.dtype)
        # LamaCDN weight init
        if self.weights is not None:
            dense_layer.set_weights(self.weights)
        # If no dropout flag present, return solely the created dense layer as iterable. If dropout flag present, return
        # the dense layer and together with the dropout layer
        if not self.dropout_flag:
            return (dense_layer,)
        else:
            dropout_layer = tf.keras.layers.Dropout(rate=self.dropout_rate,
                                                    dtype=self.dtype)
            return dense_layer, dropout_layer

    def create_mutation(self,
                        offspring_id,
                        max_degree_of_mutation) -> CoDeepNEATModuleDenseDropout:
        """
        Create mutated DenseDropout module and return it. Categorical parameters are chosen randomly from all available
        values. Sortable parameters are perturbed through a random normal distribution with the current value as mean
        and the config specified stddev
        @param offspring_id: int of unique module ID of the offspring
        @param max_degree_of_mutation: float between 0 and 1 specifying the maximum degree of mutation
        @return: instantiated DenseDropout module with mutated parameters
        """
        # Copy the parameters of this parent module for the parameters of the offspring
        offspring_params = {'merge_method': self.merge_method,
                            'units': self.units,
                            'activation': self.activation,
                            'kernel_init': self.kernel_init,
                            'bias_init': self.bias_init,
                            'dropout_flag': self.dropout_flag,
                            'dropout_rate': self.dropout_rate,
                            'weights': self.weights}

        # Create the dict that keeps track of the mutations occuring for the offspring
        parent_mutation = {'parent_id': self.module_id,
                           'mutation': 'mutation',
                           'mutated_params': dict()}

        # Determine exact integer amount of parameters to be mutated, though minimum is 1
        param_mutation_count = math.ceil(max_degree_of_mutation * 7)

        # Uniform randomly choose the parameters to be mutated
        parameters_to_mutate = random.sample(range(7), k=param_mutation_count)

        # Mutate offspring parameters. Categorical parameters are chosen randomly from all available values. Sortable
        # parameters are perturbed through a random normal distribution with the current value as mean and the config
        # specified stddev
        for param_to_mutate in parameters_to_mutate:
            if param_to_mutate == 0:
                offspring_params['merge_method'] = random.choice(self.config_params['merge_method'])
                parent_mutation['mutated_params']['merge_method'] = self.merge_method
            elif param_to_mutate == 1:
                perturbed_units = int(np.random.normal(loc=self.units,
                                                       scale=self.config_params['units']['stddev']))
                offspring_params['units'] = round_with_step(perturbed_units,
                                                            self.config_params['units']['min'],
                                                            self.config_params['units']['max'],
                                                            self.config_params['units']['step'])
                parent_mutation['mutated_params']['units'] = self.units
                # LamaCDN: del/add weights after del/add node mutation
                offspring_params['weights'] = round_weights(self.weights, offspring_params['units'])
            elif param_to_mutate == 2:
                offspring_params['activation'] = random.choice(self.config_params['activation'])
                parent_mutation['mutated_params']['activation'] = self.activation
            elif param_to_mutate == 3:
                offspring_params['kernel_init'] = random.choice(self.config_params['kernel_init'])
                parent_mutation['mutated_params']['kernel_init'] = self.kernel_init
            elif param_to_mutate == 4:
                offspring_params['bias_init'] = random.choice(self.config_params['bias_init'])
                parent_mutation['mutated_params']['bias_init'] = self.bias_init
            elif param_to_mutate == 5:
                offspring_params['dropout_flag'] = not self.dropout_flag
                parent_mutation['mutated_params']['dropout_flag'] = self.dropout_flag
            else:  # param_to_mutate == 6:
                perturbed_dropout_rate = np.random.normal(loc=self.dropout_rate,
                                                          scale=self.config_params['dropout_rate']['stddev'])
                offspring_params['dropout_rate'] = round_with_step(perturbed_dropout_rate,
                                                                   self.config_params['dropout_rate']['min'],
                                                                   self.config_params['dropout_rate']['max'],
                                                                   self.config_params['dropout_rate']['step'])
                parent_mutation['mutated_params']['dropout_rate'] = self.dropout_rate

        return LamaCoDeepNEATModuleDenseDropout(config_params=self.config_params,
                                                module_id=offspring_id,
                                                parent_mutation=parent_mutation,
                                                dtype=self.dtype,
                                                **offspring_params)

    def create_crossover(self,
                         offspring_id,
                         less_fit_module,
                         max_degree_of_mutation) -> CoDeepNEATModuleDenseDropout:
        """
        Create crossed over DenseDropout module and return it. Carry over parameters of fitter parent for categorical
        parameters and calculate parameter average between both modules for sortable parameters
        @param offspring_id: int of unique module ID of the offspring
        @param less_fit_module: second DenseDropout module with lower fitness; second parent
        @param max_degree_of_mutation: float between 0 and 1 specifying the maximum degree of mutation
        @return: instantiated DenseDropout module with crossed over parameters
        """
        # Create offspring parameters by carrying over parameters of fitter parent for categorical parameters and
        # calculating parameter average between both modules for sortable parameters
        offspring_params = dict()

        # Create the dict that keeps track of the mutations occuring for the offspring
        parent_mutation = {'parent_id': (self.module_id, less_fit_module.get_id()),
                           'mutation': 'crossover'}

        offspring_params['merge_method'] = self.merge_method
        offspring_params['units'] = round_with_step(int((self.units + less_fit_module.units) / 2),
                                                    self.config_params['units']['min'],
                                                    self.config_params['units']['max'],
                                                    self.config_params['units']['step'])
        # LamaCDN: creation of the offspring weights
        offspring_params['weights'] = weights_after_crossover(self.weights, less_fit_module.weights,
                                                              offspring_params['units'])
        offspring_params['activation'] = self.activation
        offspring_params['kernel_init'] = self.kernel_init
        offspring_params['bias_init'] = self.bias_init
        offspring_params['dropout_flag'] = self.dropout_flag
        offspring_params['dropout_rate'] = round_with_step((self.dropout_rate + less_fit_module.dropout_rate) / 2,
                                                           self.config_params['dropout_rate']['min'],
                                                           self.config_params['dropout_rate']['max'],
                                                           self.config_params['dropout_rate']['step'])

        return LamaCoDeepNEATModuleDenseDropout(config_params=self.config_params,
                                                module_id=offspring_id,
                                                parent_mutation=parent_mutation,
                                                dtype=self.dtype,
                                                **offspring_params)

    def serialize(self) -> dict:
        """
        @return: serialized constructor variables of the module as json compatible dict
        """
        return {
            'module_type': self.get_module_type(),
            'module_id': self.module_id,
            'parent_mutation': self.parent_mutation,
            'merge_method': self.merge_method,
            'units': self.units,
            'activation': self.activation,
            'kernel_init': self.kernel_init,
            'bias_init': self.bias_init,
            'dropout_flag': self.dropout_flag,
            'dropout_rate': self.dropout_rate,
            'weights': self.weights
        }


# Helper functions
def round_weights(weights, num_of_units):
    """
    Function for updating networks weights to given number of units.
    When the new number of units is bigger than before, new weights are initialized woth value 1.
    @param weights: list of np.array of weights to be changed
    @param num_of_units: new number of units
    @return updated weights for dense module with num_of_units in layer.
    """
    # Chosen parent may have been never tested - no weights
    if weights is None:
        return None
    if num_of_units < weights[0].shape[1]:
        # delete weights of last units
        weights[0] = np.delete(weights[0], np.s_[num_of_units:], 1)
        # delete biases
        weights[1] = np.delete(weights[1], np.s_[num_of_units:])
    elif num_of_units > weights[0].shape[1]:
        # add weights of last units
        weights_to_add = np.zeros((weights[0].shape[0], num_of_units - weights[0].shape[1]))
        weights[0] = np.hstack((weights[0], weights_to_add))
        # add biases
        bias_to_add = np.zeros(num_of_units - weights[1].shape[0])
        weights[1] = np.hstack((weights[1], bias_to_add))
    return weights


def weights_after_crossover(parent1_weights, parent2_weights, num_of_units):
    """
    Function for updating networks weights to given number of units after module crossover.
    @param parent1_weights: weights of fitter parent
    @param parent2_weights: weights of worse parent
    @param num_of_units: new number of units
    @return updated weights for dense module with num_of_units in layer
    """
    # If none of the parents has been trained, return none
    if parent1_weights is None and parent2_weights is None:
        return None
    # If the fitter parent has been trained, update the weights to desired shape and return
    if parent1_weights is not None:
        parent1_weights_to_avg = np.copy(parent1_weights)
        parent1_weights_to_avg = round_weights(parent1_weights_to_avg, num_of_units)
        off_weights = np.copy(parent1_weights_to_avg)
        return off_weights
    # Else copy the weights from the less fit parent, update and return
    parent2_weights_to_avg = np.copy(parent2_weights)
    parent2_weights_to_avg = round_weights(parent2_weights_to_avg, num_of_units)
    off_weights = np.copy(parent2_weights_to_avg)
    return off_weights
