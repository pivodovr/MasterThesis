from tfne.encodings.codeepneat.modules.codeepneat_module_conv2dmaxpool2ddropout \
    import CoDeepNEATModuleConv2DMaxPool2DDropout
from tfne.helper_functions import round_with_step

import math
import random
import numpy as np


class LamaCoDeepNEATModuleConv2DMaxPool2DDropout(CoDeepNEATModuleConv2DMaxPool2DDropout):
    """
    This module inherits from Conv2D CoDeepNEATModule module encapsulating a Conv2D layer,
    a optionally following MaxPooling2D layer and a optionally following Dropout layer.
    The downsampling layer is another Conv2D layer.
    Overriden methods are changed for supporting weight inheritance.
    Overriden methods:
    * __init__
    * create_mutation
    * create_crossover
    * serialize
    * __str__
    New methods:
    * set_weights
    * get_weights
    * compute_flops
    * set_flops
    * get_flops
    * get_rank
    * get_crowd_distance
    * set_id
    * set_siblings
    * get_siblings
    """

    def __init__(self,
                 config_params,
                 module_id,
                 parent_mutation,
                 dtype,
                 merge_method=None,
                 filters=None,
                 kernel_size=None,
                 strides=None,
                 padding=None,
                 activation=None,
                 kernel_init=None,
                 bias_init=None,
                 max_pool_flag=None,
                 max_pool_size=None,
                 dropout_flag=None,
                 dropout_rate=None,
                 weights=None,
                 self_initialization_flag=False):
        # Register the implementation specifics by calling parent class
        super().__init__(config_params, module_id, parent_mutation, dtype)

        # Register the module parameters
        self.merge_method = merge_method
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.max_pool_flag = max_pool_flag
        self.max_pool_size = max_pool_size
        self.dropout_flag = dropout_flag
        self.dropout_rate = dropout_rate
        self.weights = weights
        self.dominates = []
        self.rank = -1
        self.dom_counter = 0
        self.distance = 0
        self.fitness = 100
        self.siblings = 1

        # If self initialization flag is provided, initialize the module parameters as they are currently set to None
        if self_initialization_flag:
            self._initialize()

        self.flops = self.compute_flops()

    def __str__(self) -> str:
        """
        @return: string representation of the module
        """
        return "LamaCoDeepNEAT Conv2D MaxPool Dropout Module | ID: {:>6} | Fitness: {:>6} | FLOPs: {} " \
               "| Filters: {:>4} | Kernel: {:>6} | Activ: {:>6} | Pool Size: {:>6} | Dropout: {:>4}" \
            .format('#' + str(self.module_id),
                    self.fitness,
                    self.flops,
                    self.filters,
                    str(self.kernel_size),
                    self.activation,
                    "None" if self.max_pool_flag is False else str(self.max_pool_size),
                    "None" if self.dropout_flag is False else self.dropout_rate)

    def set_id(self, mod_id):
        self.module_id = mod_id

    def set_weights(self, weights):
        self.weights = weights

    def get_weights(self):
        return self.weights

    def get_siblings(self):
        return self.siblings

    def set_siblings(self, siblings):
        self.siblings = siblings

    def create_mutation(self,
                        offspring_id,
                        max_degree_of_mutation) -> CoDeepNEATModuleConv2DMaxPool2DDropout:
        # Copy the parameters of this parent module for the parameters of the offspring
        offspring_params = {'merge_method': self.merge_method,
                            'filters': self.filters,
                            'kernel_size': self.kernel_size,
                            'strides': self.strides,
                            'padding': self.padding,
                            'activation': self.activation,
                            'kernel_init': self.kernel_init,
                            'bias_init': self.bias_init,
                            'max_pool_flag': self.max_pool_flag,
                            'max_pool_size': self.max_pool_size,
                            'dropout_flag': self.dropout_flag,
                            'dropout_rate': self.dropout_rate}
        if self.weights is None:
            offspring_params['weights'] = None
        else:
            offspring_params['weights'] = np.copy(self.weights)

        # Create the dict that keeps track of the mutations occuring for the offspring
        parent_mutation = {'parent_id': self.module_id,
                           'mutation': 'mutation',
                           'mutated_params': dict()}

        # Determine exact integer amount of parameters to be mutated, though minimum is 1
        param_mutation_count = math.ceil(max_degree_of_mutation * 12)

        # Uniform randomly choose the parameters to be mutated
        parameters_to_mutate = random.sample(range(12), k=param_mutation_count)

        # Mutate offspring parameters. Categorical parameters are chosen randomly from all available values. Sortable
        # parameters are perturbed through a random normal distribution with the current value as mean and the config
        # specified stddev
        for param_to_mutate in parameters_to_mutate:
            if param_to_mutate == 0:
                offspring_params['merge_method'] = random.choice(self.config_params['merge_method'])
                parent_mutation['mutated_params']['merge_method'] = self.merge_method
            elif param_to_mutate == 1:
                perturbed_filters = int(np.random.normal(loc=self.filters,
                                                         scale=self.config_params['filters']['stddev']))
                offspring_params['filters'] = round_with_step(perturbed_filters,
                                                              self.config_params['filters']['min'],
                                                              self.config_params['filters']['max'],
                                                              self.config_params['filters']['step'])
                parent_mutation['mutated_params']['filters'] = self.filters
                # LamaCDN: the size of the weights are changed according to the change of the number of filters
                if self.weights is not None:
                    offspring_params['weights'] = round_weights_filters(offspring_params['weights'],
                                                                        offspring_params['filters'])
            elif param_to_mutate == 2:
                offspring_params['kernel_size'] = random.choice(self.config_params['kernel_size'])
                parent_mutation['mutated_params']['kernel_size'] = self.kernel_size
                # LamaCDN: the size of the weights are changed according to the change of the kernel size
                if self.weights is not None:
                    offspring_params['weights'] = round_weights_kernel(offspring_params['weights'],
                                                                       offspring_params['kernel_size'])
            elif param_to_mutate == 3:
                offspring_params['strides'] = random.choice(self.config_params['strides'])
                parent_mutation['mutated_params']['strides'] = self.strides
            elif param_to_mutate == 4:
                offspring_params['padding'] = random.choice(self.config_params['padding'])
                parent_mutation['mutated_params']['padding'] = self.padding
            elif param_to_mutate == 5:
                offspring_params['activation'] = random.choice(self.config_params['activation'])
                parent_mutation['mutated_params']['activation'] = self.activation
            elif param_to_mutate == 6:
                offspring_params['kernel_init'] = random.choice(self.config_params['kernel_init'])
                parent_mutation['mutated_params']['kernel_init'] = self.kernel_init
            elif param_to_mutate == 7:
                offspring_params['bias_init'] = random.choice(self.config_params['bias_init'])
                parent_mutation['mutated_params']['bias_init'] = self.bias_init
            elif param_to_mutate == 8:
                offspring_params['max_pool_flag'] = not self.max_pool_flag
                parent_mutation['mutated_params']['max_pool_flag'] = self.max_pool_flag
            elif param_to_mutate == 9:
                offspring_params['max_pool_size'] = random.choice(self.config_params['max_pool_size'])
                parent_mutation['mutated_params']['max_pool_size'] = self.max_pool_size
            elif param_to_mutate == 10:
                offspring_params['dropout_flag'] = not self.dropout_flag
                parent_mutation['mutated_params']['dropout_flag'] = self.dropout_flag
            else:  # param_to_mutate == 11:
                perturbed_dropout_rate = np.random.normal(loc=self.dropout_rate,
                                                          scale=self.config_params['dropout_rate']['stddev'])
                offspring_params['dropout_rate'] = round_with_step(perturbed_dropout_rate,
                                                                   self.config_params['dropout_rate']['min'],
                                                                   self.config_params['dropout_rate']['max'],
                                                                   self.config_params['dropout_rate']['step'])
                parent_mutation['mutated_params']['dropout_rate'] = self.dropout_rate

        return LamaCoDeepNEATModuleConv2DMaxPool2DDropout(config_params=self.config_params,
                                                          module_id=offspring_id,
                                                          parent_mutation=parent_mutation,
                                                          dtype=self.dtype,
                                                          **offspring_params)

    def create_crossover(self,
                         offspring_id,
                         less_fit_module,
                         max_degree_of_mutation) -> CoDeepNEATModuleConv2DMaxPool2DDropout:
        # Create offspring parameters by carrying over parameters of fitter parent for categorical parameters and
        # calculating parameter average between both modules for sortable parameters
        offspring_params = dict()

        # Create the dict that keeps track of the mutations occuring for the offspring
        parent_mutation = {'parent_id': (self.module_id, less_fit_module.get_id()),
                           'mutation': 'crossover'}

        offspring_params['merge_method'] = self.merge_method
        offspring_params['filters'] = round_with_step(int((self.filters + less_fit_module.filters) / 2),
                                                      self.config_params['filters']['min'],
                                                      self.config_params['filters']['max'],
                                                      self.config_params['filters']['step'])
        offspring_params['kernel_size'] = self.kernel_size
        # LamaCDN: creation of the offspring weights
        offspring_params['weights'] = None
        offspring_params['weights'] = weights_after_crossover(self.weights, less_fit_module.weights,
                                                              offspring_params['filters'],
                                                              offspring_params['kernel_size'])
        offspring_params['strides'] = self.strides
        offspring_params['padding'] = self.padding
        offspring_params['activation'] = self.activation
        offspring_params['kernel_init'] = self.kernel_init
        offspring_params['bias_init'] = self.bias_init
        offspring_params['max_pool_flag'] = self.max_pool_flag
        offspring_params['max_pool_size'] = self.max_pool_size
        offspring_params['dropout_flag'] = self.dropout_flag
        crossed_over_dropout_rate = round_with_step(((self.dropout_rate + less_fit_module.dropout_rate) / 2),
                                                    self.config_params['dropout_rate']['min'],
                                                    self.config_params['dropout_rate']['max'],
                                                    self.config_params['dropout_rate']['step'])
        offspring_params['dropout_rate'] = crossed_over_dropout_rate

        return LamaCoDeepNEATModuleConv2DMaxPool2DDropout(config_params=self.config_params,
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
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'activation': self.activation,
            'kernel_init': self.kernel_init,
            'bias_init': self.bias_init,
            'max_pool_flag': self.max_pool_flag,
            'max_pool_size': self.max_pool_size,
            'dropout_flag': self.dropout_flag,
            'dropout_rate': self.dropout_rate
        }

    def set_flops(self, flops):
        self.flops = flops

    def get_flops(self):
        return self.flops

    def compute_flops(self):
        """
        FLOPs of modules are computed manually.
        """
        if self.filters is None or self.kernel_size is None:
            return None
        return 2 * self.filters * self.kernel_size * self.kernel_size

    def get_rank(self):
        return self.rank

    def get_crowd_distance(self):
        return self.distance


# Helper functions
def round_weights_filters(weights, num_of_filters):
    """
    Function for updating networks weights to given number of filters. Weights are updated in 4th dimension.
    When the new number of filters is bigger than before, new weights are initialized with value 0.
    @param weights: list of np.array of weights to be changed
    @param num_of_filters: new number of filters
    @return updated weights for dense module with num_of_filters in layer.
    """
    # Chosen parent may have been never tested - no weights
    if weights is None:
        return None
    if num_of_filters < weights[0].shape[3]:
        # delete weights of last units
        weights[0] = np.delete(weights[0], np.s_[num_of_filters:], 3)
        # delete biases
        weights[1] = np.delete(weights[1], np.s_[num_of_filters:])
    elif num_of_filters > weights[0].shape[3]:
        # add weights of last units
        weights_to_add = np.zeros((weights[0].shape[0], weights[0].shape[1],
                                   weights[0].shape[2], num_of_filters - weights[0].shape[3]))
        weights[0] = np.concatenate((weights[0], weights_to_add), 3)
        # add biases
        bias_to_add = np.zeros(num_of_filters - weights[1].shape[0])
        weights[1] = np.hstack((weights[1], bias_to_add))
    return weights


def round_weights_kernel(weights, kernel_size):
    """
    Function for updating networks weights to given kernel size. Weights are updated in 1st a 2nd dimension.
    When the new size is bigger than before, new weights are initialized with value 0.
    @param weights: list of np.array of weights to be changed
    @param kernel_size: new kernel size
    @return updated weights for dense module with kernel size.
    """
    # Chosen parent may have never been tested - no weights
    if weights is None:
        return None
    # First kernel dimension
    if kernel_size < weights[0].shape[0]:
        # delete weights of last units
        weights[0] = np.delete(weights[0], np.s_[kernel_size:], 0)
    elif kernel_size > weights[0].shape[0]:
        # add weights of last units
        weights_to_add_0 = np.zeros((kernel_size - weights[0].shape[0], weights[0].shape[1],
                                     weights[0].shape[2], weights[0].shape[3]))
        weights[0] = np.concatenate((weights[0], weights_to_add_0), 0)
    # Second kernel dimension
    if kernel_size < weights[0].shape[1]:
        # delete weights of last units
        weights[0] = np.delete(weights[0], np.s_[kernel_size:], 1)
    elif kernel_size > weights[0].shape[1]:
        # add weights of last units
        weights_to_add_1 = np.zeros((weights[0].shape[0], kernel_size - weights[0].shape[1],
                                     weights[0].shape[2], weights[0].shape[3]))
        weights[0] = np.concatenate((weights[0], weights_to_add_1), 1)
    return weights


def weights_after_crossover(parent1_weights, parent2_weights, num_of_filters, kernel_size):
    """
    Function for computing offspring weights of two given parents. Update offspring network weights to given number
    of filters and kernel size after module crossover. Offspring's weights are, ideally, weights of the fitter parent.
    @param parent1_weights: weights of fitter parent
    @param parent2_weights: weights of worse parent
    @param num_of_filters: new number of filters
    @param kernel_size: new kernel size
    @return updated weights for module with given kernel_size and num_of_filters.
    """
    # If none of the parents has been trained, return none
    if parent1_weights is None and parent2_weights is None:
        return None
    # If the fitter parent has been trained, update the weights to desired shape and return
    if parent1_weights is not None:
        parent1_weights_to_avg = np.copy(parent1_weights)
        parent1_weights_to_avg = round_weights_filters(parent1_weights_to_avg, num_of_filters)
        parent1_weights_to_avg = round_weights_kernel(parent1_weights_to_avg, kernel_size)
        off_weights = np.copy(parent1_weights_to_avg)
        return off_weights
    # Else copy the weights from the less fit parent, update and return
    parent2_weights_to_avg = np.copy(parent2_weights)
    parent2_weights_to_avg = round_weights_filters(parent2_weights_to_avg, num_of_filters)
    parent2_weights_to_avg = round_weights_kernel(parent2_weights_to_avg, kernel_size)
    off_weights = np.copy(parent2_weights_to_avg)
    return off_weights
