import os
import tempfile
import sys

import tfne
from tfne.algorithms.lamacodeepneat.lamacodeepneat import LamaCoDeepNEAT

# Deactivate GPUs as pytest seems very error-prone in combination with Tensorflow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

sys.path.insert(0, '..')

def sanity_check_algorithm_state(ne_algorithm):
    """
    Very basic sanity check as the purpose of the pytest checks is the run of the evolutionary loops. If there are some
    bugs in the evolutionary process the complex logic will fail. Therefore there is not much purpose in doing extensive
    asserts after the evolutionary process succeded.
    """
    best_genome = ne_algorithm.get_best_genome()
    assert 100 >= best_genome.get_fitness() > 0

def test_lamacodeepneat_mnist():
    # Create test config
    config = tfne.parse_configuration(os.path.dirname(__file__) + '/test_codeepneat_4_config.cfg')
    environment = tfne.environments.MNISTEnvironment(weight_training=True, lamarck=True, config=config)
    ne_algorithm = LamaCoDeepNEAT(config)

    # Start test
    engine = tfne.EvolutionEngine(ne_algorithm=ne_algorithm,
                                  environment=environment,
                                  backup_dir_path=tempfile.gettempdir(),
                                  max_generations=2,
                                  max_fitness=None)
    engine.train()

    # Sanity check state of the algorithm
    sanity_check_algorithm_state(ne_algorithm)


def test_lamacodeepneat_cifar():
    # Create test config
    config = tfne.parse_configuration(os.path.dirname(__file__) + '/test_codeepneat_4_config.cfg')
    environment = tfne.environments.CIFAR10Environment(weight_training=True, lamarck=True, config=config)
    ne_algorithm = LamaCoDeepNEAT(config)

    # Start test
    engine = tfne.EvolutionEngine(ne_algorithm=ne_algorithm,
                                  environment=environment,
                                  backup_dir_path=tempfile.gettempdir(),
                                  max_generations=2,
                                  max_fitness=None)
    engine.train()

    # Sanity check state of the algorithm
    sanity_check_algorithm_state(ne_algorithm)

test_lamacodeepneat_mnist()
test_lamacodeepneat_cifar()

