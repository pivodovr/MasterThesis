import math
import random
import tensorflow as tf
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder


def dominates(ind1, ind2):
    """
    Decide if individual 1 dominates individual 2.
    @param ind1: blueprint or module
    @param ind2: blueprint or module
    @return: True if ind1 dominates ind2
    """
    fit1 = ind1.get_fitness()
    fit2 = ind2.get_fitness()
    flops1 = ind1.get_flops()
    flops2 = ind2.get_flops()
    return (fit1 <= fit2 and flops1 <= flops2) and (fit1 < fit2 or flops1 < flops2)


def fast_non_dominated_sort(pop_ids, pop_indis):
    """
    Sort individuals (blueprint/modules) in means of non-dominated sort from NSGA-II.
    Set individual's attributes: rank, dom_counter, dominates.
    Two objectives: size and fitness are used for sorting.
    @param pop_ids: list of int ids of individuals, who are sorted... [id]
    @param pop_indis: dictionary of whole population... {id : individual}
    @return: 2D list [index of front : [index of individual in pop_indis]]
    """
    fronts = [[]]  # list of all fronts
    for pid in pop_ids:
        p = pop_indis[pid]
        for qid in pop_ids:
            q = pop_indis[qid]
            if dominates(p, q):
                p.dominates.append(qid)
            elif dominates(q, p):
                p.dom_counter += 1
        if p.dom_counter == 0:
            p.rank = 0
            if p not in fronts[0]:
                fronts[0].append(pid)
    i = 0
    while fronts[i] != []:
        Q = []  # members of next front
        for pid in fronts[i]:
            for qid in pop_indis[pid].dominates:
                q = pop_indis[qid]
                q.dom_counter -= 1
                if q.dom_counter == 0:
                    q.rank = i + 1
                    if qid not in Q:
                        Q.append(qid)
        i += 1
        fronts.append(Q)
    fronts.remove([])
    return fronts  # fronts is 2D list [index of front : [index of ind in self.pop]]


def crowding_distance(front, pop_indis):
    """
    Compute crowding distance for all individuals in given front.
    Set individual's attribute distance.
    @param front: list of int ids... [id]
    @param pop_indis: dictionary of whole population... {id : individual}
    @return: sorted list of individual's id and its crowding distance... [[id : distance]]
    """
    id_fitness_flops_dist = \
        [[i, pop_indis[i].get_fitness(), pop_indis[i].get_flops(), pop_indis[i].distance] for i in front]
    sorted_by_fitness = sorted(id_fitness_flops_dist, key=lambda x: x[1])
    sorted_by_fitness[0][3] = math.inf
    sorted_by_fitness[-1][3] = math.inf
    if sorted_by_fitness[-1][1] - sorted_by_fitness[0][1] != 0:
        for i in range(1, len(sorted_by_fitness) - 1):
            sorted_by_fitness[i][3] += (sorted_by_fitness[i + 1][1] - sorted_by_fitness[i - 1][1]) / \
                                       (sorted_by_fitness[-1][1] - sorted_by_fitness[0][1])
    sorted_by_flops = sorted(sorted_by_fitness, key=lambda x: x[2])
    sorted_by_flops[0][3] = math.inf
    sorted_by_flops[-1][3] = math.inf
    if sorted_by_flops[-1][2] - sorted_by_flops[0][2] != 0:
        for i in range(1, len(sorted_by_flops) - 1):
            sorted_by_flops[i][3] += (sorted_by_flops[i + 1][2] - sorted_by_flops[i - 1][2]) / \
                                     (sorted_by_flops[-1][2] - sorted_by_flops[0][2])
    for ind_id, fit, size, distance in sorted_by_flops:
        pop_indis[ind_id].distance = distance

    sorted_by_dist = sorted(sorted_by_flops, key=lambda x: (x[3], -x[1]), reverse=True)
    return [[i[0], i[3]] for i in sorted_by_dist]


def tournament_selection(mating_pool_ids, pop_indis):
    """
    Choose winner from binary tournament, who has better rank, secondary number of siblings, then crowding distance.
    @param mating_pool_ids: list of ids of possible parents
    @param pop_indis: dictionary of whole population... {id : individual}
    @return winning individual
    """
    competitor_id1, competitor_id2 = random.sample(mating_pool_ids, 2)
    competitor1 = pop_indis[competitor_id1]
    competitor2 = pop_indis[competitor_id2]
    if competitor1.rank < competitor2.rank:
        return competitor1
    if competitor1.rank > competitor2.rank:
        return competitor2
    if competitor1.siblings > competitor2.siblings:
        return competitor1
    if competitor1.siblings < competitor2.siblings:
        return competitor2
    if competitor1.distance > competitor2.distance:
        return competitor1
    if competitor1.distance < competitor2.distance:
        return competitor2
    if random.random() < 0.5:
        return competitor1
    return competitor2


def get_flops(model):
    """
    Compute a models FLOPs value.
    Source: https://github.com/tensorflow/tensorflow/issues/32809#issuecomment-849439287
    """
    forward_pass = tf.function(
        model.call,
        input_signature=[tf.TensorSpec(shape=(1,) + model.input_shape[1:])])

    graph_info = profile(forward_pass.get_concrete_function().graph,
                         options=ProfileOptionBuilder.float_operation())

    # The //2 is necessary since `profile` counts multiply and accumulate
    # as two flops, here we report the total number of multiply accumulate ops
    flops = graph_info.total_float_ops // 2
    return flops
