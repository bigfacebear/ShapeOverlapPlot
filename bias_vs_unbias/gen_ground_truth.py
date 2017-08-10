import time
import os
import random
import sys

sys.path.append('../')

import cv2
from deap import creator, base, tools, algorithms
from scoop import futures
import numpy as np
import pickle

from image_utils import overlapArea
import FLAGS

POP_SIZE = 1200         # The size of population. In this case, the number of sampling point on the shape.
NGEN = 20               # The number of generation, or iteration times.
CXPB = 0.5              # The probability of gene crossover.
MUTPB = 0.6             # The probability of gene mutation.

pair_num = 1000
data_dir = FLAGS.primitives_dir

images = [cv2.imread(os.path.join(data_dir, str(i)+'.png'), cv2.IMREAD_GRAYSCALE) for i in xrange(775)]

toolbox = base.Toolbox()

creator.create('FitnessMax', base.Fitness, weights=(1.0,))
creator.create('Individual', list, fitness=creator.FitnessMax)

def newTransformation(img_size):
    x = random.randint(0, img_size[1])
    y = random.randint(0, img_size[0])
    angle = 360 * random.random()
    return creator.Individual([x, y, angle])


def checkBounds(img_size):
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                child[0] = min(max(0, int(child[0])), img_size[1])
                child[1] = min(max(0, int(child[1])), img_size[0])
                child[2] %= 360.0
            return offspring

        return wrapper

    return decorator


def evaluate(individual, L, K):
    return overlapArea(L, K, individual[0:2], individual[2])


def get_max_overlap(pair):
    beg_time = time.time()
    L = images[pair[0]]
    K = images[pair[1]]
    if L is None or K is None:
        return 0

    toolbox.register('individual', newTransformation, L.shape)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    toolbox.register('mate', tools.cxUniform, indpb=0.5)
    toolbox.register('mutate', tools.mutGaussian, mu=0, sigma=5, indpb=0.5)
    toolbox.register('select', tools.selTournament, tournsize=5)
    toolbox.register('evaluate', evaluate)

    toolbox.decorate('mate', checkBounds(L.shape))
    toolbox.decorate('mutate', checkBounds(L.shape))

    pop = toolbox.population(n=POP_SIZE)

    for g in range(NGEN):
        # Select and clone the next generation individuals
        offspring = map(toolbox.clone, toolbox.select(pop, len(pop)))

        # Apply crossover and mutation on the offspring
        offspring = algorithms.varAnd(offspring, toolbox, CXPB * (1 - float(g) / NGEN), MUTPB * (1 - float(g) / NGEN))

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = []
        for ind in invalid_ind:
            fitnesses.append(evaluate(ind, L, K))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = [fit]

        # The population is entirely replaced by the offspring
        pop[:] = offspring


    best = tools.selBest(pop, k=1)[0]
    max_area = best.fitness.values[0]

    # print 'max = %d, duration = %f' % (max_area, time.time() - beg_time)
    return int(max_area)


if __name__ == '__main__':

    pairs_file_name = 'pairs_indices'

    def gen_pairs():
        print 'generate pairs.'
        pairs = [(random.randint(0, 774), random.randint(0, 774)) for _ in xrange(pair_num)]
        with open(pairs_file_name, 'wb') as fp:
            pickle.dump(pairs, fp)
        return pairs

    if os.path.exists(pairs_file_name):
        with open(pairs_file_name) as fp:
            pairs = pickle.load(fp)
            if len(pairs) != pair_num:
                pairs = gen_pairs()
    else:
        pairs = gen_pairs()

    ground_truth = np.zeros((pair_num), dtype=np.int)
    for i in xrange(100):
        beg_time = time.time()
        max_overlap_list = np.array(list(futures.map(get_max_overlap, pairs)), dtype=np.int)
        ground_truth = np.maximum(ground_truth, max_overlap_list)
        print '%d/%d, duration = %f'%(i+1, 100, time.time()-beg_time)
        sys.stdout.flush()
        with open('ground_truth', 'wb') as fp:
            pickle.dump(ground_truth, fp)
