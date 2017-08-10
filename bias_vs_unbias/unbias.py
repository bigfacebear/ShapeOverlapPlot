import os
import sys

sys.path.append('../')

import random
import time

import cv2
import numpy as np
import pickle
from scoop import futures

from image_utils import overlapArea
import FLAGS

pair_num = 1000
sample_num = 1200 * 20
data_dir = FLAGS.primitives_dir

images = [cv2.imread(os.path.join(data_dir, str(i)+'.png'), cv2.IMREAD_GRAYSCALE) for i in xrange(775)]

def get_overlap(params):
    pair, trans = params
    return overlapArea(images[pair[0]], images[pair[1]], trans[0:2], trans[2])

def get_max_overlap(pair):
    beg_time = time.time()
    rows, cols = images[pair[0]].shape
    trans_list = [(random.randint(0, rows - 1), random.randint(0, cols - 1), random.random() * 360)
                  for _ in xrange(sample_num)]
    params_list = [(pair, trans) for trans in trans_list]
    overlaps = np.array(list(map(get_overlap, params_list)))
    max_overlap = np.max(overlaps)
    print 'max = %d, duration = %f' % (max_overlap, time.time() - beg_time)
    return int(max_overlap)

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

    max_overlap_list = list(futures.map(get_max_overlap, pairs))

    with open('unbias', 'wb') as fp:
        pickle.dump(max_overlap_list, fp)
