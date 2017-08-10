import sys

sys.path.append('../')

import pickle
import numpy as np
import matplotlib.pyplot as plt

import FLAGS

print FLAGS.data_dir

if __name__ == '__main__':

    with open('bias') as fp:
        bias = np.array(pickle.load(fp))

    with open('unbias') as fp:
        unbias = np.array(pickle.load(fp))

    x = bias - unbias

    num_bins = 50

    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(x, num_bins)
    ax.set_xlabel('bias - unbias')
    ax.set_ylabel('Number')

    # fig.tight_layout()
    plt.show()