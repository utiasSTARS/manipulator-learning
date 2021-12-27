""" Various utilities that don't fit under other headers. """
import tensorflow as tf
import numpy as np
import random


def set_training_seed(s):
    tf.random.set_seed(s)
    np.random.seed(s)
    random.seed(s)
    print("Random seed for tf, np, and random set to %d" % s)
