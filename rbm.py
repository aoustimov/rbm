import tensorflow as tf
from tensorflow import keras
from dist_util import sample_bernoulli
import numpy as np
import os
from util import save_merged_images, convert_to_onehot
import imageio

class Positive_Bern(keras.layers.Layer):

    def __init__(self, args, W_b, b):
        super(Positive_Bern, self).__init__()

        self.W_b = W_b
        self.b = b

        if args.dist_type_hid == "bernoulli":
            self.presample_h_distribution = tf.nn.sigmoid
            self.sample_h_distribution = sample_bernoulli

    def call(self. inputs):
        p