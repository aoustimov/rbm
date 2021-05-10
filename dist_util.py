import numpy as np
import tensorflow as tf

def sample_bernoulli(probs, straight_through=False):
    sample = tf.nn.relu(tf.sign(probs - tf.random.uniform(tf.shape(probs))))

    if straight_through:
        sample = tf.stop_gradient(sample - probs) + probs
    return sample