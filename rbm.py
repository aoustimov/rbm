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

    def call(self, inputs):
        prob_h_given_v = self.presample_h_distribution(tf.matmul(inputs, self.W_b) + self.b)
        hid_activations = self.sample_h_distributin(prob_h_given_v)

        return prob_h_given_v, hid_activations

class Negative_Bern(keras.layers.Layer):


    def __init__(self, W_b, a_b):
        super(Negative_Bern, self).__init__()

        self.W_b = W_b
        self.a_b = a_b

        if args.dist_type_vis == "bernoulli":
            self.presample_v_distribution = tf.nn.sigmoid
            self.sample_v_distribution = sample_bernoulli
    
    def call(self, inputs):
        prob_v_given_h = self.presample_v_distribution(tf.matmul(inputs,
                                                       tf.transpose(self.W_b)) 
                                                       + self.a_b)
        vis_activations = self.sample_v_distribution(prob_v_given_h)

        return prob_v_given_h, vis_activations

class Functions:
    
    def __init__(self, args, W_b, a_b, b):
        super(Functions, self).__init()

        self.args = args
        self.W_b = W_b
        self.a_b = a_b
        self.b = b

        self.sample_h_distribution = sample_bernoulli

        self.Negative_Bern = Negative_Bern(args = self.args, W_b = self.W_b, a_b = self.a_b)
        self.Positive_Bern = Positive_Bern(args = self.args, W_b = self.W_b, b = self.b)

    def constrastive_divergence(self, inputs_bern):
        #positive phase
        h_tensor, positive_hidden_probs = self.Positive_Bern(inputs=inputs_bern)
        positive_hidden_activations = tf.nn.relu(tf.sign(positive_hidden_probs - 
                                                         tf.random.uniform(tf.shape(positive_hidden_probs))))

        if self.args.proba_activation_toggle == 'activation':
            positive_grads_b = tf.matmul(tf.transpose(inputs_bern),positive_hidden_activations)
        else:
            positive_grads_b = tf.matmul(tf.transpose(input_bern), positive_hidden_probs)

        # negative phase
        hidden_activations = positive_hidden_activations
        for step in range(self.args.cd_k):
            visible_probs_b, visible_acts_b = self.Negative_Bern(inputs=hidden_activations)
            temp, hidden_probs = self.Positive_Bern(inputs = inputs_bern)
            hidden_activations = tf.nn.relu(tf.sign(hidden_probs - tf.random.uniform(tf.shape(hidden_probs))))

        negative_visible_activations_b = visible_acts_b
        negative_hidden_activations = hidden_activations
        negative_grads_b = tf.matmul(tf.tanspose(negative_visible_activations_b), negative_hidden_activations)

        # gradients
        grad_w_new_b = -((positive_grads_b - negative_grads_b + 
                          tf.scalar_mul(self.args.l2_param, self.W_b))\
                              /tf.cast(self.args.batch_size, tf.float32))

        ## reductions factor not implemented
        lr_mod = self.args.lr_red_factor
        grad_w_new_b = tf.scalar_mul(self.args.lr*lr_mod, grad_w_new_b)

        grad_visible_bias_new_b = -(tf.reduce_mean(inputs_bern - negative_visible_activations_b,
                                    axis =0))
        
        grad_visible_bias_new_b = tf.scalar_mul(self.args.lr_b_bias*lr_mod,
                                                grad_visible_bias_new_b)

        # add sparsity later

        grads = [grad_w_new_b, grad_visible_bias_new_b, grad_hidden_bias_new]

        return grads

    def free_energy(self, input_bern):
        fe_b = -tf.squeeze(tf.matmul(inputs_bern, tf.expand_dims(self.a-b, -1))) \
            -tf.reduce_sum(tf.math.log(1 + tf.exp(self.b + tf.matmul(input_bern, self.W_b))), axis=1)
        fe_b_mean = tf.reduce_mean(fe_b)

        return fe_b_mean

    def gibbs_sampling(self,inputs_bern, steps=1):

        for step in range(steps):
            h_tensor, h_prob = self.Positive_Bern(inputs=inputs_bern)
            h_sample = self.sample_h_distribution(h_prob)

            visible_probs_b, visible_acts_b = self.Negative_Bern(inputs=h_sample)
        return visible_acts_b

class RBM(keras.Model):

    def __init__(self, args):
        super(RBM, self).__init__()

        self.args = args

        self.n_bern = self.args_n_bernoulli
        self.n_hidden = self.args.n_hidden

        w_init = tf.random_normal_initializer(mean = 0.0, stddev = 0.01)
        self.W_b = tf.Variable(initial_value = w_init(shape=(self.n_bern, self.n_hidden), dtype=tf.float32),
                                trainable = True, 
                                name = 'weight_matrix_bernoulli')

        a_init = tf.random_normal_initializer(mean = 0.0, stddev = 0.02)
        self.a_b = tf_Variable(initial_value = a_init(shape = (self.n_bern,), dtype = tf.float32),
                                trainable = True, name = 'visible_bias_bernoulli')

        b_init = tf.random_normal_initializer(mean = 0.0, stddev = 0.02)
        self.b = tf.Variable(initial_value = b_init(shape = (self.n_hidden,), dtype = tf.float32),
                                trainable = True, name = 'hidden_bias')

        self.Negative_Bern = Negative_Bern(args=self.args, W_b=self.Wb, a_b=self.a_b)
        self.Positive_Bern = Positive_Bern(args=self.args, W_b=self.W_b, b=self.b)
        self.Functions = Functions(args=self.args, W_b=self.W_b, a_b=self.a_b, b=self.b)

    def compile(self, optimizer):
        super(RBM, self).compile()
        self.optimizer = optimizer

    def train_step(self, inputs):
        inputs_bern = inputs["inputs_bern"]

        grads = self.Functions.constrastive_divergence(inputs=inputs_bern)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        fe_b = self.Functions.free_energy(inputs_bern=inputs_bern)
        fe_loss = fe_b
        return {"FE Loss": fe_loss}

    def call(self, inputs):

        inputs_bern = inputs['inputs_bern']
        v_acts_b = self.Functions.gibbs_sampling(inputs_bern=inputs_bern)

        fe_b = self.Functions.free_energy(inputs_bern = v_acts_b)
        
        fe_b_mod = self.Function.free_energy(inputs_bern - v_acts_b)
        fe_diff_b = (fe_b_mod - fe_b)*(fe_b_mod - fe_b)
        if self.loss_type == 'fe_reconstruction_cost':
            fe_loss = fe_diff_b
        else:
            fe_loss = fe_b
        self.add_loss(fe_loss)
        return fe_loss





            







    

