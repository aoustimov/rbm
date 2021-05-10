import tensorflow as tf
from tensorflow import keras
from dist_util import sample_bernoulli
import numpy as np
import os
from util import save_merged_images, convert_to_onehot
import imageio

class Negative_Bern(keras.layers.Layer):
    
    def __init__(self, args, W_b, a_b):
        super(Negative_Bern, self).__init__()

        self.W_b = W_b
        self.a_b = a_b

        if args.dist_type_vis == 'bernoulli':
            self.presample_v_distribution = tf.nn.sigmoid
            self.sample_v_distribution = sample_bernoulli

    def call(self, inputs):
        prob_v_given_h = self.presample_v_distribution(tf.matmul(inputs, 
                                                       tf.transpose(self.W_b))
                                                       + self.a_b)
        vis_activations = self.sample_v_distributions(prob_v_given_h)
        return prov_v_given_h, vis_activations

class Positive_Bern_SM(keras.layers.Layer):

    def __init__(self, args, W_b, W_sm, b):
        super(Positive_Bern, self).__init__()

        self.args = args

        self.W_b = W_b
        self.W_sm = W_sm
        self.b = b

        if args.dist_type_hid == "bernoulli":
            self.presample_h_distribution = tf.nn.sigmoid
            self.sample_h_distribution = sample_bernoulli

    def call(self, inputs_bern, inputs_sm):
        h_list_all = []

        for i in range(self.args.batch_size):
            a_mat = inputs_sm[i, :, :]
            h_list = []
            for j in range(self.W_sm.shape[0]):
                h_mat = self.W_sm[j, :, :]
                mul = tf.mat_mul(a_mat, h_mat, transpose_a=True)
                tr = tf.linalg.trace(mul)
                h_list.append(tr)
            h_list_all.append(h_list)
        h_tensor = tf.convert_to_tensor(h_list_all)

        prob_h_given_v_sm_b = self.presample_h_distribution((tf.matmul(inputs_bern, self.W_b) + 
                                                             h_tensor + self.b))
        return h_tensor, prob_h_given_v_sm_b

class Negative_Bern_SM(keras.layers.Layer):
    def __init__(self, args, W_sm, a_sm):
        super(Negative_Bern_SM, self).__init__()

        self.args = args
        self.W_sm = W_sm
        self.a_sm = a_sm

    def call(self, inputs_sm, hid_acts):
        sm_list_all = []
        for j in range(self.args.batch_size):
            a_mat = tf.reshape(hid_acts[j,:], [-1, hid_acts.shape[1]])
            sm_list = []
            for i in range(self.W_sm.shape[2]):
                w_mat = self.W_sm[:, :, i]
                mul = tf.matmul(a_mat, w_mat) + self.a_sm[:,i]
                softmax = tf.nn.softmax(mul)
                sm_list.append(softmax)
            sm_list_all.append(sm_list)
        v_tensor = tf.transpose(tf.reshape(tf.convert_to_tensor(sm_list_all),
                                           [self.args.batch_size, inputs_sm.shape[2], inputs_sm.shape[1]]),
                                perm=[0, 2, 1])       
        return v_tensor


class Functions:
    
    def __init__(self, args, W_b, W_sm, a_b, a_sm, b):
        super(Functions, self).__init()

        self.args = args
        self.W_b = W_b
        self.W_sm = W_sm
        self.a_b = a_b
        self.a_sm = a_sm
        self.b = b

        self.sample_h_distribution = sample_bernoulli

        self.Negative_Bern = Negative_Bern(args = self.args,
                                           W_b = self.W_b,
                                           a_b = self.a_b)
        self.Positive_Bern_SM = Positive_Bern(args = self.args,
                                           W_b = self.W_b,
                                           W_sm = self.W_sm,
                                           b = self.b)
        self.Negative_Bern_SM = Negative_Bern_SM(args = self.args,
                                                 W_sm = self.W_sm,
                                                 a_sm = self.a_sm)

    def softmax_grads(self, inputs_sm, hid_acts):
        batch_list = []
        for i in range(self.args.batch_size):
            mat_list = []
            for j in range(hid_acts.shape[1]):
                mat_j = tf.scalar_mul(hid_acts[i,j],inputs_sm[i, :, :])
                mat_list.append(mat_j)
            mat_all = tf.stack(mat_list)
            batch_list.append(mat_all)
        tensor_output = tf.reduce_sum(tf.convert_to_tensor(batch_list), axis=0)

        return tensor_output

    def contrastive_divergence(self, inputs_bern, inputs_sm):
        
        # positive phase
        h_tensor, positive_hidden_probs = self.Positive_Bern_SM(inputs_bern = inputs_bern,
                                                                inputs_sm = inputs_sm)
        positive_hidden_activations = tf.nn.relu(tf.sign(positive_hidden_probs - 
                                                        tf.random.uniform(tf.shape(positive_hidden_probs))))

        if self.args.proba_activation_toggle == 'activation':
            positive_grads_b = tf.matmul(tf.transpose(inputs_bern), positive_hidden_activations)
            prositive_grads_s = tf.softmax_grads(inputs_sm=inputs_sm, hid_acts=positive_hidden_activations)
        else:
            positive_grads_b = tf.matmul(tf.transpose(inputs_bern), positive_hidden_probs)
            positive_grads_s = self.softmax_grads(inputs_sm=inputs_sm, hid_acts=positive_hidden_probs)

        # negative phase
        hidden_activations = positive_hidden_activations

        for step in range(self.args.cd_k):
            visible_probs_b, visible_acts_b = self.Negative_Bern(inputs=hidden_activations)
            visible_probs_s = self.Negative_Bern_SM(inputs_sm=inputs_sm, hid_acts = hidden_activatinos)

            visible_acts_s = visible_probs_s

            temp, hidden_probs = self.Positive_Bern_SM(inputs_bern=inputs_bern, inputs_sm=inputs_sm)
            negative_activations = tf.nn.relu(tf.sign(hidden_probs - tf.random.uniform(tf.shape(hidden_probs))))

        negative_visible_activations_b = visible_acts_b
        negative_visible_activations_s = visible_acts_s

        negative_hidden_activations = hidden_activations

        negative_grads_b = tf.matmul(tf.transpose(negative_visible_activations_b),
                                     negative_hidden_activations)
        negative_grads_s = self.softmax_grads(inputs_sm=negative_visible_activations_s,
                                              hid_acts=negative_hidden_activations)

        grad_w_new_b = -((positive_grads_b - negative_grads_b +
                          tf.scalar_mul(self.args.l2_param, self.W_b))\
                              /tf.cast(self.args.batch_size, tf.float32))

        grad_w_new_s = -((positive_grads_s - negative_grads_s + 
                          tf.scalar_mul(self.args.l2_param, self.W_sm))\
                              /tf.cast(self.args.batch_size, tf.float32))

        lr_mod = self.args.lr_red_factor
        grad_w_new_b = tf.scalar_mul(self.args.lr_b*lr_mod, grad_w_new_b)
        grad_w_new_s = tf.scalar_mul(self.args.lr_s*lr_mod, grad_w_new_s)

        grad_visible_bias_new_b = -(tf.reduce_mean(inputs_bern - negative_visible_activations_b, 0))
        grad_visible_bias_new_s = -(tf.reduce_mean(inputs_sm - negative_visible_activations_s, 0))

        grad_visible_bias_new_b = tf.scalar_mul(self.args.lr_b_bias*lr_mod,grad_visible_bias_new_b)
        grad_visible_bias_new_s = tf.scalar_mul(self.args.lr_s_bias*lr_mod,grad_visible_bias_new_s)

        grad_hidden_bias_new = -(tf.reduce_mean(positive_hidden_probs - negative_hidden_activations, 0))
        grad_hidden_bias_new = tf.scalar_mul(self.args.lr_h_bias*lr_mod, grad_hidden_bias_new)

        ## add sparsity later

        grads = [grad_w_new_b, grad_visible_bias_new_b, grad_w_new_s,
                grad_hidden_bias_new, grad_visible_bias_new_s]

        return grads

    def free_energy(self, inputs_bern, inputs_sm):
        fe_b = -tf.squeeze(tf.matmul(inputs_bern, tf.expand_dims(self.a_b, -1)))\
            -tf.reduce_sum(tf.math.log(1 + tf.exp(self.b + tf.matmul(inputs_bern, self.W_b))), axis=1)
        fe_b_mean = tf.reduce_mean(fe_b)

        list_va = []
        for i in np.arnage(self.args.batch_size):
            vamt = inputs_sm[i,:,:]
            s1 = -tf.linalg.trace(tf.matmul(self.a_sm, vamt, transpose_a=True))
            list_va.append(s1)

        fe_s_bias = tf.convert_to_tensor(list_va)

        batch_list = []
        for i in np.arange(self.args.batch_size):
            vmat = inputs_sm[i, :, :]
            h_list=[]
            for h in range(self.W_sm.shape[0]):
                W_mat = self.W_sm[h, :, :]
                mul = tf.linalg.trace(tf.matmul(vmat, W_mat, transpose_a=True)) + self.b[h]
                exp_mul = tf.exp(mul)
                log = tf.math.log(1 + exp_mul)
                h_list.append(log)
            h_sum = -tf.reduce_sum(h_list)
            batch_list.append(h_sum)

        fe_s_fe = tf.convert_to_tensor(batch_list)
        fe_s = fe_s_bias + fe_s_fe
        fe_s_mean = tf.reduce_mean(fe_s)

        return fe_b_mean, fe_s_mean

    def gibbs_sampling(self, inputs_bern, inputs_sm, steps=1):

        for step in range(steps):
            h_tensor, h_prob = self.Positive_Bern_SM(inputs_bern, inputs_sm=inputs_sm)
            h_sample = self.sample_h_distribution(h_prob)

            visible_probs_b, visible_acts_b = self.Negative_Bern(inputs=h_sample)
            visible_probs_s = self.Negative_Bern_SM(inputs_sm, hid_acts=h_sample)
            visible_acts_s = visible_acts_s

        return visible_acts_b, visible_acts_s

class RBM(keras.Model):
    def __init__(self, args):
        super(RBM, self).__init__()

        self.args = args
        
        self.n_bern = self.args.n_bernoulli
        self.n_softmax = self.args.n_softmax
        self.n_cat_sm = self.args.n_cat_sm
        self.n_hidden = self.args.n_hidden

        w_init = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        self.W_b = tf.Variable(initial_value=w_init(shape=(self.n_bern, self.n_hidden), dtype= tf.float32),
                               trainable=True, name='weight_matrix_bernoulli')
        self_W_sm = tf.Variable(initial_value=w_init(shape=(self.n_hidden, self.n_cat_sm, self.n_softmax),
                                dtype=tf.float32), trainable=True, name='weight_matrix_softmax')
        a_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
        self.a_b = tf.Variable(initial_value=a_init(shape=self.n_bern,), dtype=tf.float32),
                               trainable=True, name='visible_bias_bernoulli')
        b_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
        self.b - tf.Variable(initial_value=b_init(shape=(self.n_hidden,), dtype=tf.float32),
                            trainable=True, name='hidden_bias')

        self.Negative_Bern = Negative_Bern(args=self.args, W_b=self.W_b, a_b=self.a_b)
        self.Positive_Bern_SM = Positive_Bern_SM(args=self.args, W_b=self.W_b, W_sm=self.W_sm, b=self.b)
        self.Negative_Bern_SM = Negative_Bern_SM(args=self.args, W_sm=self.W_sm, a_sm= self.a_sm)
        self.Functions = Functions(args=self.args, W_b=self.W_b, W_sm=self.W_sm, a_b=self.a_b, a_sm=self.a_sm, b=self.b)

    def compile(self, optimizer):
        super(RBM, self).compile()
        self.optimizer = optimizer

    def train_step(self, inputs):
        inputs_bern = inputs['inputs_bern']
        inputs_sm = inputs['inputs_sm']

        grads = self.Functions.contrastive_divergence(inputs_bern=inputs_bern, inputs_sm=inputs_sm)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        fe_b, fe_sm = self.Functions.free_energy(inputs_bern=inputs_bern, inputs_sm=inputs_sm)
        bern_proportion = 0.1
        fe_loss = -(bern_proportion*fe_b + (1-bern_proportion)*fe_sm)

        return {'FE Loss': fe_loss}

    def call(self, inputs):
        inputs_bern = inputs['inputs_bern']
        inputs_sm = inputs['inputs_sm']

        v_acts_b, v_acts_sm = self.Functions.gibbs_sampling(inputs_bern=inputs_bern,inputs_sm=inputs_sm)

        #add free energy as a loss function
        fe_b, fe_sm = self.Functions.free_energy(inputs_bern=inputs_bern, inputs_sm=inputs_sm)
        fe_b_mod, fe_sm_mod = self.Functions.free_energy(inputs_bern=v_acts_b, inputs_sm=v_acts_sm)
        fe_diff_b = (fe_b_mod - fe_b)**2
        fe_diff_sm = (fe_sm_mod - fe_sm)**2
        bern_proportion = 0.1
        fe_loss = -(bern_proportion*fe_diff_b + (1-bern_proportion)*fe_diff_sm)
        self.add_loss(fe_loss)

        return fe_loss





        




        



                            

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





            







    

