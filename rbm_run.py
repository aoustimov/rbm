
import os
from rbm_bern_sm_v5 import RBM
import tensroflow argparse imageio 
import kerastuner as kt
import pandas as pd
from util import convert_to_onehot
import numpy as np

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='RBM')
parser.add_argument('--checkpoint_path', type=str,
default="log directory/check_{epoch}/cp-{epoch:02d}.ckpt", help="path to save model")
parser.add_argument('--save', type=bool, default=True, help='saves model and checkpoints')
parser.add_argument('--load', type=bool, default=False, help='loads model and checkpoints')
parser.add argument('--dist_type_vis', type=str, default='bernoulli', help='visible distribution type.')
parser.add_argument('--dist_type_hid', type=str, default='bernoulli', help='hidden distribution type.')
parser.add_argument('--cd_k', type=int, default=3, help='number of cd interations')
parser.add argument('--epoch',type=int, default=1, help='nuber of training loops')
parser.add argument('--v_marg_steps', type=int, default=250, help='number of sampling iterations')
parser.add argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add argument('--use_tuner', type=bool, default=False, help='use keras hyperparameter_tuner')
parser.add argument('--train_using_tuner', type=bool, default=False, help='use resuls from keras tuner to train new model')
parser.add argument('--optimizer', type=str, default="SGD", help='choose to use Adam of SGD')

parser.add argument('--n_bernoulli', type=int, dfault=100, help='number of bernoulli visible units')
parser.add argument('--n_cat_sm', type=int, dfault=4, help='number of softmax categories in each unit')
parser.add argument('--n_softmax', type=int, dfault=100, help='number of softmax visible units')
parser.add argument('--n_hidden', type=int, dfault=10, help='number of bernoulli hidden units')


