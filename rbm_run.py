
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


