
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
parser.add_argument('--dist_type_vis', type=str, default='bernoulli', help='visible distribution type.')
parser.add_argument('--dist_type_hid', type=str, default='bernoulli', help='hidden distribution type.')
parser.add_argument('--cd_k', type=int, default=3, help='number of cd interations')
parser.add_argument('--epoch',type=int, default=1, help='nuber of training loops')
parser.add_argument('--v_marg_steps', type=int, default=250, help='number of sampling iterations')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--use_tuner', type=bool, default=False, help='use keras hyperparameter_tuner')
parser.add_argument('--train_using_tuner', type=bool, default=False, help='use resuls from keras tuner to train new model')
parser.add_argument('--optimizer', type=str, default="SGD", help='choose to use Adam of SGD')

parser.add_argument('--n_bernoulli', type=int, dfault=100, help='number of bernoulli visible units')
parser.add_argument('--n_cat_sm', type=int, dfault=4, help='number of softmax categories in each unit')
parser.add_argument('--n_softmax', type=int, dfault=100, help='number of softmax visible units')
parser.add_argument('--n_hidden', type=int, dfault=10, help='number of bernoulli hidden units')
parser.add_argument('--proba_activation_toggle', type=str, default="not_activation",
                    help='changing bias to stochastic activation values')
parser.add_argument('--l2_param', type=float, default=0.0001, help='l2 param')
parser.add_argument('--lr_red_factor', type=float, default=1.0, help='lr_red_factor')
parser.add_argument('--lr_b', type=float, default=0.0015, help='lr for bernoulli weights')
parser.add_argument('--lr_s', type=float, default=0.0015, help='lr for somftmax wieghts')
parser.add_argument('--lr_b_bias', type=float, default=0.0015, help='lr for bernoulli bais')
parser.add_argument('--lr_s_bias', type=float, default=0.0015, help='lr for softmax bias')
parser.add_argument('--lr_h_bias', type=float, default=0.0015, help='lr for the hidden bias')
parser.add_argument('--loss_type', type=str, default='fe_reconstruction_cost', help='loss type')

args = parser.parse_args()

## building model for tuner
def build_model(hp):
    # number of hidden units
    hp_n_hid = hp.Int('n_hidden', min_value=1000, max_value=1010, step=10)
    args.n_hidden = hp_n_hid
    # learning rates
    hp_lr = hp.Choice('learning_rate', values=[1e-2,1e-3, 1e-4])
    args.lr = hp_lr
    hp_l2_param = hp.Choice('l2_param', values=[1e-2,1e-3, 1e-4])
    args.l2_param = hp_l2_param
    hp_lr_red_factor = hp.Choice('lr_red_factor', values=[0.5, 0.25])
    args.lr_red_factor  = hp_lr_red_factor
    hp_lr_b = hp.Choice('lr_b', values=[1e-2,1e-3, 1e-4])
    args.lr_b = hp_lr_b
    hp_lr_s = hp.Choice('lr_s', values=[1e-2,1e-3, 1e-4])
    args.lr_s = hp_lr_s
    hp_lr_b_bias = hp.Choice('lr_b_bias', values=[1e-2,1e-3, 1e-4])
    args.lr_b_bias = hp_lr_b_bias
    hp_lr_s_bias = hp.Choice('lr_s_bias', values=[1e-2,1e-3, 1e-4])
    args.lr_s_bias = hp_lr_s_bias
    hp_lr_h_bias = hp.Choice('lr_h_bias', values=[1e-2,1e-3, 1e-4])
    args.lr_h_bias = hp_lr_h_bias

    rbm(RBM(args))
    rbm.compile(optimizer-tf.keras.optimizers.SGD(learning_rate=1e-3))
    return rbm

##
Class MyTuner(kt.Tuner):
    def run_trial(self, trial, train_ds):
        hp = trial.hyperparameters
        train_ds = train_ds.batch(10, drop_remainder=True)
        model = self.hypermodel.build(trial.hyperparameters)
        lr = hp.Float('learning_rate', le-3,1e-2,sampling='log', default= le-3)
        epoch_loss_metric = tf.keras.metrics.Mean()

        @tf.function
        def run_train_step(data):
            loss = model(data)
            print('loss = ' + str(loss))
            gradients = model.Functions.constrastive_divergence(input_bern= data['inputs_bern')])
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            epoch_loss_metric.update_state(loss)
            return loss

        for epoch in range(10):
            print('Epoch: {}'.format(epoch))

            self.on_epoch_begin(trial, model, epoch, logs{})
            for batch, data in enumberate(train_ds):
                self.on_batch_begin(trial, model, batch, logs={})
                batch_loss = float(run_train_step(data))
                self.on_batch_end(trial, model, batch, logs={'loss':batch_loss"})

                if batch % 100 == 0:
                    loss = epoch_loss_metric.result().numpy()
                    print('Batch: {}, Average Loss: {}'.format(batch, loss))

            epoch_loss = epoch_loss_metric.result().numpy()
            self.on_epoch_end(trial, model, epoch, logs={'loss': epoch_loss})
            epoch_loss_metric.reset_states()

def main():
    tuner = MyTuner(
        oracle = kt.oracles.BayesianOptimization(
            objective = kt.Objective('loss', 'min'),
            max_trials=2),
        hypermodel = build_model,
        directory = '/workdir/data/rbm',
        project_name = 'training_1')
        )
    
    data = pd.read_csv('input_dataset.csv')
    bern_data = np.asarray(data).astype("float32")

    train_data = tf.data.Dataset.from_tensor_slices({"inputs_bern":bern_data})
    train_dataset train_data.shuffle(buffer_size=1024)

    tuner.search(train_ds=train_dataset)
    best_hps = tuner_get_best_hyperparameters()[0]
    best_model = tuner.get_best_models()[0]
    best_model.save_weights('./model_name.hd5')
    print(best_hps.values)

if __name__ == '__main__':
    main()


        




    





