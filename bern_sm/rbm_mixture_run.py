
import os
from rbm_mixture import RBM
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
parser.add_argument('--batch_size', type=int, default=100, help='batch_size')
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
# parser.add_argument('--loss_type', type=str, default='fe_reconstruction_cost', help='loss type')

args = parser.parse_args()

data = pd.read_csv('dataset_df.csv')
bern_data = np.asarray(data.iloc[:, :args.n_bernoulli]).astype('float32')
sm_data = np.ararray(data.iloc[:, args.n_bernoulli:])
sm_data_one_hot = np.asarray(convert_to_onehot(datain=sm_data,
                                               nb_classes=args.n_cat_sm)).astype('float32')
train_data = tf.data.Dataset.from_tensor_slices({'inputs_bern':bern_data, 
                                                 'inputs_sm':sm_data_one_hot})
train_dataset = train_data.shuffle(buffer_size=1024).batch(args.batch_size,
                                                           drop_remainder=True)
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
    rbm.compile(optimizer-tf.keras.optimizers.SGD(learning_rate=hp_lr))
    return rbm

## custom training loop
def custom_training(model):
    optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr)

    for epoch in range(args.epoch):
        print('Epoch: ', epoch)
        for step, x_batch_train in enumerate(train_dataset):
            loss = model(x_batch_train)
            print('batch number: ' + str(step) + ', loss: ', tf.reduce_mean(loss))
            grads = model.Functions.contrastive_divergence(inputs_bern=x_batch_train['inputs_bern'],
                                                           inputs_sm=x_batch_train['inputs_sm'])
            optimizer.apply_gradients(zip(grads, model.trainable_weights))


def main():
    if args.use_tuner:

        tuner = kt.Hyperband(build_model,
        objective=kt.Objective('FE Loss', 'min')
        max_epochs=2,
        directory='tuner_results'
        project_name='mixture_training_1')

        tuner.search(train_dataset, epochs=args.epoch)
        best_hps = tuner_get_best_hyperparameters(num_trials-1)[0]
        print('optimal hidden units:', best_hps.get('n_hidden'))
        print('optimal learningr rate', best_hps.get('learning_rate'))
        print('optimal l2_param:' best_hps.get('l2_param'))
        print('optimal lr_red_factor:' best_hps.get('lr_red_factor'))
        print('optimal lr_b:', best_hps.get('lr_b'))
        print('optimal lr_s:' , best_hps.get('lr_s')
        print('optimal lr_b_bias', best_hps.get('lr_b_bias'))
        print('optimal lr_s_bias', best_hps.get('lr_s_bias'))
        print('optimal lr_h_bias', best_hps.get('lr_h_bias'))

        if args.train_using_tuner:
            if args.save:
                cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=args.checkpoint_path,
                                                                 save_weights_only=True,
                                                                 save_freq='epoch')
                model = tuner.hypermodel.build(best_hps)
                model.fit(train_dataset, epochs=args.epoch, callbacks=[cp_callback])
    
    else:
        rbm = RBM(args)
        optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr)
        rbm.compile(optimizer=optimizer)

        if args.save:
            custom_training(model=rbm)
            checkpoint_dir = os.path.dirname(args.checkpoint_path)
            rbm.save_weights(checkpoint_dir)
        else:
            custom_training(model=rbm)

if __name__ == "__main__":
    main()

                                                                 
    best_model = tuner.get_best_models()[0]
    best_model.save_weights('./model_name.hd5')
    print(best_hps.values)

if __name__ == '__main__':
    main()


        




    





