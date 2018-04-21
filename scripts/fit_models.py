"""
fit LN and BN_CNN models
"""
import sys
sys.path.append('../')
import os
import functools
import argparse
import tensorflow as tf
import keras.backend as K
import tableprint as tp
from deepretina.core import train
from deepretina.models import bn_cnn, linear_nonlinear, nips_cnn, fc_rnn, spatial_cnn, copy_cnn, conv_to_lstm, fc_lstm, conv_lstm, tcn, fc_rnn_large, cn_tcn, bn_rnn
from deepretina.models import *


def context(func):
    def wrapper(*args, **kwargs):
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as sess:
                K.set_session(sess)
                result = func(*args, **kwargs)
        return result
    return wrapper


@context
def fit_bn_cnn(expt, stim):
    train(bn_cnn, expt, stim, lr=1e-2, nb_epochs=250, val_split=0.05)

@context
def fit_bn_rnn(expt, stim):
    train(bn_rnn, expt, stim, lr=1e-2,model_args=("add_dim", "2_GPU","mse"), nb_epochs=250, val_split=0.05)

@context
def fit_bn_spat_cnn(expt, stim):
    train(bn_spat_cnn, expt, stim, model_args=("spatial"), lr=1e-2, nb_epochs=250, val_split=0.05)

@context
def fit_nips_cnn(expt, stim):
    train(nips_cnn, expt, stim, lr=1e-2, nb_epochs=250, val_split=0.05)

@context
def fit_ln(expt, ci, stim, activation, l2_reg=0.1):

    if activation.lower() == 'rbf':
        model_args = (30, 6)
    else:
        model_args = ()

    model = functools.partial(linear_nonlinear, activation=activation, l2_reg=l2_reg)
    #tp.banner(f'Training LN-{activation}, expt {args.expt}, {args.stim}, cell {ci+1:02d}')
    train(model, expt, stim, model_args=model_args, lr=1e-2, nb_epochs=500, val_split=0.05, cells=[ci])

@context
def fit_fc_rnn(expt, stim):
    train(fc_rnn, expt, stim, model_args=("flatten", "mse"), lr=1e-3, nb_epochs=250, val_split=0.05)

@context
def fit_fc_rnn_large(expt, stim):
    train(large_fc_rnn, expt, stim, model_args=("flatten", "mse"), lr=1e-3, nb_epochs=250, val_split=0.05)

@context
def fit_fc_lstm(expt, stim):
    train(fc_lstm, expt, stim, model_args=("flatten", "mse"), lr=1e-4, nb_epochs=250, val_split=0.05)

@context
def fit_conv_lstm(expt, stim):
    train(conv_lstm, expt, stim, model_args=("add_dim", "2_GPU"), lr=1e-3, bz=128, nb_epochs=250, val_split=0.05)

@context
def fit_spatial_cnn(expt, stim):
    train(spatial_cnn, expt, stim, model_args=("spatial"), lr=1e-2, nb_epochs=250, val_split=0.05)

@context
def fit_copy_cnn(expt, stim):
    train(copy_cnn, expt, stim, model_args=(), lr=1e-3, nb_epochs=250, val_split=0.05)

@context
def fit_conv_to_lstm(expt, stim):
    train(conv_to_lstm, expt, stim, model_args=("add_dim"), lr=0.1, nb_epochs=250, val_split=0.05, bz=2048)

@context
def fit_conv_to_rnn(expt, stim):
    train(conv_to_rnn, expt, stim, model_args=("add_dim","2_GPU","mse"), lr=1e-4, nb_epochs=250, val_split=0.05, bz=1024)

@context
def fit_tcn(expt, stim):
    train(tcn, expt, stim, model_args=("flatten", "mse"), lr=1e-3, nb_epochs=250, val_split=0.05)

@context
def fit_cn_tcn(expt, stim):
    train(cn_tcn, expt, stim, model_args=("add_dim", "2_GPU", "mse"), lr=1e-3, nb_epochs=250, val_split=0.05)


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # expts = ('15-10-07', '15-11-21a', '15-11-21b')
    # stims = ('whitenoise', 'naturalscene')

    parser = argparse.ArgumentParser(description='Train a BN_CNN model')
    parser.add_argument('--expt', help='Experiment date (e.g. 15-10-07)')
    parser.add_argument('--stim', help='Stimulus class (e.g. naturalscene)')
    parser.add_argument('--model', help='Model architecture (e.g. BN_CNN or LN_softplus)')
    parser.add_argument('--cell', help='Cell index (only for LN models)')
    args = parser.parse_args()

    if args.model.upper() == 'BN_CNN':
        tp.banner(f'Training BN_CNN, expt {args.expt}, {args.stim}')
        fit_bn_cnn(args.expt, args.stim)
    elif args.model.upper() == 'BN_RNN':
        fit_bn_rnn(args.expt, args.stim)
    elif args.model.upper() == 'BN_SPAT_CNN':
        fit_bn_spat_cnn(args.expt, args.stim)
    elif args.model.upper() == 'FC_RNN':
        fit_fc_rnn(args.expt, args.stim)
    elif args.model.upper() == 'FC_RNN_LARGE':
        fit_fc_rnn_large(args.expt, args.stim)
    elif args.model.upper() == 'FC_LSTM':
        fit_fc_lstm(args.expt, args.stim)
    elif args.model.upper() == 'SPATIAL_CNN':
        fit_spatial_cnn(args.expt, args.stim)
    elif args.model.upper() == 'COPY_CNN':
        fit_copy_cnn(args.expt, args.stim)
    elif args.model.upper() == 'CONV_TO_LSTM':
        fit_conv_to_lstm(args.expt, args.stim)
    elif args.model.upper() == 'CONV_TO_RNN':
        fit_conv_to_rnn(args.expt, args.stim)
    elif args.model.upper() == 'CONV_LSTM':
        fit_conv_lstm(args.expt, args.stim)
    elif args.model.upper() == 'TCN':
        fit_tcn(args.expt, args.stim)
    elif args.model.upper() == 'CN_TCN':
        fit_cn_tcn(args.expt, args.stim)
    elif args.model.upper() == 'NIPS_CNN':
        fit_nips_cnn(args.expt, args.stim)
    elif args.model.split('_')[0].upper() == 'LN':
        activation = args.model.split('_')[1]
        fit_ln(args.expt, int(args.cell), args.stim, activation)

