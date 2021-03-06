"""
Core tools for training models
"""
import os
from datetime import datetime
import deepdish as dd
import keras.callbacks as cb
from keras.layers import Input
from deepretina import metrics, activations
from deepretina.experiments import loadexpt, CELLS
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
import numpy as np
import tensorflow as tf

__all__ = ['train', 'load']


def load(filepath):
    """Reload a keras model"""
    objects = {k: activations.__dict__[k] for k in activations.__all__}
    objects.update({k: metrics.__dict__[k] for k in metrics.__all__})
    return load_model(filepath, custom_objects=objects)


def train(model, expt, stim, model_args=(), lr=1e-2, bz=5000, nb_epochs=500, val_split=0.05, cells=None):
    """Trains a model"""
    if cells is None:
        width = None
        cells = CELLS[expt]
        cellname = ''
    else:
        width = 11
        cellname = f'cell-{cells[0]+1:02d}'

    # Get rid of temporal dimension
    if 'spatial' in model_args:
        print("spatial")
        window = 1
    # Set increased history length
    elif "window" in model_args:
        window = 80
    else:
        window = 40

    # load experimental data
    data = loadexpt(expt, cells, stim, 'train', window, 6000, cutout_width=width)

    newX = None
    # In case we want to use predicted firing rate as training data
    #if 'use_predict' in model_args:
    #    weight_path = '../results/SPAT_CNN__15-10-07_naturalscene_2018.04.13-05.16/weights-008--0.682.h5'
    #    cnn_mdl = load(weight_path)
    #    X_new = cnn_mdl.predict(data.X)
    #    print (X_new.shape)
    #    return

    # Add channels, and set window to temporal dimension for conv_lstm
    if 'add_dim' in model_args:
        print("add_dim")
        input_shape = data.X.shape
        print(input_shape)
        # All three methods work, decide later if any of them are preferable
        newX = data.X[:,:,np.newaxis,:,:]

    # flatten if
    elif 'flatten' in model_args:
        print("flatten!")
        input_shape = data.X.shape
        print(input_shape)
        newX = data.X.reshape(input_shape[0],input_shape[1],input_shape[2]*input_shape[3])
    else:
        newX = data.X
    # try different val set
    """
    train_cutoff = int(newX.shape[0]*val_split)
    print(train_cutoff)
    X_train = newX[train_cutoff:]
    y_train = data.y[train_cutoff:]
    X_val = newX[:train_cutoff]
    y_val = data.y[:train_cutoff]
    print(X_train.shape)
    
    print(X_val.shape)
    """
    # build the model
    n_cells = data.y.shape[1]
    x = Input(shape=newX.shape[1:])
    print("newX.shape[1:] = ", newX.shape[1:])
    mdl = model(x, n_cells, *model_args)

    if '2_GPU' in model_args:
        print("2 GPU")
        mdl = multi_gpu_model(mdl, gpus=2)

    # compile the model
    run_opts = tf.RunOptions(report_tensor_allocations_upon_oom=True)
    if 'mse' in model_args:
        print("mse!")
        mdl.compile(loss='mean_squared_error', optimizer=Adam(lr), metrics=[metrics.cc, metrics.rmse, metrics.fev], options=run_opts)
    else:
        mdl.compile(loss='poisson', optimizer=Adam(lr), metrics=[metrics.cc, metrics.rmse, metrics.fev], options=run_opts)

    # store results in this directory
    name = '_'.join([mdl.name, cellname, expt, stim, datetime.now().strftime('%Y.%m.%d-%H.%M')])
    base = f'../results/{name}'
    os.mkdir(base)

    # define model callbacks
    cbs = [#cb.ModelCheckpoint(os.path.join(base, 'weights-{epoch:03d}-{val_loss:.3f}.h5')),
           cb.TensorBoard(log_dir=base, histogram_freq=1, batch_size=bz, write_grads=True),
           cb.ReduceLROnPlateau(min_lr=0, factor=0.2, patience=10),
           cb.CSVLogger(os.path.join(base, 'training.csv')),
           cb.EarlyStopping(monitor='val_loss', patience=20),
           #cb.LearningRateScheduler((lambda epoch, lr : lr * 0.97**epoch), verbose=1)
          ]

    # train
    history = mdl.fit(x=newX, y=data.y, batch_size=bz, epochs=nb_epochs,
                      callbacks=cbs, validation_split=val_split, shuffle=True)
    dd.io.save(os.path.join(base, 'history.h5'), history.history)

    return history
