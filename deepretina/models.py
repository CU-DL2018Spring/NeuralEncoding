"""
Construct Keras models
"""
from __future__ import absolute_import, division, print_function
from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Reshape, SimpleRNN, Conv1D, Add
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianNoise
from keras.regularizers import l1, l2
from deepretina import activations
from keras.initializers import RandomNormal

__all__ = ['bn_cnn', 'bn_spat_cnn', 'linear_nonlinear', 'ln', 'nips_cnn', 'fc_rnn', 'fc_rnn_large', 'spatial_cnn', 'copy_cnn', 'conv_to_lstm', 'conv_to_rnn', 'fc_lstm', 'conv_lstm', 'fc_rnn_large', 'tcn', 'cn_tcn']


def bn_layer(x, nchan, size, l2_reg, sigma=0.05):
    """An individual batchnorm layer"""
    n = int(x.shape[-1]) - size + 1
    y = Conv2D(nchan, size, data_format="channels_first", kernel_regularizer=l2(l2_reg))(x)
    y = Reshape((nchan, n, n))(BatchNormalization(axis=-1)(Flatten()(y)))
    return Activation('relu')(GaussianNoise(sigma)(y))


def bn_cnn(inputs, n_out, l2_reg=0.01):
    """Batchnorm CNN model"""
    y = bn_layer(inputs, 8, 15, l2_reg)
    y = bn_layer(y, 8, 11, l2_reg)
    y = Dense(n_out, use_bias=False)(Flatten()(y))
    outputs = Activation('softplus')(BatchNormalization(axis=-1)(y))
    return Model(inputs, outputs, name='BN-CNN')

def bn_spat_cnn(inputs, n_out, *args, l2_reg=0.01):
    """Batchnorm CNN model"""
    y = bn_layer(inputs, 8, 15, l2_reg)
    y = bn_layer(y, 8, 11, l2_reg)
    y = Dense(n_out, use_bias=False)(Flatten()(y))
    outputs = Activation('softplus')(BatchNormalization(axis=-1)(y))
    return Model(inputs, outputs, name='BN-SPAT-CNN')

def linear_nonlinear(inputs, n_out, *args, activation='softplus', l2_reg=0.01):
    """A linear-nonlinear model"""

    # a default activation
    if activation in ('softplus', 'sigmoid', 'relu', 'exp'):
        nonlinearity = Activation(activation)

    # is a nonlinearity class
    elif activation.lower() == ('rbf', 'psp'):
        nonlinearity = activations.__dict__[activation](*args)

    # one of the custom deepretina activations
    elif activation in activations.__all__:
        nonlinearity = activations.__dict__[activation]

    # a custom class
    else:
        nonlinearity = activation

    y = Flatten()(inputs)
    y = Dense(n_out, kernel_regularizer=l2(l2_reg))(y)
    outputs = nonlinearity(y)

    return Model(inputs, outputs, name=f'LN-{str(activation)}')


def nips_cnn(inputs, n_out):
    """NIPS 2016 CNN Model"""
    # injected noise strength
    sigma = 0.1

    # first layer
    y = Conv2D(16, 15, data_format="channels_first", kernel_regularizer=l2(1e-3))(inputs)
    y = Activation('relu')(GaussianNoise(sigma)(y))

    # second layer
    y = Conv2D(8, 9, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
    y = Activation('relu')(GaussianNoise(sigma)(y))

    y = Flatten()(y)
    y = Dense(n_out, init='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)
    outputs = Activation('softplus')(y)

    return Model(inputs, outputs, name='NIPS_CNN')

from keras.layers import RNN
def fc_rnn(inputs, n_out, *args):
    """Fully Connected RNN (Batty et al.)"""
    print("input shape = ", inputs.shape)
    y = SimpleRNN(50, activation='relu', return_sequences=True, kernel_regularizer=l2(1e-3), recurrent_regularizer=l2(1e-3))(inputs)
    y = SimpleRNN(50, activation='relu', kernel_regularizer=l2(1e-3), recurrent_regularizer=l2(1e-3))(y)
    y = Dense(n_out, init='normal')(y)
    outputs = Activation('softplus')(y)

    return Model(inputs, outputs, name="FC_RNN")

def fc_rnn_large(inputs, n_out, *args):
    """Fully Connected RNN, scaled with input size increase (Batty et al.)"""
    print("input shape = ", inputs.shape)
    y = SimpleRNN(80, activation='relu', return_sequences=True, kernel_regularizer=l2(1e-3), recurrent_regularizer=l2(1e-3))(inputs)
    y = SimpleRNN(80, activation='relu', kernel_regularizer=l2(1e-3), recurrent_regularizer=l2(1e-3))(y)
    y = Dense(n_out, init='normal')(y)
    outputs = Activation('softplus')(y)

    return Model(inputs, outputs, name="FC_RNN_LARGE")

from keras.layers import LSTM
def fc_lstm(inputs, n_out, *args):
    """Fully Connected LSTM (Batty et al.)"""
    print("input shape = ", inputs.shape)
    y = LSTM(50, activation='relu', return_sequences=True)(inputs)
    y = LSTM(50, activation='relu')(y)
    y = Dense(n_out, init='normal')(y)
    outputs = Activation('softplus')(y)

    return Model(inputs, outputs, name="FC_LSTM")


from keras.layers import ConvLSTM2D
def conv_lstm(inputs, n_out, *args):
    """Convolutional LSTM (Shi et al.)"""
    print("input shape = ", inputs.shape)
    y = ConvLSTM2D(4, 15, data_format="channels_first", activation='relu', return_sequences=False)(inputs)
    #y = ConvLSTM2D(8, 11, data_format="channels_first", activation='relu')(y)
    y = Flatten()(y)
    y = Dense(n_out, init='normal')(y)
    outputs = Activation('softplus')(y)

    return Model(inputs, outputs, name="CONV_LSTM")


from keras.layers import Dropout
def spatial_cnn(inputs, n_out, *args, l2_reg=0.01):
    """Standard CNN with no temporal dimension"""
    print(inputs.shape)
    y = Conv2D(4, 7, data_format="channels_first", activation='relu', kernel_regularizer=l2(1e-3))(inputs)
    y = Dense(n_out)(Flatten()(y))
    outputs = Activation('softplus')(y)
    return Model(inputs, outputs, name='SPAT_CNN')

from keras.layers import Dropout
def copy_cnn(inputs, n_out, *args, l2_reg=0.01):
    """Standard CNN with no batch norm"""
    print(inputs.shape)
    y = Conv2D(8, 15, data_format="channels_first", kernel_regularizer=l2(1e-3))(inputs)
    y = Activation('relu')(GaussianNoise(sigma)(y))
    y = Conv2D(8, 11, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
    y = Activation('relu')(GaussianNoise(sigma)(y))
    y = Dense(n_out, use_bias=False)(Flatten()(y))
    outputs = Activation('softplus')(y)
    return Model(inputs, outputs, name='COPY_CNN')

from keras.layers import TimeDistributed
from keras.layers import LSTM
def conv_to_lstm(inputs, n_out, *args, l2_reg=0.01):
    """Convolution on each stimulus, then pass sequence to an LSTM"""
    print("conv_to_lstm input shape = ", inputs.shape)
    # Applies this conv layer to each stimulus in the sequence individually
    y = TimeDistributed(Conv2D(8, 7, data_format="channels_first", activation='relu', kernel_regularizer=l2(1e-3)), input_shape=(40, 1, 50, 50))(inputs)
    print("after first conv layer", y.shape)
    #y = TimeDistributed(Conv2D(8, 7, data_format="channels_first", activation='relu', kernel_regularizer=l2(1e-3)))(y)
    #print("after second conv layer", y.shape)
    # Flatten feature maps to pass to LSTM 
    y = TimeDistributed(Flatten())(y)
    print("after flatten layer", y.shape)
    y = LSTM(50, activation='relu')(y)
    print("after lstm layer", y.shape)
    y = Dense(n_out, init='normal')(y)
    print("after dense layer", y.shape)
    outputs = Activation('softplus')(y)
    print("after activation layer", outputs.shape)
    return Model(inputs, outputs, name="CONV_TO_LSTM")

def conv_to_rnn(inputs, n_out, *args, l2_reg=0.01):
    """Convolution on each stimulus, then pass sequence to an RNN"""
    print("conv_to_rnn input shape = ", inputs.shape)
    # Applies this conv layer to each stimulus in the sequence individually
    y = TimeDistributed(Conv2D(8, 15, data_format="channels_first", activation='relu', kernel_regularizer=l2(1e-3)), input_shape=(40, 1, 50, 50))(inputs)
    print("after first conv layer", y.shape)
    y = TimeDistributed(Conv2D(8, 11, data_format="channels_first", activation='relu', kernel_regularizer=l2(1e-3)))(y)
    #print("after second conv layer", y.shape)
    # Flatten feature maps to pass to LSTM 
    y = TimeDistributed(Flatten())(y)
    print("after flatten layer", y.shape)
    y = SimpleRNN(50, activation='relu', kernel_initializer='random_normal', recurrent_initializer='random_normal')(y)
    #print("after lstm layer", y.shape)
    y = Dense(n_out, init='normal')(y)
    #print("after dense layer", y.shape)
    outputs = Activation('softplus')(y)
    #print("after activation layer", outputs.shape)
    return Model(inputs, outputs, name="CONV_TO_RNN")

def tcn_block(inputs, n_out, dilation, padding = 'same', kernel_size=2, dropout=0.3, l2_reg = 1e-3, flatten = False):
    """ Basic building block of TCN (Bai et al.)"""
    print("TCN input shape = ", inputs.shape)
    conv = Conv1D(filters = n_out, kernel_size = kernel_size, strides = 1, dilation_rate = dilation, padding = padding,
                  kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg),
                  kernel_initializer=RandomNormal(stddev=0.01))(inputs)
    # Dilated CNN
    conv = Activation('relu')(conv)
    conv = Dropout(dropout)(conv)
    print("Conv 1st layer", conv.shape)
    conv = Conv1D(filters = n_out, kernel_size = kernel_size, strides = 1, dilation_rate = dilation, padding = padding,
                  kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg),
                  kernel_initializer=RandomNormal(stddev=0.01))(conv)
    conv = Activation('relu')(conv)
    conv = Dropout(dropout)(conv)
    print("Conv 2nd layer", conv.shape)
    if flatten:
        conv = Dense(n_out, init='normal')(Flatten()(conv))
    print("Conv out layer", conv.shape)

    fcn = Conv1D(n_out,kernel_size=1,padding='same',
                 kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg),
                 kernel_initializer=RandomNormal(stddev=0.01))(inputs)

    # Parallel Fully-Convolutional Network
    if flatten:
        print("FCN 1st layer", fcn.shape)
        fcn = Dense(n_out, init='normal')(Flatten()(fcn))
        print("FCN out layer", fcn.shape)
    
    # Merge ConvNet and FCN
    outputs = Add()([conv, fcn])
    outputs = Activation('relu')(outputs)
    print("Output layer ", outputs.shape)
    return outputs

def tcn(inputs, n_out, *args):
    """TCN by Bai et al. Called following a conv-net """
    """
    print("TCN input shape = ", inputs.shape)
    conv = Conv1D(filters = 8, kernel_size = 3, strides = 1, dilation_rate = 1, padding = 'same',
                  kernel_regularizer=l2(1e-3), bias_regularizer=l2(1e-3),
                  kernel_initializer=RandomNormal(stddev=0.01))(inputs)
    conv = Activation('relu')(conv)
    conv = Dropout(0.3)(conv)
    print("Conv 1st layer", conv.shape)
    conv = Conv1D(filters = 5, kernel_size = 2, strides = 1, dilation_rate = 2, padding = 'same',
                  kernel_regularizer=l2(1e-3), bias_regularizer=l2(1e-3),
                  kernel_initializer=RandomNormal(stddev=0.01))(conv)
    conv = Activation('relu')(conv)
    conv = Dropout(0.3)(conv)
    print("Conv 2nd layer", conv.shape)
    conv = Dense(n_out, init='normal')(Flatten()(conv))
    print("Conv out layer", conv.shape)

    fcn = Conv1D(n_out,kernel_size=1,padding='same',
                 kernel_regularizer=l2(1e-3), bias_regularizer=l2(1e-3),
                 kernel_initializer=RandomNormal(stddev=0.01))(inputs)
    print("FCN 1st layer", fcn.shape)
    fcn = Dense(n_out, init='normal')(Flatten()(fcn))
    print("FCN out layer", fcn.shape)

    outputs = Add()([conv, fcn])
    #outputs = Activation('relu')(fcn + conv)
    print("Output layer ", outputs.shape)
    """
    print("Creat TCN 1")
    y = tcn_block(inputs, 10, dilation = 1, padding='same', kernel_size=3)
    print("-------------------------------------------------------")
    print("Creat TCN 2")
    y = tcn_block(y, 5, dilation = 2, padding='same', kernel_size=3)
    print("Creat TCN 3")
    print("-------------------------------------------------------")
    outputs = tcn_block(y, n_out, dilation = 4, padding='same', kernel_size=3, flatten=True)

    return Model(inputs, outputs, name="TCN")


def cn_tcn(inputs, n_out, *args):
    """TCN by Bai et al. Called following a conv-net """
    sigma = .05
    # Perform convolution on each stimulus, then pass flattened feature maps as sequence to TCN
    # Applies this conv layer to each stimulus in the sequence individually
    y = TimeDistributed(Conv2D(4, 15, data_format="channels_first", kernel_regularizer=l2(1e-3)), input_shape=(40, 1, 50, 50))(inputs)
    y = Activation('relu')(GaussianNoise(sigma)(y))
    print("after first conv layer", y.shape)
    #y = TimeDistributed(Conv2D(8, 7, data_format="channels_first", activation='relu', kernel_regularizer=l2(1e-3)))(y)
    #print("after second conv layer", y.shape)
    # Flatten feature maps to pass to TCN
    y = TimeDistributed(Flatten())(y)
    print("after flatten layer", y.shape)
    conv = Conv1D(filters = 8, kernel_size = 3, strides = 1, dilation_rate = 1, padding = 'same',
                  kernel_regularizer=l2(1e-3), bias_regularizer=l2(1e-3),
                  kernel_initializer=RandomNormal(stddev=0.01))(y)
    conv = Activation('relu')(GaussianNoise(sigma)(conv))
    conv = Dropout(0.3)(conv)
    print("Conv 1st layer", conv.shape)
    conv = Conv1D(filters = 5, kernel_size = 2, strides = 1, dilation_rate = 2, padding = 'same',
                  kernel_regularizer=l2(1e-3), bias_regularizer=l2(1e-3),
                  kernel_initializer=RandomNormal(stddev=0.01))(conv)
    conv = Activation('relu')(GaussianNoise(sigma)(conv))
    conv = Dropout(0.3)(conv)
    print("Conv 2nd layer", conv.shape)
    conv = Dense(n_out, init='normal')(Flatten()(conv))
    print("Conv out layer", conv.shape)

    fcn = Conv1D(n_out,kernel_size=1,padding='same',
                 kernel_regularizer=l2(1e-3), bias_regularizer=l2(1e-3),
                 kernel_initializer=RandomNormal(stddev=0.01))(y)
    print("FCN 1st layer", fcn.shape)
    fcn = Dense(n_out, init='normal')(Flatten()(fcn))
    print("FCN out layer", fcn.shape)

    outputs = Add()([conv, fcn])
    #outputs = Activation('relu')(fcn + conv)
    print("Output layer ", outputs.shape)
    return Model(inputs, outputs, name="TCN")



# aliases
ln = linear_nonlinear
