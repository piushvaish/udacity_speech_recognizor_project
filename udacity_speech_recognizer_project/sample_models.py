from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, Dropout, MaxPooling1D)


def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2,dropout = 0.1,recurrent_dropout = 0.5, name='rnn')(input_data)
    # TODO: Add batch normalization 
    bn_rnn = BatchNormalization(name ='bn_rnn')(input_data)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim),name ='time_dense')(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu',
        return_sequences=True, name='rnn')(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization(name ='bn_rnn')(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim),name='time_dense')(bn_cnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    rnn_layer = input_data
    for i in range(recur_layers):
        num = str(i+1)
        rnn_name = 'rnn'+num
        bn_name = 'bn_rnn'+num
        rnn_layer = GRU(units, 
                        activation='relu',
                        return_sequences=True, 
                        implementation=2,
                        name=rnn_name,dropout = 0.1,recurrent_dropout = 0.5)(rnn_layer)
        bn_layer = BatchNormalization(name=bn_name)(rnn_layer)
        rnn_layer = bn_layer
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(rnn_layer)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    bidir_rnn = Bidirectional(GRU(units,
                                  activation='relu',
                                  return_sequences=True,
                                  implementation=2,
                                  name='bidir_rnn'), 
                              merge_mode='concat')(input_data)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def final_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a deep network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Specify the layers in your network
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    #print(conv_1d)
    # Add batch normalization
    bn_cnn1 = BatchNormalization(name='bn_conv_1')(conv_1d)
    # add max pooling layer
    max_1d = MaxPooling1D(pool_size=kernel_size, strides=1, padding='same', name='maxpool_1d')(bn_cnn1)
    img_1 = Dropout(rate=0.1)(max_1d)
    #img_1 = Conv1D(128, kernel_size,strides=conv_stride,padding=conv_border_mode,activation='relu',name='conv1d1')(img_1)
    #dense_1 = Dense(64, activation=activations.relu)(img_1)    
    bn_cnn2 = BatchNormalization(name='bn_conv_2')(img_1)
    # dropout = Dropout(0.5)(bn_cnn2)
    # Add a recurrent layer
    bidir_rnn = Bidirectional(GRU(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn'))(bn_cnn2)
    # TODO: Add batch normalization
    bn_rnn3 = BatchNormalization(name='bn_rnn')(bidir_rnn)
    o = TimeDistributed(Dense(64))(bn_rnn3)
    o = Activation(lambda x: relu(x, max_value= 20), name='relu1')(o)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(o)
    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    model.output_length = lambda x: cnn_output_length(x, kernel_size, 
                                                      conv_border_mode, 
                                                      conv_stride)
    print(model.summary())
    return model

#####################################################################################################################
#Trying different models
#https://www.kaggle.com/CVxTz/keras-cnn-starter
from keras import optimizers, losses, activations, models
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, GlobalAveragePooling1D, \
    concatenate
    
def dense_cnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode,units,output_dim=29, num_hiddens=1824):
    #nclass = len(list_labels)
    inp = Input(name='the_input', shape=(None, input_dim))
    img_1 = Conv1D(filters, kernel_size,strides=conv_stride,padding=conv_border_mode,activation='relu',name='conv1d')(inp)
    img_1 = MaxPooling1D(pool_size=kernel_size, strides=1, padding='same', name='maxpool_1d')(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    #img_1 = Conv1D(128, kernel_size,strides=conv_stride,padding=conv_border_mode,activation='relu',name='conv1d1')(img_1)
    dense_1 = Dense(64, activation=activations.relu)(img_1)
    # Add a recurrent layer
    bidir_rnn = Bidirectional(GRU(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn'))(dense_1)
    # TODO: Add batch normalization 
    bn_cnn = BatchNormalization(name ='bn_rnn')(bidir_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_cnn)
    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    model = Model(inputs=inp, outputs=y_pred)
    
    model.output_length = lambda x: cnn_output_length(x, kernel_size, 
                                                      conv_border_mode, 
                                                      conv_stride)
    print(model.summary())
    return model

###########################################################################################################################
#Reference : https://github.com/igormq/asr-study/blob/master/core/models.py
def LSTM_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_LSTM = LSTM(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # TODO: Add batch normalization 
    bn_rnn = BatchNormalization(name ='bn_rnn')(simp_LSTM)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim),name ='time_dense')(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def eyben(input_dim, num_hiddens=[78, 120, 27], num_classes=29):
    """ Implementation of Eybens' model
    Reference:
        [1] Eyben, Florian, et al. "From speech to letters-using a novel neural
        network architecture for grapheme based asr." Automatic Speech
        Recognition & Understanding, 2009. ASRU 2009. IEEE Workshop on. IEEE,
        2009.
    """

    assert len(num_hiddens) == 3

    input_data = Input(name='the_input', shape=(None, input_dim))
    o = input_data

    if num_hiddens[0]:
        o = TimeDistributed(Dense(num_hiddens[0]))(o)
    if num_hiddens[1]:
        o = Bidirectional(LSTM(120, return_sequences=True, implementation=2))(o)
    if num_hiddens[2]:
        o = Bidirectional(LSTM(27, return_sequences=True, implementation=2))(o)

    o = TimeDistributed(Dense(num_classes))(o)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(o)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

from keras.activations import relu
def maas(input_dim, num_classes=29, num_hiddens=1824, dropout=0.1,
         max_value=20):
    """ Maas' model.
    Reference:
        [1] Maas, Andrew L., et al. "Lexicon-Free Conversational Speech
        Recognition with Neural Networks." HLT-NAACL. 2015.
    """
    
    input_data = Input(name='the_input', shape=(None, input_dim))
    o = input_data

    # First layer
    o = TimeDistributed(Dense(num_hiddens))(o)
    o = Activation(lambda x: relu(x, max_value= max_value), name='relu1')(o)
    o = TimeDistributed(Dense(num_hiddens))(o)

    # Second layer
    o = TimeDistributed(Dense(num_hiddens))(o)
    o = Activation(lambda x: relu(x, max_value= max_value), name='relu2')(o)
    o = TimeDistributed(Dense(num_hiddens))(o)

    # Third layer
    o = Bidirectional(SimpleRNN(num_hiddens, return_sequences=True,
                                activation= 'relu', kernel_initializer="he_normal", dropout=0.1))(o)

    # Fourth layer
    o = TimeDistributed(Dense(num_hiddens))(o)
    o = Activation(lambda x: relu(x, max_value= max_value), name='relu3')(o)
    o = TimeDistributed(Dense(num_hiddens))(o)
    

    # Fifth layer
    o = TimeDistributed(Dense(num_hiddens))(o)    
    o = Activation(lambda x: relu(x, max_value= max_value),  name='relu4')(o)
    o = TimeDistributed(Dense(num_hiddens))(o)

    # Output layer
    o = TimeDistributed(Dense(num_classes))(o)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(o)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

from keras.layers import GaussianNoise
def graves(input_dim, num_hiddens=100, num_classes=29, std=.6):
    """ Implementation of Graves' model
    Reference:
        [1] Graves, Alex, et al. "Connectionist temporal classification:
        labelling unsegmented sequence data with recurrent neural networks."
        Proceedings of the 23rd international conference on Machine learning.
        ACM, 2006.
    """

    input_data = Input(name='the_input', shape=(None, input_dim))
    inp = input_data

    gn = GaussianNoise(std)(inp)
    bi_rnn = Bidirectional(LSTM(100, return_sequences=True, implementation=2))(gn)
    time_dense = TimeDistributed(Dense(num_classes))(bi_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model
