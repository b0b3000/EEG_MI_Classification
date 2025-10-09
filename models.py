import sys
import numpy
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D, GlobalAveragePooling2D, LeakyReLU
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers 

"""
Some of the below Keras implementations of existing EEG-MI classification models have been inspired/adopted from the GitHub repo:
"ARL_EEGModels - A collection of Convolutional Neural Network models for EEG Signal Processing and Classification, using Keras and Tensorflow.
This repo can be accessed at https://github.com/vlawhern/arl-eegmodels/

This code is accessed fairly under the terms of the Creative Commons Zero 1.0 license, and the Apache 2.0 license.
Full details on the license of the code can be found at https://github.com/vlawhern/arl-eegmodels/blob/master/LICENSE.txt
"""

#added dense head for the proposed model
def dense_head(x, rate):
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, name="Dense2")(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    x = Dropout(rate)(x)
    return x

#The proposed model. 
# This is heavily based on EEGNet. The full explanation of this can be found in the Honours Paper:
# 'An EEGNet based approach to Motor Imagery classification in BCI systems'

def EEGNet_Modified(nb_classes, Chans = 64, Samples = 128, 
             dropoutRate = 0.5, kernLength = 64, F1 = 8, 
             D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'SpatialDropout2D', l2_penalty=0.1):
    
    if l2_penalty:
        l2=regularizers.l2(l2_penalty)
    else:
        l2=None

    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    
    input1   = Input(shape = (Chans, Samples, 1))

    block1       = Conv2D(F1, (1, kernLength), padding = 'same', kernel_regularizer=l2, kernel_initializer='he_normal', input_shape = (Chans, Samples, 1), use_bias = False)(input1)
    block1       = BatchNormalization()(block1)
    block1       = DepthwiseConv2D((Chans, 1), kernel_regularizer=l2, kernel_initializer='he_normal', use_bias = False, depth_multiplier = D, depthwise_constraint = max_norm(1.))(block1)
    block1       = BatchNormalization()(block1)
    block1       = Activation('elu')(block1)
    block1       = AveragePooling2D((1, 4))(block1)
    block1       = dropoutType(dropoutRate)(block1)
    
    block2       = SeparableConv2D(F2, (1, 16), kernel_regularizer=l2,kernel_initializer='he_normal', use_bias = False, padding = 'same')(block1)
    block2       = BatchNormalization()(block2)
    block2       = Activation('elu')(block2)
    block2       = AveragePooling2D((1, 8))(block2)
    block2       = dropoutType(dropoutRate)(block2)
    
    block3 = dense_head(block2, dropoutRate)
        
    flatten      = Flatten(name = 'flatten')(block3)
    
    dense        = Dense(nb_classes,  kernel_initializer='he_normal', name = 'dense', 
                         kernel_constraint = max_norm(norm_rate))(flatten)
    softmax      = Activation('softmax', name = 'softmax')(dense)
    
    return Model(inputs=input1, outputs=softmax)

# The EEGNet architecure. 
# Implemented as specified in: 'EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces' [1]
def EEGNet(nb_classes, Chans = 64, Samples = 128, dropoutRate = 0.5, kernLength = 64, F1 = 8, D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'SpatialDropout2D'):
    
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D or Dropout, passed as a string.')
    
    input1   = Input(shape = (Chans, Samples, 1))

    ##################################################################

    block1       = Conv2D(F1, (1, kernLength), padding = 'same',
                                   input_shape = (Chans, Samples, 1),
                                   use_bias = False)(input1)
    block1       = BatchNormalization()(block1)
    block1       = DepthwiseConv2D((Chans, 1), use_bias = False, 
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.))(block1)
    block1       = BatchNormalization()(block1)
    block1       = Activation('elu')(block1)
    block1       = AveragePooling2D((1, 4))(block1)
    block1       = dropoutType(dropoutRate)(block1)

    ##################################################################

    block2       = SeparableConv2D(F2, (1, 16),
                                   use_bias = False, padding = 'same')(block1)
    block2       = BatchNormalization()(block2)
    block2       = Activation('elu')(block2)
    block2       = AveragePooling2D((1, 8))(block2)
    block2       = dropoutType(dropoutRate)(block2)

    ##################################################################
        
    flatten      = Flatten(name = 'flatten')(block2)
    dense        = Dense(nb_classes, name = 'dense', kernel_constraint = max_norm(norm_rate))(flatten)
    outputs      = Activation('softmax', name = 'softmax')(dense)
    
    return Model(inputs=input1, outputs=outputs)

def DeepConvNet(nb_classes, Chans = 64, Samples = 256,
                dropoutRate = 0.5):
    # Implementation of the 'Deep ConvNet' atchitecture as described in [2].
    
    input_main   = Input((Chans, Samples, 1))
    block1       = Conv2D(25, (1, 10), 
                                 input_shape=(Chans, Samples, 1),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(input_main)
    block1       = Conv2D(25, (Chans, 1),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block1       = BatchNormalization(epsilon=1e-05, momentum=0.9)(block1)
    block1       = Activation('elu')(block1)
    block1       = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block1)
    block1       = Dropout(dropoutRate)(block1)
  
    block2       = Conv2D(50, (1, 10),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block2       = BatchNormalization(epsilon=1e-05, momentum=0.9)(block2)
    block2       = Activation('elu')(block2)
    block2       = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block2)
    block2       = Dropout(dropoutRate)(block2)
    
    block3       = Conv2D(100, (1, 10),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block2)
    block3       = BatchNormalization(epsilon=1e-05, momentum=0.9)(block3)
    block3       = Activation('elu')(block3)
    block3       = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block3)
    block3       = Dropout(dropoutRate)(block3)
    
    block4       = Conv2D(200, (1,10),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block3)
    block4       = BatchNormalization(epsilon=1e-05, momentum=0.9)(block4)
    block4       = Activation('elu')(block4)
    block4       = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block4)
    block4       = Dropout(dropoutRate)(block4)
    
    flatten      = Flatten()(block4)
    
    dense        = Dense(nb_classes, kernel_constraint = max_norm(0.5))(flatten)
    softmax      = Activation('softmax')(dense)
    
    return Model(inputs=input_main, outputs=softmax)

#helper funcs for shallow convnet
def square(x):
    return K.square(x)

def log(x):
    return K.log(K.clip(x, min_value = 1e-7, max_value = 10000))   

def ShallowConvNet(nb_classes, Chans = 64, Samples = 128, dropoutRate = 0.5):
    # Implementation of the 'Shallow ConvNet' as described in [2].

    input_main   = Input((Chans, Samples, 1))
    block1       = Conv2D(40, (1, ), input_shape=(Chans, Samples, 1), kernel_constraint = max_norm(2., axis=(0,1,2)))(input_main)
    block1       = Conv2D(40, (Chans, 1), use_bias=False, kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block1       = BatchNormalization(epsilon=1e-05, momentum=0.9)(block1)
    block1       = Activation(square)(block1)
    block1       = AveragePooling2D(pool_size=(1, 35), strides=(1, 7))(block1)
    block1       = Activation(log)(block1)
    block1       = Dropout(dropoutRate)(block1)
    flatten      = Flatten()(block1)
    dense        = Dense(nb_classes, kernel_constraint = max_norm(0.5))(flatten)
    softmax      = Activation('softmax')(dense)
    
    return Model(inputs=input_main, outputs=softmax)

# [1] Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., and Lance, B. J. 
# Eegnet: a compact convolutional neural network for eeg-based brain–computer interfaces. 
# Journal of neural engineering 15, 5 (2018), 056013.

# [2] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J., Glasstetter, M., Eggensperger, 
# K., Tangermann, M., Hutter, F., Burgard, W., and Ball, T. Deep learning with convolutional neural 
# networks for eeg decoding and visualization. Human brain mapping 38, 11 (2017), 5391–5420.