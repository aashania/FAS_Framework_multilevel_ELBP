from keras.preprocessing import image
import matplotlib.pyplot as plt
import time
from tqdm.notebook import tqdm as log_progress
import pandas as pd
import numpy as np
import time
import random
import os
from matplotlib import pyplot as plt
import cv2
import tensorflow as tf

import warnings
import numpy as np

from keras.preprocessing import image
from keras.utils import data_utils
from keras.models import Model
from keras import layers
from keras.models import Sequential
from keras.layers import *
import os
import pandas  as pd
from keras.utils.vis_utils import plot_model
def model_photu(model,n=True,expand=False,direct='TB',shapes=True,name='model_plot'):
   return plot_model(model, to_file=os.getcwd()+'/model_images/'+name+'.png', show_shapes=shapes, show_layer_names=n,expand_nested=expand,rankdir=direct)
#from keras.engine.topology import get_source_inputs
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
#from keras.applications.imagenet_utils import _obtain_input_shape
from tensorflow.python.keras.layers import Layer, InputSpec
#from keras.applications.imagenet_utils import _obtain_input_shape
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +    'haarcascade_frontalface_default.xml')
from mtcnn.mtcnn import MTCNN
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'

def SqueezeAndExcitation(z,name1):
    b,h,w,c = z.shape
    ratio=16
    #squeeze
    y = keras.layers.GlobalAveragePooling2D(name=name1+'glob')(z) 
    #Excitation operation
    y= Dense(c//ratio, activation='relu',name=name1+'relu', use_bias= False)(y)
    y = keras.layers.Dense(c, activation='sigmoid', use_bias=False,name=name1+'sig')(y)
    y = keras.layers.multiply([z,y],name=name1)
    #y.name=name1
    return y  

def SE_Xception(include_top=False, weights='imagenet', input_shape=(299,299,3),input_tensor=None):
    
    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
       if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
       else:
          img_input = input_tensor
    x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, name='block1_conv1')(img_input)
    x = BatchNormalization(name='block1_conv1_bn')(x)
    x = Activation('relu', name='block1_conv1_act')(x)
    x = Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
    x = BatchNormalization(name='block1_conv2_bn')(x)
    x = Activation('relu', name='block1_conv2_act')(x)

    residual = Conv2D(128, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv1')(x)
    x = BatchNormalization(name='block2_sepconv1_bn')(x)
    x = Activation('relu', name='block2_sepconv2_act')(x)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv2')(x)
    x = BatchNormalization(name='block2_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block2_pool')(x)
    x = layers.add([x, residual])

    residual = Conv2D(256, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block3_sepconv1_act')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv1')(x)
    x = BatchNormalization(name='block3_sepconv1_bn')(x)
    x = Activation('relu', name='block3_sepconv2_act')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv2')(x)
    x = BatchNormalization(name='block3_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block3_pool')(x)
    x = layers.add([x, residual])

    residual = Conv2D(728, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block4_sepconv1_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv1')(x)
    x = BatchNormalization(name='block4_sepconv1_bn')(x)
    x = Activation('relu', name='block4_sepconv2_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv2')(x)
    x = BatchNormalization(name='block4_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block4_pool')(x)
    x = layers.add([x, residual])
    # entry flow features to squeeze excitation block
    s=x
    # entry flow complete
    # middle flow starts

    for i in range(8):
        residual = x
        prefix = 'block' + str(i + 5)

        x = Activation('relu', name=prefix + '_sepconv1_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv1')(x)
        x = BatchNormalization(name=prefix + '_sepconv1_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv2_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv2')(x)
        x = BatchNormalization(name=prefix + '_sepconv2_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv3_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv3')(x)
        x = BatchNormalization(name=prefix + '_sepconv3_bn')(x)
        
       
        x = layers.add([x, residual])

    f=x
    # middle flow endend---- add squeeze excitation block
        # exit flow starting
    residual = Conv2D(1024, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block13_sepconv1_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block13_sepconv1')(x)
    x = BatchNormalization(name='block13_sepconv1_bn')(x)
    x = Activation('relu', name='block13_sepconv2_act')(x)
    x = SeparableConv2D(1024, (3, 3), padding='same', use_bias=False, name='block13_sepconv2')(x)
    x = BatchNormalization(name='block13_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block13_pool')(x)
    x = layers.add([x, residual])

    x = SeparableConv2D(1536, (3, 3), padding='same', use_bias=False, name='block14_sepconv1')(x)
    x = BatchNormalization(name='block14_sepconv1_bn')(x)
    x = Activation('relu', name='block14_sepconv1_act')(x)

    x = SeparableConv2D(2048, (3, 3), padding='same', use_bias=False, name='block14_sepconv2')(x)
    x = BatchNormalization(name='block14_sepconv2_bn')(x)
    x = Activation('relu', name='block14_sepconv2_act')(x)
    

    if include_top:
          x = GlobalAveragePooling2D(name='avg_pool')(x)
          outputs = Dense(2, activation='softmax', name='predictions')(x)
    else:
         y=SqueezeAndExcitation(s,'Se1') #squeeze excitation to entry block
         z=SqueezeAndExcitation(f,'Se2') #squeeze excitation to middle flow
         q=SqueezeAndExcitation(x,'Se3')
         a=keras.layers.Flatten()(q)
        # print('y=',y)
         #print('z=',z)
        # print('q=',q)
    #concating features output from squeeze excitation block
         #d=SqueezeAndExcitation(layers.add([s,f]))
         d=keras.layers.concatenate([y,z])
        # print("d=",d)
         e= keras.layers.Flatten()(d)
         g=keras.layers.concatenate([a,e])
         #print("g=",g)
         p=keras.layers.Dense(512, activation='relu')(g)
         p=keras.layers.Dense(64,activation='relu')(p)
         outputs = keras.layers.Dense(2,activation='softmax')(p)

           #applying squeeze excitation at the ending block of exit
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    
    # Create model.
    model = Model(inputs, outputs, name='xception')
    # Load weights
    if weights == 'imagenet':
           weights_path =  get_file('xception_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    TF_WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models')

    model.load_weights(weights_path,by_name = True, skip_mismatch = True)
    # load weights
    return model	