#Custom Layers
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
from keras.applications.imagenet_utils import decode_predictions
from tensorflow.python.keras.layers import Layer, InputSpec
import tensorflow as tf
import tensorflow.keras as keras
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +    'haarcascade_frontalface_default.xml')
from mtcnn.mtcnn import MTCNN
from ELBP_Functions     import *
def tf_loader(path,radi):
    img_list=tf.unstack(path)
    hists=[]
    for img_org in img_list:
     
        img_org=img_org*255
        [Rows,Cols]=[img_org.shape[0],img_org.shape[1]]
        if radi==53:
            yTf  = tf_lbp_53_(img_org.reshape(1,Rows,Cols).astype('uint8')).numpy()
        else:
            yTf  = tf_lbp_35_(img_org.reshape(1,Rows,Cols).astype('uint8')).numpy()
        
        hist=uniformer(yTf.reshape(1,yTf.shape[1]*yTf.shape[2]))
        hists.append(tf.convert_to_tensor(hist,dtype='float32'))
    return tf.stack(hists)
    
def uniformer(p):
  h,b=np.histogram(p,bins=256)
  uniformed=np.empty(59, dtype='uint8')
  #Do not try ti optimise this , takes only 0.5 % of time
  unis=[0, 1, 2, 3, 4, 6, 7, 8, 12, 14, 15, 16, 24, 28, 30, 31, 32, 48, 56, 60, 62, 63, 64, 96, 112, 120, 124, 126, 127, 128, 129, 131, 135, 143, 159, 191, 192, 193, 195, 199, 207, 223, 224, 225, 227, 231, 239, 240, 241, 243, 247, 248, 249, 251, 252, 253, 254, 255]
  i=0
  for n in unis: 
    uniformed[i]=h[n]
    h[n]=0
    i=i+1
  uniformed[58]=np.sum(h)  
  return uniformed

def sliceBy16(path,radi):
    #img_org =image.load_img(path,target_size=(299,299,1),color_mode='grayscale')
    #img_org=image.img_to_array(img_org)
    #print('Enter by 16 '+str(path.shape))
    img_list=tf.unstack(path)
    hists=[]
    for img_org in img_list:
        #img_org= tf.squeeze(path, axis=0)
        #print('Squeese '+str(img_org.shape))
        img_org=img_org*255
        [H,w]=[img_org.shape[0],img_org.shape[1]]
        
        roi_00=img_org[           :int(H*0.25),           :int(w*0.25),:]
        roi_10=img_org[           :int(H*0.25),int(w*.25 ):int(w*0.5 ),:]
        roi_20=img_org[           :int(H*0.25),int(w*0.5 ):int(w*0.75),:]
        roi_30=img_org[           :int(H*0.25),int(w*0.75):           ,:]
        roi_01=img_org[int(H*0.25):int(H*0.5 ),           :int(w*0.25),:]
        roi_11=img_org[int(H*0.25):int(H*0.5 ),int(w*.25 ):int(w*0.5 ),:]
        roi_21=img_org[int(H*0.25):int(H*0.5 ),int(w*0.5 ):int(w*0.75),:]
        roi_31=img_org[int(H*0.25):int(H*0.5 ),int(w*0.75):           ,:]
        roi_02=img_org[int(H*0.5 ):int(H*0.75),           :int(w*0.25),:]
        roi_12=img_org[int(H*0.5 ):int(H*0.75),int(w*.25 ):int(w*0.5 ),:]
        roi_22=img_org[int(H*0.5 ):int(H*0.75),int(w*0.5 ):int(w*0.75),:]
        roi_32=img_org[int(H*0.5 ):int(H*0.75),int(w*0.75):           ,:]
        roi_03=img_org[int(H*0.75):      ,           :int(w*0.25)     ,:]
        roi_13=img_org[int(H*0.75):      ,int(w*.25 ):int(w*0.5 )     ,:]
        roi_23=img_org[int(H*0.75):      ,int(w*0.5 ):int(w*0.75)     ,:]
        roi_33=img_org[int(H*0.75):      ,int(w*0.75):                ,:]
    
    
    
        if radi==53:
            yTf_00  = tf_lbp_53_(roi_00.reshape(1,roi_00.shape[0],roi_00.shape[1]).astype('uint8')).numpy()
            yTf_10  = tf_lbp_53_(roi_10.reshape(1,roi_10.shape[0],roi_10.shape[1]).astype('uint8')).numpy()
            yTf_20  = tf_lbp_53_(roi_20.reshape(1,roi_20.shape[0],roi_20.shape[1]).astype('uint8')).numpy()
            yTf_30  = tf_lbp_53_(roi_30.reshape(1,roi_30.shape[0],roi_30.shape[1]).astype('uint8')).numpy()
            yTf_01  = tf_lbp_53_(roi_01.reshape(1,roi_01.shape[0],roi_01.shape[1]).astype('uint8')).numpy()
            yTf_11  = tf_lbp_53_(roi_11.reshape(1,roi_11.shape[0],roi_11.shape[1]).astype('uint8')).numpy()
            yTf_21  = tf_lbp_53_(roi_21.reshape(1,roi_21.shape[0],roi_21.shape[1]).astype('uint8')).numpy()
            yTf_31  = tf_lbp_53_(roi_31.reshape(1,roi_31.shape[0],roi_31.shape[1]).astype('uint8')).numpy()
            yTf_02  = tf_lbp_53_(roi_02.reshape(1,roi_02.shape[0],roi_02.shape[1]).astype('uint8')).numpy()
            yTf_12  = tf_lbp_53_(roi_12.reshape(1,roi_12.shape[0],roi_12.shape[1]).astype('uint8')).numpy()
            yTf_22  = tf_lbp_53_(roi_22.reshape(1,roi_22.shape[0],roi_22.shape[1]).astype('uint8')).numpy()
            yTf_32  = tf_lbp_53_(roi_32.reshape(1,roi_32.shape[0],roi_32.shape[1]).astype('uint8')).numpy()
            yTf_03  = tf_lbp_53_(roi_03.reshape(1,roi_03.shape[0],roi_03.shape[1]).astype('uint8')).numpy()
            yTf_13  = tf_lbp_53_(roi_13.reshape(1,roi_13.shape[0],roi_13.shape[1]).astype('uint8')).numpy()
            yTf_23  = tf_lbp_53_(roi_23.reshape(1,roi_23.shape[0],roi_23.shape[1]).astype('uint8')).numpy()
            yTf_33  = tf_lbp_53_(roi_33.reshape(1,roi_33.shape[0],roi_33.shape[1]).astype('uint8')).numpy()
        else:
            yTf_00  = tf_lbp_35_(roi_00.reshape(1,roi_00.shape[0],roi_00.shape[1]).astype('uint8')).numpy()
            yTf_10  = tf_lbp_35_(roi_10.reshape(1,roi_10.shape[0],roi_10.shape[1]).astype('uint8')).numpy()
            yTf_20  = tf_lbp_35_(roi_20.reshape(1,roi_20.shape[0],roi_20.shape[1]).astype('uint8')).numpy()
            yTf_30  = tf_lbp_35_(roi_30.reshape(1,roi_30.shape[0],roi_30.shape[1]).astype('uint8')).numpy()
            yTf_01  = tf_lbp_35_(roi_01.reshape(1,roi_01.shape[0],roi_01.shape[1]).astype('uint8')).numpy()
            yTf_11  = tf_lbp_35_(roi_11.reshape(1,roi_11.shape[0],roi_11.shape[1]).astype('uint8')).numpy()
            yTf_21  = tf_lbp_35_(roi_21.reshape(1,roi_21.shape[0],roi_21.shape[1]).astype('uint8')).numpy()
            yTf_31  = tf_lbp_35_(roi_31.reshape(1,roi_31.shape[0],roi_31.shape[1]).astype('uint8')).numpy()
            yTf_02  = tf_lbp_35_(roi_02.reshape(1,roi_02.shape[0],roi_02.shape[1]).astype('uint8')).numpy()
            yTf_12  = tf_lbp_35_(roi_12.reshape(1,roi_12.shape[0],roi_12.shape[1]).astype('uint8')).numpy()
            yTf_22  = tf_lbp_35_(roi_22.reshape(1,roi_22.shape[0],roi_22.shape[1]).astype('uint8')).numpy()
            yTf_32  = tf_lbp_35_(roi_32.reshape(1,roi_32.shape[0],roi_32.shape[1]).astype('uint8')).numpy()
            yTf_03  = tf_lbp_35_(roi_03.reshape(1,roi_03.shape[0],roi_03.shape[1]).astype('uint8')).numpy()
            yTf_13  = tf_lbp_35_(roi_13.reshape(1,roi_13.shape[0],roi_13.shape[1]).astype('uint8')).numpy()
            yTf_23  = tf_lbp_35_(roi_23.reshape(1,roi_23.shape[0],roi_23.shape[1]).astype('uint8')).numpy()
            yTf_33  = tf_lbp_35_(roi_33.reshape(1,roi_33.shape[0],roi_33.shape[1]).astype('uint8')).numpy()
        
        hs=[]
        hist=uniformer(yTf_00.reshape(1,yTf_00.shape[1]*yTf_00.shape[2]));   hs=np.append(hs,hist)
        hist=uniformer(yTf_10.reshape(1,yTf_10.shape[1]*yTf_10.shape[2]));   hs=np.append(hs,hist)
        hist=uniformer(yTf_20.reshape(1,yTf_20.shape[1]*yTf_20.shape[2]));   hs=np.append(hs,hist)
        hist=uniformer(yTf_30.reshape(1,yTf_30.shape[1]*yTf_30.shape[2]));   hs=np.append(hs,hist)
        hist=uniformer(yTf_01.reshape(1,yTf_01.shape[1]*yTf_01.shape[2]));   hs=np.append(hs,hist)
        hist=uniformer(yTf_11.reshape(1,yTf_11.shape[1]*yTf_11.shape[2]));   hs=np.append(hs,hist)
        hist=uniformer(yTf_21.reshape(1,yTf_21.shape[1]*yTf_21.shape[2]));   hs=np.append(hs,hist)
        hist=uniformer(yTf_31.reshape(1,yTf_31.shape[1]*yTf_31.shape[2]));   hs=np.append(hs,hist)
        hist=uniformer(yTf_02.reshape(1,yTf_02.shape[1]*yTf_02.shape[2]));   hs=np.append(hs,hist)
        hist=uniformer(yTf_12.reshape(1,yTf_12.shape[1]*yTf_12.shape[2]));   hs=np.append(hs,hist)
        hist=uniformer(yTf_22.reshape(1,yTf_22.shape[1]*yTf_22.shape[2]));   hs=np.append(hs,hist)
        hist=uniformer(yTf_32.reshape(1,yTf_32.shape[1]*yTf_32.shape[2]));   hs=np.append(hs,hist)
        hist=uniformer(yTf_03.reshape(1,yTf_03.shape[1]*yTf_03.shape[2]));   hs=np.append(hs,hist)
        hist=uniformer(yTf_13.reshape(1,yTf_13.shape[1]*yTf_13.shape[2]));   hs=np.append(hs,hist)
        hist=uniformer(yTf_23.reshape(1,yTf_23.shape[1]*yTf_23.shape[2]));   hs=np.append(hs,hist)
        hist=uniformer(yTf_33.reshape(1,yTf_33.shape[1]*yTf_33.shape[2]));   hs=np.append(hs,hist)
        hists.append(tf.convert_to_tensor(hs,dtype='float32'))
        #print('hists');print(hists)
        
    batched=tf.stack(hists)
#    print("16 "+str(batched.shape))

    return batched



def sliceBy4(path,radi):
    img_list=tf.unstack(path)
    hists=[]
    for img_org in img_list:
        img_org=img_org*255 
        [H,w]=[img_org.shape[0],img_org.shape[1]]
        
        roi_00=img_org[:int(H*0.5),:int(w*0.5),:]
        roi_01=img_org[int(H*0.5):,:int(w*0.5),:]
        roi_10=img_org[:int(H*0.5),int(w*0.5):,:]
        roi_11=img_org[int(H*0.5):,int(w*0.5):,:]
        if radi==53:
            yTf_00  = tf_lbp_53_(roi_00.reshape(1,roi_00.shape[0],roi_00.shape[1]).astype('uint8')).numpy()
            yTf_01  = tf_lbp_53_(roi_01.reshape(1,roi_01.shape[0],roi_01.shape[1]).astype('uint8')).numpy()
            yTf_10  = tf_lbp_53_(roi_10.reshape(1,roi_10.shape[0],roi_10.shape[1]).astype('uint8')).numpy()
            yTf_11  = tf_lbp_53_(roi_11.reshape(1,roi_11.shape[0],roi_11.shape[1]).astype('uint8')).numpy()
        else:
            yTf_00  = tf_lbp_35_(roi_00.reshape(1,roi_00.shape[0],roi_00.shape[1]).astype('uint8')).numpy()
            yTf_01  = tf_lbp_35_(roi_01.reshape(1,roi_01.shape[0],roi_01.shape[1]).astype('uint8')).numpy()
            yTf_10  = tf_lbp_35_(roi_10.reshape(1,roi_10.shape[0],roi_10.shape[1]).astype('uint8')).numpy()
            yTf_11  = tf_lbp_35_(roi_11.reshape(1,roi_11.shape[0],roi_11.shape[1]).astype('uint8')).numpy()
        hs=[]
        hist=uniformer(yTf_00.reshape(1,yTf_00.shape[1]*yTf_00.shape[2])) ;   hs=np.append(hs,hist)   
        hist=uniformer(yTf_01.reshape(1,yTf_01.shape[1]*yTf_01.shape[2])) ;   hs=np.append(hs,hist)   
        hist=uniformer(yTf_10.reshape(1,yTf_10.shape[1]*yTf_10.shape[2])) ;   hs=np.append(hs,hist)   
        hist=uniformer(yTf_11.reshape(1,yTf_11.shape[1]*yTf_11.shape[2])) ;   hs=np.append(hs,hist)
        hists.append(tf.convert_to_tensor(hs,dtype='float32'))

    return tf.stack(hists)    
        

# elbp for 5,3 --- general function for elbp--- calling function

def tf_lbp_53_(img):    
    
    paddings = tf.constant([[0,0],[3, 3], [5, 5]])
   # print('original shape '+str(img.shape))
    img=tf.pad(img, paddings,"CONSTANT")        
    b=img.shape 
    #print('padded shape '+str(img.shape))
    
  
    Y=b[1]
    X=b[2]
    #print('Y= '+str(Y)+'  X ='+str(X))
    img_padded=img
    #select the pixels of masks in the form of matrices
  
    i00=img_padded[:,0:Y-6, 2:X-8 ]  #T-left
    i01=img_padded[:,0:Y-6, 5:X-5 ]  #T-mid
    i02=img_padded[:,0:Y-6, 8:X-2 ]  #T-right     
    i10=img_padded[:,3:Y-3, 0:X-10]  #left
    i11=img_padded[:,3:Y-3, 5:X-5 ]  # center /threshold
    i12=img_padded[:,3:Y-3, 10:   ]  # right
    i20=img_padded[:,6:   , 2:X-8 ]  #R-left
    i21=img_padded[:,6:   , 5:X-5 ]  #R-mid
    i22=img_padded[:,6:   , 8:X-2 ]  #R-right
    g=tf.greater_equal(i01,i11)
    z=   tf.multiply(tf.cast(g,dtype='uint8'),tf.constant(1,dtype='uint8') )      
    # 2 ---------------------------------
    g   =tf.greater_equal(i02,i11)
    tmp = tf.multiply(tf.cast(g,dtype='uint8'),tf.constant(2,dtype='uint8') )
    z   =tf.add(z,tmp)              
    # 3 ---------------------------------
    g   =tf.greater_equal(i12,i11)
    tmp = tf.multiply(tf.cast(g,dtype='uint8'),tf.constant(4,dtype='uint8') )
    z   =tf.add(z,tmp)
    # 4 ---------------------------------
    g   =tf.greater_equal(i22,i11)
    tmp = tf.multiply(tf.cast(g,dtype='uint8'),tf.constant(8,dtype='uint8') )
    z   =tf.add(z,tmp)  
    # 5 ---------------------------------
    g   =tf.greater_equal(i21,i11)
    tmp = tf.multiply(tf.cast(g,dtype='uint8'),tf.constant(16,dtype='uint8') )
    z   =tf.add(z,tmp)  
    # 6 ---------------------------------
    g   =tf.greater_equal(i20,i11)
    tmp = tf.multiply(tf.cast(g,dtype='uint8'),tf.constant(32,dtype='uint8') )
    z   =tf.add(z,tmp)  
    # 7 ---------------------------------
    g   =tf.greater_equal(i10,i11)
    tmp = tf.multiply(tf.cast(g,dtype='uint8'),tf.constant(64,dtype='uint8') )
    z   =tf.add(z,tmp)  
    # 8 ---------------------------------
    g   =tf.greater_equal(i00,i11)
    tmp = tf.multiply(tf.cast(g,dtype='uint8'), tf.constant(128,dtype='uint8') )
    z   =tf.add(z,tmp)  
        
    return tf.cast(z,dtype=tf.uint8)




# elbp for 3,5 --- general function for elbp--- calling function

def tf_lbp_35_(img):    
    
    paddings = tf.constant([[0,0],[5,5], [3, 3]])
    img=tf.pad(img, paddings,"CONSTANT")        
    b=img.shape 
  
    Y=b[1]
    X=b[2]
    img_padded=img
    #select the pixels of masks in the form of matrices
  
    i00=img_padded[:, 2:Y-8 ,0:X-6] #T-left
    i01=img_padded[:, 5:Y-5 ,0:X-6] #T-mid
    i02=img_padded[:, 8:Y-2 ,0:X-6] #T-right     
    i10=img_padded[:, 0:Y-10,3:X-3] #left
    i11=img_padded[:, 5:Y-5 ,3:X-3] # center /threshold
    i12=img_padded[:,10:    ,3:X-3] # right
    i20=img_padded[:, 2:Y-8 ,6:   ] #R-left
    i21=img_padded[:, 5:Y-5 ,6:   ] #R-mid
    i22=img_padded[:, 8:Y-2 ,6:   ] #R-right
    g=tf.greater_equal(i01,i11)
    z=   tf.multiply(tf.cast(g,dtype='uint8'),tf.constant(1,dtype='uint8') )      
    # 2 ---------------------------------
    g   =tf.greater_equal(i02,i11)
    tmp = tf.multiply(tf.cast(g,dtype='uint8'),tf.constant(2,dtype='uint8') )
    z   =tf.add(z,tmp)              
    # 3 ---------------------------------
    g   =tf.greater_equal(i12,i11)
    tmp = tf.multiply(tf.cast(g,dtype='uint8'),tf.constant(4,dtype='uint8') )
    z   =tf.add(z,tmp)
    # 4 ---------------------------------
    g   =tf.greater_equal(i22,i11)
    tmp = tf.multiply(tf.cast(g,dtype='uint8'),tf.constant(8,dtype='uint8') )
    z   =tf.add(z,tmp)  
    # 5 ---------------------------------
    g   =tf.greater_equal(i21,i11)
    tmp = tf.multiply(tf.cast(g,dtype='uint8'),tf.constant(16,dtype='uint8') )
    z   =tf.add(z,tmp)  
    # 6 ---------------------------------
    g   =tf.greater_equal(i20,i11)
    tmp = tf.multiply(tf.cast(g,dtype='uint8'),tf.constant(32,dtype='uint8') )
    z   =tf.add(z,tmp)  
    # 7 ---------------------------------
    g   =tf.greater_equal(i10,i11)
    tmp = tf.multiply(tf.cast(g,dtype='uint8'),tf.constant(64,dtype='uint8') )
    z   =tf.add(z,tmp)  
    # 8 ---------------------------------
    g   =tf.greater_equal(i00,i11)
    tmp = tf.multiply(tf.cast(g,dtype='uint8'), tf.constant(128,dtype='uint8') )
    z   =tf.add(z,tmp)  
    #---------------------------------    
    return tf.cast(z,dtype=tf.uint8)
class ComputeLBP_full_53(tf.keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super(ComputeLBP_full_53, self).__init__(name=name)
        super(ComputeLBP_full_53, self).__init__(**kwargs)
        self.LBP_features = tf.Variable(initial_value=tf.zeros(59,dtype='uint8'), trainable=False)



    def get_config(self):
        config = super(ComputeLBP_full_53, self).get_config()
       
        return config

    def call(self, input):
        hist= tf.py_function( tf_loader, 
                           [input,53],
                           'float32')
        
        self.LBP_features=tf.cast(tf.reshape(hist,[-1,59]), tf.float32)
        return self.LBP_features

class ComputeLBP_full_35(tf.keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super(ComputeLBP_full_35, self).__init__(name=name)
        super(ComputeLBP_full_35, self).__init__(**kwargs)
        self.LBP_features = tf.Variable(initial_value=tf.zeros(59,dtype='uint8'), trainable=False)



    def get_config(self):
        config = super(ComputeLBP_full_35, self).get_config()
       
        return config

    def call(self, input):
        hist= tf.py_function( tf_loader, 
                           [input,35],
                           'float32')
        
        self.LBP_features=tf.cast(tf.reshape(hist,[-1,59]), tf.float32)
        return self.LBP_features

class ComputeLBP_4_53(keras.layers.Layer):
    def __init__(self,name=None, **kwargs):
        super(ComputeLBP_4_53, self).__init__(name=name)
        super(ComputeLBP_4_53, self).__init__(**kwargs)
        self.LBP_features = tf.Variable(initial_value=tf.zeros(236,dtype='uint8'), trainable=False)

    def call(self, input):
        
        hs= tf.py_function( sliceBy4, 
                           [input,53],
                           'float32')   
        
        self.LBP_features=tf.cast(tf.reshape(hs,[-1,236]), tf.float32)
        return self.LBP_features
    def get_config(self):

        config = super(ComputeLBP_4_53,self).get_config()
        return config
 

class ComputeLBP_4_35(keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super(ComputeLBP_4_35, self).__init__(name=name)
        super(ComputeLBP_4_35, self).__init__(**kwargs)
        self.LBP_features = tf.Variable(initial_value=tf.zeros(236,dtype='uint8'), trainable=False)
        
    def get_config(self):

        config = super(ComputeLBP_4_35, self).get_config()
        return config
    
    def call(self, input):
        
        hs= tf.py_function( sliceBy4, 
                           [input,35],
                           'float32') 
        
        self.LBP_features=tf.cast(tf.reshape(hs,[-1,236]), tf.float32)

        return self.LBP_features

    
class FacePlate(keras.layers.Layer):
    
    def __init__(self, name=None, **kwargs):
        super(FacePlate, self).__init__(name=None)
        super(FacePlate, self).__init__(**kwargs)
        self.Faces = tf.Variable(initial_value=tf.zeros([299,299,3]), trainable=False)
    def get_config(self):

        config = super(FacePlate, self).get_config()
        return config
    
    def call(self, inputs):
        #print('in Faceplae')
        hs= tf.py_function( getFace,                            [inputs],                           'float32') 
        self.Faces=tf.cast(tf.reshape(hs,[-1,299,299,3]), tf.float32)
        return self.Faces


def getFace(img):
    
    frames=[]
    for frame in img:
        f=getFace1(frame)
        frames.append(f)
    f_s=tf.Variable(initial_value=tf.zeros((299,299,3),dtype= 'float32'), trainable=False)
    i=0;
    return tf.stack(frames)
    

def getFace1(im):
        
        #face_list.append(face_cropped)
        #gray =float_to_Img(im.numpy())
        #gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        # 20% faster alternate
        gray= cv2.cvtColor(((im.numpy())*255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(30, 30)
        ) 
        try:
            if len(faces) == 0:
                    return im[0:299,0:299,:]
                    print('no face')
                    faces = detector.detect_faces(im.numpy())
                    if len(faces) == 0:
                                return im[0:299,0:299,:]
                    x, y, w, h = faces[0]['box']
                    face_cropped=im[y-10:y+h+10,x-10:x+w+10,:]
                    
                    rescaled=cv2.resize((face_cropped.numpy()), [299,299], interpolation=cv2.INTER_AREA)
                    return rescaled
            [x,y,w,h]=faces[0]
            face_cropped=im[y-10:y+h+10,x-10:x+w+10,:]
            rescaled=cv2.resize((face_cropped.numpy()), [299,299], interpolation=cv2.INTER_AREA)
                
            return rescaled
        except :
            return im[0:299,0:299,:]



def float_to_Img(ar):
    
    shape=ar.shape
    img = np.zeros([ar.shape[0],ar.shape[0],3],dtype='uint8')

    img[:,:,0] = ar[:,:,0]*255
    img[:,:,1] = ar[:,:,1]*255
    img[:,:,2] = ar[:,:,2]*255
    #plt.imshow(img)
    return img
        