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
        
        face_list.append(yTf)
        face_titles.append('Full face')
        hist,p= np.histogram(yTf.reshape(1,yTf.shape[1]*yTf.shape[2]),bins=range(0,256,31))
        hists.append(tf.convert_to_tensor(hist,dtype='float32'))
        
    batched=tf.stack(hists)
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
        hist,p= np.histogram(yTf_00.reshape(1,yTf_00.shape[1]*yTf_00.shape[2]),bins=range(0,256,31)) ;   hs=np.append(hs,hist)   
        hist,p= np.histogram(yTf_01.reshape(1,yTf_01.shape[1]*yTf_01.shape[2]),bins=range(0,256,31)) ;   hs=np.append(hs,hist)   
        hist,p= np.histogram(yTf_10.reshape(1,yTf_10.shape[1]*yTf_10.shape[2]),bins=range(0,256,31)) ;   hs=np.append(hs,hist)   
        hist,p= np.histogram(yTf_11.reshape(1,yTf_11.shape[1]*yTf_11.shape[2]),bins=range(0,256,31)) ;   hs=np.append(hs,hist)
        hists.append(tf.convert_to_tensor(hs,dtype='float32'))
        #print('hists');print(hists)
        
    batched=tf.stack(hists)
 #   print("04 "+str(batched.shape))

    return batched    
        

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
  
    i00=img_padded[:,0:Y-6, 2:X-8] #T-left
    i01=img_padded[:,0:Y-6, 5:X-5] #T-mid
    i02=img_padded[:,0:Y-6, 8:X-2] #T-right     
    i10=img_padded[:,3:Y-3, 0:X-10] #left
    i11=img_padded[:,3:Y-3, 5:X-5]  # center /threshold
    i12=img_padded[:,3:Y-3, 10:]    # right
    i20=img_padded[:,6: , 2:X-8] #R-left
    i21=img_padded[:,6: , 5:X-5] #R-mid
    i22=img_padded[:,6: , 8:X-2] #R-right
    g=tf.greater_equal(i01,i11)
    z=   tf.multiply(tf.cast(g,dtype='uint8'),tf.constant(1,dtype='uint8') )      
    # 2 ---------------------------------
    g=tf.greater_equal(i02,i11)
    tmp= tf.multiply(tf.cast(g,dtype='uint8'),tf.constant(2,dtype='uint8') )
    z =tf.add(z,tmp)              
    # 3 ---------------------------------
    g=tf.greater_equal(i12,i11)
    tmp= tf.multiply(tf.cast(g,dtype='uint8'),tf.constant(4,dtype='uint8') )
    z =tf.add(z,tmp)
    # 4 ---------------------------------
    g=tf.greater_equal(i22,i11)
    tmp= tf.multiply(tf.cast(g,dtype='uint8'),tf.constant(8,dtype='uint8') )
    z =tf.add(z,tmp)  
    # 5 ---------------------------------
    g=tf.greater_equal(i21,i11)
    tmp= tf.multiply(tf.cast(g,dtype='uint8'),tf.constant(16,dtype='uint8') )
    z =tf.add(z,tmp)  
    # 6 ---------------------------------
    g=tf.greater_equal(i20,i11)
    tmp= tf.multiply(tf.cast(g,dtype='uint8'),tf.constant(32,dtype='uint8') )
    z =tf.add(z,tmp)  
    # 7 ---------------------------------
    g=tf.greater_equal(i10,i11)
    tmp= tf.multiply(tf.cast(g,dtype='uint8'),tf.constant(64,dtype='uint8') )
    z =tf.add(z,tmp)  
    # 8 ---------------------------------
    g=tf.greater_equal(i00,i11)
    tmp= tf.multiply(tf.cast(g,dtype='uint8'), tf.constant(128,dtype='uint8') )
    z =tf.add(z,tmp)  
    #---------------------------------    
    return tf.cast(z,dtype=tf.uint8)




# elbp for 3,5 --- general function for elbp--- calling function

def tf_lbp_35_(img):    
    
    paddings = tf.constant([[0,0],[5,5], [3, 3]])
    #print('original shape '+str(img.shape))
    img=tf.pad(img, paddings,"CONSTANT")        
    b=img.shape 
    #print('padded shape '+str(img.shape))
    
  
    Y=b[1]
    X=b[2]
   # print('Y= '+str(Y)+'  X ='+str(X))
    img_padded=img
    #select the pixels of masks in the form of matrices
  
    i00=img_padded[:, 2:Y-8 ,0:X-6] #T-left
    i01=img_padded[:, 5:Y-5 ,0:X-6] #T-mid
    i02=img_padded[:, 8:Y-2 ,0:X-6] #T-right     
    i10=img_padded[:, 0:Y-10,3:X-3] #left
    i11=img_padded[:, 5:Y-5 ,3:X-3]  # center /threshold
    i12=img_padded[:,10:    ,3:X-3]    # right
    i20=img_padded[:, 2:Y-8 ,6:   ] #R-left
    i21=img_padded[:, 5:Y-5 ,6:   ] #R-mid
    i22=img_padded[:, 8:Y-2 ,6:   ] #R-right
    g=tf.greater_equal(i01,i11)
    z=   tf.multiply(tf.cast(g,dtype='uint8'),tf.constant(1,dtype='uint8') )      
    # 2 ---------------------------------
    g=tf.greater_equal(i02,i11)
    tmp= tf.multiply(tf.cast(g,dtype='uint8'),tf.constant(2,dtype='uint8') )
    z =tf.add(z,tmp)              
    # 3 ---------------------------------
    g=tf.greater_equal(i12,i11)
    tmp= tf.multiply(tf.cast(g,dtype='uint8'),tf.constant(4,dtype='uint8') )
    z =tf.add(z,tmp)
    # 4 ---------------------------------
    g=tf.greater_equal(i22,i11)
    tmp= tf.multiply(tf.cast(g,dtype='uint8'),tf.constant(8,dtype='uint8') )
    z =tf.add(z,tmp)  
    # 5 ---------------------------------
    g=tf.greater_equal(i21,i11)
    tmp= tf.multiply(tf.cast(g,dtype='uint8'),tf.constant(16,dtype='uint8') )
    z =tf.add(z,tmp)  
    # 6 ---------------------------------
    g=tf.greater_equal(i20,i11)
    tmp= tf.multiply(tf.cast(g,dtype='uint8'),tf.constant(32,dtype='uint8') )
    z =tf.add(z,tmp)  
    # 7 ---------------------------------
    g=tf.greater_equal(i10,i11)
    tmp= tf.multiply(tf.cast(g,dtype='uint8'),tf.constant(64,dtype='uint8') )
    z =tf.add(z,tmp)  
    # 8 ---------------------------------
    g=tf.greater_equal(i00,i11)
    tmp= tf.multiply(tf.cast(g,dtype='uint8'), tf.constant(128,dtype='uint8') )
    z =tf.add(z,tmp)  
    #---------------------------------    
    return tf.cast(z,dtype=tf.uint8)
