#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 11:01:30 2020

@author: adrian
"""


from __future__ import print_function

import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, concatenate
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
import SimpleITK as sitk
import numpy as np
import csv
from glob import glob
import pandas as pd
from skimage import morphology
from skimage import measure
from sklearn.cluster import KMeans
from skimage.transform import resize
import cv2

working_path = "/media/adrian/963EAF133EAEEC05/Users/Adrián/Documents/Projects/Tensorflow/LUNA_Taurons_git/DSB3Tutorial-master/tutorial_allsubsets/"

#K.set_image_dim_ordering('th')  # Theano dimension ordering in this code
#K.common.image_dim_ordering()
#K.image_data_format() == 'channels_first'
K.set_image_data_format('channels_first')
img_rows = 512
img_cols = 512

smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_np(y_true,y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_unet():
    inputs = Input((1,img_rows, img_cols))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)
    
    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=1)

    #up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=1)
    #up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=1)
    #up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=1)
    #up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)
    print (model.summary())

    #model.compile(optimizer=Adam(lr=1.0e-5), loss=dice_coef_loss, metrics=[dice_coef])
    #import pdb;pdb.set_trace()
    model.compile(optimizer=Adam(lr=1e-5), loss='binary_crossentropy', metrics = ['accuracy'])
    
    #import pdb;pdb.set_trace()

    return model


# load the data once
itk_img = sitk.ReadImage('/media/adrian/Data/Descargas/subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.108197895896446896160048741492.mhd') 
imgs_to_process = sitk.GetArrayFromImage(itk_img) # indexes are z,y,x (notice the ordering)
masks = np.zeros_like(imgs_to_process)

for i in range(len(imgs_to_process)):
    img = imgs_to_process[i]
    #Standardize the pixel values
    mean = np.mean(img)
    std = np.std(img)
    img = img-mean
    img = img/std
    # Find the average pixel value near the lungs
    # to renormalize washed out images
    middle = img[100:400,100:400] 
    mean = np.mean(middle)  
    max = np.max(img)
    min = np.min(img)
    # To improve threshold finding, I'm moving the 
    # underflow and overflow on the pixel spectrum
    img[img==max]=mean
    img[img==min]=mean
    #
    # Using Kmeans to separate foreground (radio-opaque tissue)
    # and background (radio transparent tissue ie lungs)
    # Doing this only on the center of the image to avoid 
    # the non-tissue parts of the image as much as possible
    #
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image
    #
    # I found an initial erosion helful for removing graininess from some of the regions
    # and then large dialation is used to make the lung region 
    # engulf the vessels and incursions into the lung cavity by 
    # radio opaque tissue
    #
    eroded = morphology.erosion(thresh_img,np.ones([4,4]))
    dilation = morphology.dilation(eroded,np.ones([10,10]))
    #
    #  Label each region and obtain the region properties
    #  The background region is removed by removing regions 
    #  with a bbox that is to large in either dimnsion
    #  Also, the lungs are generally far away from the top 
    #  and bottom of the image, so any regions that are too
    #  close to the top and bottom are removed
    #  This does not produce a perfect segmentation of the lungs
    #  from the image, but it is surprisingly good considering its
    #  simplicity. 
    #
    labels = measure.label(dilation)
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if B[2]-B[0]<475 and B[3]-B[1]<475 and B[0]>40 and B[2]<472:
            good_labels.append(prop.label)
    mask = np.ndarray([512,512],dtype=np.int8)
    mask[:] = 0
    #
    #  The mask here is the mask for the lungs--not the nodes
    #  After just the lungs are left, we do another large dilation
    #  in order to fill in and out the lung mask 
    #
    for N in good_labels:
        mask = mask + np.where(labels==N,1,0)
    mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation
    cv2.imwrite('/media/adrian/963EAF133EAEEC05/Users/Adrián/Documents/Projects/Tensorflow/LUNA_Taurons_git/DSB3Tutorial-master/tutorial_code/folder_check/'+str(i)+'test_process.jpg', mask*255)
    #import pdb;pdb.set_trace()
    masks[i] = mask
    print(mask.shape)
    
    #node_masks[i]=mask
#np.save(img_file.replace("images","lungmask"),imgs_to_process)


#import pdb;pdb.set_trace()
#file_list=glob(working_path+"lungmask_*.npy")
out_images = []      #final set of images
out_nodemasks = []   #final set of nodemasks
# for fname in file_list:
#     print ("working on file "+fname)
#imgs_to_process = np.load(fname.replace("lungmask","images"))
#masks = np.load(fname)
#node_masks = np.load(fname.replace("lungmask","masks"))
for i in range(len(imgs_to_process)):
    mask = masks[i]
    #node_mask = node_masks[i]
    img = imgs_to_process[i]
    new_size = [512,512]   # we're scaling back up to the original size of the image
    try:
        img= mask*img          # apply lung mask
    except:
        import pdb;pdb.set_trace()
    #
    # renormalizing the masked image (in the mask region)
    #
    new_mean = np.mean(img[mask>0])  
    new_std = np.std(img[mask>0])
    #
    #  Pulling the background color up to the lower end
    #  of the pixel range for the lungs
    #
    old_min = np.min(img)       # background color
    img[img==old_min] = new_mean-1.2*new_std   # resetting backgound color
    img = img-new_mean
    img = img/new_std
    #make image bounding box  (min row, min col, max row, max col)
    labels = measure.label(mask)
    regions = measure.regionprops(labels)
    #
    # Finding the global min and max row over all regions
    #
    min_row = 512
    max_row = 0
    min_col = 512
    max_col = 0
    for prop in regions:
        B = prop.bbox
        if min_row > B[0]:
            min_row = B[0]
        if min_col > B[1]:
            min_col = B[1]
        if max_row < B[2]:
            max_row = B[2]
        if max_col < B[3]:
            max_col = B[3]
    width = max_col-min_col
    height = max_row - min_row
    if width > height:
        max_row=min_row+width
    else:
        max_col = min_col+height
    # 
    # cropping the image down to the bounding box for all regions
    # (there's probably an skimage command that can do this in one line)
    # 
    img = img[min_row:max_row,min_col:max_col]
    mask =  mask[min_row:max_row,min_col:max_col]
    if max_row-min_row <5 or max_col-min_col<5:  # skipping all images with no god regions
        pass
    else:
        # moving range to -1 to 1 to accomodate the resize function
        #mean = np.mean(img)
        #img = img - mean
        min = np.min(img)
        max = np.max(img)
        #img = img/(max-min)
        img = (img-min) / (max-min)
        new_img = resize(img,[512,512])
        #new_node_mask = resize(node_mask[min_row:max_row,min_col:max_col],[512,512])
        #new_node_mask = (new_node_mask > 0.0).astype(np.float32)
        out_images.append(new_img)
        #out_nodemasks.append(new_node_mask)
        cv2.imwrite('/media/adrian/963EAF133EAEEC05/Users/Adrián/Documents/Projects/Tensorflow/LUNA_Taurons_git/DSB3Tutorial-master/tutorial_code/folder_check/'+str(i)+'img.jpg', new_img*255)
        #import pdb;pdb.set_trace()
        
        
        
num_images = len(out_images)
final_images = np.ndarray([num_images,1,512,512],dtype=np.float32)

for i in range(num_images):
    final_images[i,0] = out_images[i]


model = get_unet()
model.load_weights('/media/adrian/963EAF133EAEEC05/Users/Adrián/Documents/Projects/Tensorflow/LUNA_Taurons_git/DSB3Tutorial-master/tutorial_code/unet_3sets100epochs.hdf5')

num_test = len(final_images)
imgs_mask_test = np.ndarray([num_test,1,512,512],dtype=np.float32)
for i in range(num_test):
    imgs_mask_test[i] = model.predict([final_images[i:i+1]], verbose=0)[0]
    
np.save('/media/adrian/963EAF133EAEEC05/Users/Adrián/Documents/Projects/Tensorflow/LUNA_Taurons_git/DSB3Tutorial-master/tutorial_code/folder_check/img.npy',imgs_mask_test)
    
for t in range(num_test):
    cv2.imwrite('/media/adrian/963EAF133EAEEC05/Users/Adrián/Documents/Projects/Tensorflow/LUNA_Taurons_git/DSB3Tutorial-master/tutorial_code/folder_check/'+str(t)+'pred.jpg', imgs_mask_test[t,0,:,:]*255)

segmented_lung = imgs_mask_test[:,0,:,:]


result = np.where(segmented_lung == np.amax(segmented_lung))
# binary = binary_closing(segmented_lung, selem)

# label_scan = label(binary)

# areas = [r.area for r in regionprops(label_scan)]
# areas.sort()


