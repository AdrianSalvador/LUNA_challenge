#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 11:38:42 2020

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

working_path = "/media/adrian/963EAF133EAEEC05/Users/Adrián/Documents/Projects/Tensorflow/LUNA_Taurons_git/DSB3Tutorial-master/tutorial_allsubsets/"

#K.set_image_dim_ordering('th')  # Theano dimension ordering in this code
#K.common.image_dim_ordering()
#K.image_data_format() == 'channels_first'
K.set_image_data_format('channels_first')
img_rows = 512
img_cols = 512

smooth = 1.

#line changed
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



# def get_model():

#     inputs = Input((1, img_rows, img_cols))

#     conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
#     conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
#     pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

#     conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
#     conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
#     pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

#     conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
#     conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
#     pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

#     conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
#     conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
#     pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

#     conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
#     conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

#     up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=1)
#     conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
#     conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

#     up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=1)
#     conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
#     conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

#     up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=1)
#     conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
#     conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

#     up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=1)
#     conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
#     conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
#     conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

#     model = Model(inputs=inputs, outputs=conv10)
#     print (model.summary())
# #     model.compile(optimizer=Adam(lr=1.0e-5), loss=dice_coef_loss, metrics=[dice_coef])
#     model.compile(optimizer=Adam(lr=1e-5), loss='binary_crossentropy', metrics = ['accuracy'])
#     return model




def train_and_predict(use_existing):
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train = np.load(working_path+"trainImages.npy").astype(np.float32)
    imgs_mask_train = np.load(working_path+"trainMasks.npy").astype(np.float32)
    imgs_mask_train_true = np.load(working_path+"trainMasks.npy").astype(np.float32)

    imgs_test = np.load(working_path+"testImages.npy").astype(np.float32)
    imgs_mask_test_true = np.load(working_path+"testMasks.npy").astype(np.float32)
    
    #mean = np.mean(imgs_train)  # mean for data centering
    #std = np.std(imgs_train)  # std for data normalization

    #imgs_train -= mean  # images should already be standardized, but just in case
    #imgs_train /= std

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()
    # Saving weights to unet.hdf5 at checkpoints
    #model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', save_best_only=True)
    
    model_checkpoint = ModelCheckpoint('unet_3sets100epochs.hdf5', monitor='val_accuracy',mode='max', save_best_only=True)
    #
    # Should we load existing weights? 
    # Set argument for call to train_and_predict to true at end of script
    if use_existing:
        model.load_weights('./unet.hdf5')
        
    # 
    # The final results for this tutorial were produced using a multi-GPU
    # machine using TitanX's.
    # For a home GPU computation benchmark, on my home set up with a GTX970 
    # I was able to run 20 epochs with a training set size of 320 and 
    # batch size of 2 in about an hour. I started getting reseasonable masks 
    # after about 3 hours of training. 
    #
    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    # model.fit(imgs_train, imgs_mask_train, batch_size=2, nb_epoch=1, verbose=1, shuffle=True,
    #           callbacks=[model_checkpoint])
    
    model.fit(imgs_train, imgs_mask_train, batch_size=2, nb_epoch=100, verbose=1, shuffle=True,validation_data=(imgs_test, imgs_mask_test_true),
              callbacks=[model_checkpoint])
    

    # loading best weights from training session
    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    #model.load_weights('./unet.hdf5')
    #model.load_weights('unet_3sets.hdf5')
    model.load_weights('/media/adrian/963EAF133EAEEC05/Users/Adrián/Documents/Projects/Tensorflow/LUNA_Taurons_git/DSB3Tutorial-master/tutorial_code/unet_3sets100epochs.hdf5')
    #model.load_weights('/media/adrian/963EAF133EAEEC05/Users/Adrián/Documents/Projects/Tensorflow/LUNA_Taurons_git/DSB3Tutorial-master/tutorial_code/unet_3sets.hdf5')

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    num_test = len(imgs_test)
    imgs_mask_test = np.ndarray([num_test,1,512,512],dtype=np.float32)
    for i in range(num_test):
        imgs_mask_test[i] = model.predict([imgs_test[i:i+1]], verbose=0)[0]
    np.save('masksTestPredicted.npy', imgs_mask_test)
    mean = 0.0
    for i in range(num_test):
        mean+=dice_coef_np(imgs_mask_test_true[i,0], imgs_mask_test[i,0])
    mean/=num_test
    print("Mean Dice Coeff : ",mean)
    print('-'*30)
    print('Predicting masks on train data...')
    print('-'*30)
    num_train = len(imgs_train)
    imgs_mask_train = np.ndarray([num_train,1,512,512],dtype=np.float32)
    for i in range(num_train):
        imgs_mask_train[i] = model.predict([imgs_train[i:i+1]], verbose=0)[0]
    np.save('masksTrainPredicted.npy', imgs_mask_train)
    mean = 0.0
    for i in range(num_train):
        mean+=dice_coef_np(imgs_mask_train_true[i,0], imgs_mask_train[i,0])
    mean/=num_train
    print("Mean Train Dice Coeff : ",mean)



# def train_and_predict(use_existing):
#     print('-'*30)
#     print('Loading and preprocessing train data...')
#     print('-'*30)
#     imgs_train = np.load(working_path+"trainImages.npy").astype(np.float32)
#     imgs_mask_train = np.load(working_path+"trainMasks.npy").astype(np.float32)
#     imgs_mask_train_true = np.load(working_path+"trainMasks.npy").astype(np.float32)

#     imgs_test = np.load(working_path+"testImages.npy").astype(np.float32)
#     imgs_mask_test_true = np.load(working_path+"testMasks.npy").astype(np.float32)
    
#     #mean = np.mean(imgs_train)  # mean for data centering
#     #std = np.std(imgs_train)  # std for data normalization

#     #imgs_train -= mean  # images should already be standardized, but just in case
#     #imgs_train /= std

#     print('-'*30)
#     print('Creating and compiling model...')
#     print('-'*30)
#     all_train_scores = []
#     all_val_scores = []
#     for tt in range(10):
#         #print(tt)
#         model = get_unet()
#         # Saving weights to unet.hdf5 at checkpoints
#         #model_checkpoint = ModelCheckpoint('unet_bucle.hdf5', monitor='loss', save_best_only=True)
#         #
#         # Should we load existing weights? 
#         # Set argument for call to train_and_predict to true at end of script
#         if use_existing:
#             model.load_weights('./unet.hdf5')
            
            
#         if(tt !=0):
#             model.load_weights('unet_bucle.hdf5')
#         # 
#         # The final results for this tutorial were produced using a multi-GPU
#         # machine using TitanX's.
#         # For a home GPU computation benchmark, on my home set up with a GTX970 
#         # I was able to run 20 epochs with a training set size of 320 and 
#         # batch size of 2 in about an hour. I started getting reseasonable masks 
#         # after about 3 hours of training. 
#         #
#         # print('-'*30)
#         # print('Fitting model...')
#         # print('-'*30)
        
#         # model.fit(imgs_train, imgs_mask_train, batch_size=2, nb_epoch=5, verbose=1, shuffle=True,
#         #           callbacks=[model_checkpoint])
        
#         model.fit(imgs_train, imgs_mask_train, batch_size=2, nb_epoch=5, verbose=1, shuffle=True)
    
#         # loading best weights from training session
#         # print('-'*30)
#         # print('Loading saved weights...')
#         # print('-'*30)
#         #model.load_weights('./unet.hdf5')
#         #model.load_weights('unet.hdf5')
    
#         # print('-'*30)
#         # print('Predicting masks on test data...')
#         # print('-'*30)
        
#         print('*********************')
#         print(str(tt))
#         print('*********************')
#         num_test = len(imgs_test)
#         imgs_mask_test = np.ndarray([num_test,1,512,512],dtype=np.float32)
#         for i in range(num_test):
#             imgs_mask_test[i] = model.predict([imgs_test[i:i+1]], verbose=0)[0]
#         np.save('masksTestPredicted.npy', imgs_mask_test)
#         mean = 0.0
#         for i in range(num_test):
#             mean+=dice_coef_np(imgs_mask_test_true[i,0], imgs_mask_test[i,0])
#         mean/=num_test
#         print("Mean Dice Coeff : ",mean)
#         print('-'*30)
#         all_val_scores.append(mean)
#         # print('Predicting masks on train data...')
#         # print('-'*30)
#         num_train = len(imgs_train)
#         imgs_mask_train = np.ndarray([num_train,1,512,512],dtype=np.float32)
#         for i in range(num_train):
#             imgs_mask_train[i] = model.predict([imgs_train[i:i+1]], verbose=0)[0]
#         np.save('masksTrainPredicted.npy', imgs_mask_train)
#         mean = 0.0
#         for i in range(num_train):
#             mean+=dice_coef_np(imgs_mask_train_true[i,0], imgs_mask_train[i,0])
#         mean/=num_train
#         print("Mean Train Dice Coeff : ",mean)
#         all_train_scores.append(mean)
        
#         model.save_weights('unet_bucle.hdf5')
#     with open('train_scores.txt', 'w') as f:
#         for item in all_train_scores:
#             f.write("%s\n" % item)
    
#     with open('val_scores.txt', 'w') as f:
#         for item in all_val_scores:
#             f.write("%s\n" % item)

if __name__ == '__main__':
    train_and_predict(False)
