#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 18:34:30 2020

@author: adrian
"""

import numpy as np
import pickle
import cv2
from skimage.measure import label,regionprops, perimeter

def getRegionFromMap(slice_npy):
    #thr = np.where(slice_npy > np.mean(slice_npy),0.,1.0)
    #thr = np.where(slice_npy > np.mean(slice_npy),0.,1.0)
    thr = np.where(slice_npy > np.mean(slice_npy),1.,0.0)
    cv2.imwrite('/media/adrian/963EAF133EAEEC05/Users/Adrián/Documents/Projects/Tensorflow/LUNA_Taurons_git/DSB3Tutorial-master/tutorial_code/folder_check/th.jpg', thr*255)
    print(thr)
    #import pdb;pdb.set_trace()
    label_image = label(thr)
    labels = label_image.astype(int)
    regions = regionprops(labels)
    return regions

def getRegionMetricRow(img):
    seg = np.load(fname)
    nslices = seg.shape[0]
    #metrics
    totalArea = 0.
    avgArea = 0.
    maxArea = 0.
    avgEcc = 0.
    avgEquivlentDiameter = 0.
    stdEquivlentDiameter = 0.
    weightedX = 0.
    weightedY = 0.
    numNodes = 0.
    numNodesperSlice = 0.
    # do not allow any nodes to be larger than 10% of the pixels to eliminate background regions
    maxAllowedArea = 0.10 * 512 * 512 
    areas = []
    eqDiameters = []
    for slicen in range(nslices):
        regions = getRegionFromMap(seg[slicen,0,:,:])
        for region in regions:
            if region.area > maxAllowedArea:
                continue
            totalArea += region.area
            areas.append(region.area)
            avgEcc += region.eccentricity
            avgEquivlentDiameter += region.equivalent_diameter
            eqDiameters.append(region.equivalent_diameter)
            weightedX += region.centroid[0]*region.area
            weightedY += region.centroid[1]*region.area
            numNodes += 1
    weightedX = weightedX / totalArea 
    weightedY = weightedY / totalArea
    avgArea = totalArea / numNodes
    avgEcc = avgEcc / numNodes
    avgEquivlentDiameter = avgEquivlentDiameter / numNodes
    stdEquivlentDiameter = np.std(eqDiameters)
    maxArea = max(areas)
    numNodesperSlice = numNodes*1. / nslices
    return np.array([avgArea,maxArea,avgEcc,avgEquivlentDiameter,\
                     stdEquivlentDiameter, weightedX, weightedY, numNodes, numNodesperSlice])
        

        
img_path = '/media/adrian/963EAF133EAEEC05/Users/Adrián/Documents/Projects/Tensorflow/LUNA_Taurons_git/DSB3Tutorial-master/tutorial_code/folder_check/33pred.jpg'

img = cv2.imread(img_path, 0)



totalArea = 0.
avgArea = 0.
maxArea = 0.
avgEcc = 0.
avgEquivlentDiameter = 0.
stdEquivlentDiameter = 0.
weightedX = 0.
weightedY = 0.
numNodes = 0.
numNodesperSlice = 0.
# do not allow any nodes to be larger than 10% of the pixels to eliminate background regions
maxAllowedArea = 0.10 * 512 * 512 
areas = []
eqDiameters = []
regions = getRegionFromMap(img)
for region in regions:
    if region.area > maxAllowedArea or region.area<100:
        continue
    totalArea += region.area
    areas.append(region.area)
    avgEcc += region.eccentricity
    avgEquivlentDiameter += region.equivalent_diameter
    eqDiameters.append(region.equivalent_diameter)
    weightedX += region.centroid[0]*region.area
    weightedY += region.centroid[1]*region.area
    numNodes += 1



img = np.load('/media/adrian/963EAF133EAEEC05/Users/Adrián/Documents/Projects/Tensorflow/LUNA_Taurons_git/DSB3Tutorial-master/tutorial_code/folder_check/img.npy')[:,0,:,:]

thr = np.where(img > np.mean(img),1.,0.0)
areas =[]
density =[]
label_image = label(thr)
labels = label_image.astype(int)
regions = regionprops(labels)
al_info = []
all_info2 = []
for region in regions:
    if region.area<500:
        continue
    areas.append(region.area)
    #test = img[region.bbox]

    test = img[region.bbox[0]:region.bbox[3],region.bbox[1]:region.bbox[4],region.bbox[2]:region.bbox[5]]
    dens = np.sum(test)/region.area
    al_info.append(str(region.centroid)+'  '+str(dens))
    all_info2.append([region.centroid,dens,region.area,])

all_info2.sort(key=lambda x: x[1])










