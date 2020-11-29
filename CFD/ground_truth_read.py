#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 13:44:36 2019

@author: jiaqiwang0301@win.tu-berlin.de
generate png format segmentation files
"""

import os
import scipy.io as sio
import numpy as np

import cv2
import matplotlib.pyplot as plt

label_folder = 'contour'
os.makedirs(label_folder,exist_ok= True)

path = os.path.join('groundTruth')
filename = [file for file in os.listdir(path)
              if file.endswith('.mat')][0]
filename = [filename]

for file in filename:
    dic = sio.loadmat(os.path.join(path, file))
    matrix = np.array(dic['groundTruth'][0])
    labels = matrix[0][0] * 255
    
    s = np.unique(labels,return_counts=1) 
    print(s)
    
    plt.figure()
    plt.imshow((labels), cmap='gray')
    
    label_name = os.path.join(label_folder,'contour_'+file.replace('.mat','.png'))
#    cv2.imwrite(label_name,labels)
    
    print(file, 'finished')
    break

    


      
# original
#for file in filename:
#    dic = sio.loadmat(os.path.join(path, file))
#    matrix = np.array(dic['groundTruth'][0])
#    labels = matrix[0][0]-1
#    
#    s = np.unique(labels,return_count=1)
#    labels[labels==2] = 0
#    labels[labels==3] = 1    
#    
##    plt.figure()
##    plt.imshow((labels-1)*255, cmap='gray')
#    
##    label_name = os.path.join(label_folder,file.replace('.mat','.png'))
##    cv2.imwrite(label_name,labels)
#    print(file, 'finished')
#    break


# label test    
#c2 = [] # class2
#c3 = [] # class3
#for file in filename:
#    dic = sio.loadmat(os.path.join(path, file))
#    matrix = (dic['groundTruth'][0])
#    labels = matrix[0][0]-1
#    
#    
#    if sum(labels[labels==2]) > 0:
#        c2.append(file)
#        labels[labels==3] = 0
#        labels[labels==1] = 0  
#        labels[labels==2] = 1
#        plt.figure()       
#        plt.imshow((labels)*255, cmap='gray')
#        label_name = os.path.join(label_folder,file.replace('.mat','.png'))
#        cv2.imwrite(label_name,labels)                    
#    if sum(labels[labels==3]) > 0:
#        c3.append(file)
#        labels[labels==2] = 0
#        labels[labels==1] = 0  
#        labels[labels==3] = 1
#        print(np.unique(labels,return_counts=1))
#        plt.figure()       
#        plt.imshow((labels)*255, cmap='gray')
#        label_name = os.path.join(label_folder,file.replace('.mat','.png'))
#        cv2.imwrite(label_name,labels)        




