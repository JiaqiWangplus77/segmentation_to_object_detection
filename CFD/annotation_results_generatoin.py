#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 15:16:18 2019

@author: jiaqiwang0301@win.tu-berlin.de
28042020
merge the segmentation labels and the images
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

deal_with_all_files = 0
# if true, deal with all files of chosen type
# if false, deal with single file
if_merge = 1

image_num = 0  # raise error when out of range

# base folder is GAPs folder
#base_folder = os.getcwd()
# define the folder to extracct images and labels
image_folder = os.path.join('images/') 
label_folder = os.path.join('segmentation_labels/') 
# other option: only_class2

# create new folders to save the annotation results
annot_folder = os.path.join('segmentation_labels/annotation_result_merged/')
# other option: only_class2
os.makedirs(annot_folder,exist_ok= True)
prefix = 'annot_'
# 0:intact_road, 1:applied_patch, 2:pothole, 3:inlaid_patch, 4:open_joint, 5:crack
#data_type = [0,1,2,3,4,5]
data_type = [0,1]

# input part finished

labels = [os.path.join(label_folder,f) for f in os.listdir(label_folder)
             if f.endswith('png')]

images = [os.path.join(image_folder,f) for f in os.listdir(label_folder)
             if f.endswith('jpg')]

    
if deal_with_all_files:
    image_num = 0
    plot_num = range(len(labels))
else:
    plot_num = [image_num]


# define the name format(pure name or name with folder) for photo and txt file


#def new_dataset_num(data_type):
#    return np.arange(len(data_type))
#
#new_dataset_num = new_dataset_num(data_type)

#intact_road = (0,0,0)  # black
#applied_patch = (0,255,0)  # green
#pothole = (100,0,255)  # blue
#inlaid_patch = (255,255,255)   # white
#open_joint = (204,0,255) #purple
#crack = (255,0,0) # red
#colors = [intact_road,applied_patch, pothole, inlaid_patch, open_joint, crack]


for image_num in plot_num:
    
    label = np.array(Image.open(labels[image_num]))

    channel1 = label.copy()
    channel1[label==1] = 255
    channel1[label==0] = 255
    channel2 = np.ones_like(label) * 255
    channel2[label==1] = 0
    label_output = np.stack((channel2,channel2,channel1),axis=-1)

    if not deal_with_all_files:
        plt.figure()
        plt.imshow(label_output)   
    
    #label = np.stack((label,)*3, axis=-1)

    image_name = os.path.join(image_folder,
                              labels[image_num].split('/')[-1].replace('png','jpg'))

    
    if not os.path.isfile(image_name):  
        print(f'{image_name} file does not exist')
        
        pass
        #cv2.imwrite(annot_folder_name+imag_name2.format(image_num),np.zeros([imag_width, imag_height]))        

    else:
        image = np.array(Image.open(image_name))
        w_img = 0.8
        w_label = 0.2           
        image_merged = cv2.addWeighted(image,w_img,label_output,w_label,0)  
        if not deal_with_all_files:
            plt.figure()
            plt.imshow(image_merged) 
        #if_merge = 0 
        
        name = prefix + labels[image_num].split('/')[-1].replace('png','jpg')
        if if_merge:
            cv2.imwrite(os.path.join(annot_folder,name),
                       image_merged)
        else:
            cv2.imwrite(os.path.join(annot_folder,name),
                       label*255)
        



#        empty_image1 = np.ones_like(image) * 255
#        empty_image2 = np.ones_like(image) * 255        
#        yolo_table_whole = []
#        obj = np.loadtxt(label_filename).reshape(-1,5) # when there is only 1 object
#    #        obj_class = np.unique(obj[:,0])
#        obj_class = data_type
#    
#        for item in range(len(obj_class)):
#            #plt.figure()
#            #img = np.zeros([imag_height,imag_width,3],dtype=np.uint8)
#            index = np.array(np.where(obj[:,0] == data_type[item]))[0,:]   
#            if index.shape[0] == 0:
#                continue
##            # compared with rectangel bounding box
##            coordinate = np.array([imag_width * obj[index,1]-32-bbox_padding, 
##                                   imag_height * obj[index,2]-32-bbox_padding,
##                                   imag_width * obj[index,1]+32+bbox_padding, 
##                                   imag_height * obj[index,2]+32+bbox_padding]).astype(int).T
##           
##            for i in range(len(coordinate)):
##                bbox=[(coordinate[i,0],coordinate[i,1]),(coordinate[i,2],coordinate[i,3])]
##                cv2.rectangle(image, pt1=bbox[0], pt2=bbox[1],color=colors[item],thickness=-1) 
#             
#            # compared with circle bounding box
#            coordinate = np.array([imag_width * obj[index,1], 
#                                   imag_height * obj[index,2]
#                                   ]).astype(int).T
#            # draw bbox, fill the bounding box with color
#            data_num = int(new_dataset_num[item])
#            labels = np.ones([imag_height,imag_width])
#            for i in range(len(coordinate)):
#                bbox=[(coordinate[i,0],coordinate[i,1])]
#                cv2.circle(labels, center=bbox[0], radius=radius, color=(data_num),thickness=-1) 
#                cv2.circle(empty_image2, center=bbox[0], radius=radius, color=colors[item],thickness=-1)  
##            if not deal_with_all_files:
##                plt.figure()
##                plt.imshow(image) 
##                plt.title("after circle")                             
#            
#
#            
#        #image_combine = np.concatenate((image_original,image,image2), axis=1)    
#        w_img = 0.8
#        w_label = 0.2
#            
#        image_merged = cv2.addWeighted(image,w_img,empty_image2,w_label,0)
#        if not deal_with_all_files:
#            plt.figure()
#            plt.imshow(image_merged)
#            plt.title('circle bounding box')  
#            
#            
##            plt.figure()
##            plt.imshow(image2)  
##            plt.title('original rectangle bounding box')
##            plt.figure()
##            plt.imshow(image_combine)  
##            plt.title('image combination')            
#            
#            
##        cv2.imwrite(image_folder_name+imag_name2.format(image_num),image_merged)
##        cv2.imwrite(annot_folder_name+imag_name2.format(image_num),labels)
#        
#        print(imag_name2.format(image_num), ' finished')
#        

