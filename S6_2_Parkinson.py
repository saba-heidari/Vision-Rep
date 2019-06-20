# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 13:42:46 2019

@author: Saba
"""

from imutils import paths
import cv2
import numpy as np
import pandas as pd
from skimage import feature
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
image_list =[]
label_list = []
hog_list = []

image_list_t =[]
label_list_t = []
hog_list_t = []


for address in paths.list_images("dataset\\spiral\\training"):
      
    #print(address)
    image = cv2.imread(address)
    img = cv2.resize(image, (250, 250))
    image_list.append(img)
  # make labels and save it
    label = address.split("\\")[-2]
    label_list.append(label)
    
    (H, hogImage) = feature.hog(img, orientations=8, pixels_per_cell=(10, 10),  cells_per_block=(3,3), block_norm = "L1", transform_sqrt = True, visualize = True)  
    hog_list.append(H)
    
    
clf = KNeighborsClassifier(n_neighbors= 3)
clf.fit(hog_list, label_list)
#print("accuracy_train: {}".format(clf.fit(hog_list, label_list)))


for address in paths.list_images("dataset\\spiral\\testing"):
    image_t = cv2.imread(address)
    img_t = cv2.resize(image_t, (250, 250))
    image_list_t.append(img_t)
   
    label_t = address.split("\\")[-2]
    label_list_t.append(label_t)
    
    (H_t, hogImage) = feature.hog(img_t, orientations=8, pixels_per_cell=(10, 10),  cells_per_block=(3,3), block_norm = "L1", transform_sqrt = True, visualize = True)  
    hog_list_t.append(H_t)
    
#clf.score(hog_list_t, label_list_t)
print("accuracy_test: {}".format(clf.score(hog_list_t, label_list_t)))








 
  
  
  