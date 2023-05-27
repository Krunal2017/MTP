
# coding: utf-8

# In[1]:


import cv2
import argparse
import imutils
import time
import scipy.io
# from sklearn.svm import SVC
import skvideo.io
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from scipy.ndimage import morphology as mp
from skimage.feature import canny
from thundersvm import SVC
from random import sample
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction import image
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt
import mahotas
from imutils.video import VideoStream
from imutils.video import FPS
import os
import pickle
import copy


# In[2]:


label=40
psize=8
basepath = "training/"+str(label)+"/"
loadpath = basepath+"train/"

svc = SVC()
svc.load_from_file("output/40/"+"vid2_svc.pkl")

##### Testing for stride 1 ########
basepath = "training/"+str(label)+"/"
loadpath = basepath+"test/"
storepath = basepath+"output/"
if not os.path.exists(storepath):
    os.mkdir(storepath)
for n in range(1,2):
    print("Frame:"+str(n));
    next_img = cv2.imread(loadpath+'next_frame'+str(n)+'.png');
    im_max = max(next_img.flatten())
    im_min = min(next_img.flatten())
    next_img = (next_img - im_min)/(im_max - im_min)
#     next_img_hsv = cv2.cvtColor(next_img, cv2.COLOR_BGR2HSV)
#     next_img_hsv = next_img_hsv[:,:,0:1]
    
    next_patches = image.extract_patches_2d(next_img, (psize, psize))
#     next_patches = image.extract_patches_2d(next_img_hsv, (psize, psize))
    TEST=[]
    for i in range(len(next_patches)):
        a = next_patches[i].flatten()
        TEST.append(a)
    predictions = svc.predict(TEST)
    mask = np.zeros_like(next_patches)
    for i in range(len(predictions)):
        if(predictions[i]==1):
            mask[i] = mask[i] + 1
    ## Using thresholding on the image
    reconstructed = []
    reconstructed = image.reconstruct_from_patches_2d(mask, next_img.shape)
    # plt.imshow(reconstructed),plt.show();
    ret,thresh1 = cv2.threshold(reconstructed,0.6,1,cv2.THRESH_BINARY)
    # plt.imshow(thresh1),plt.show();
    
    mask2 = thresh1.astype(np.bool)
    silhouette = np.zeros_like(next_img)
    silhouette[mask2] = next_img[mask2]
    
    ## Using maximally connected components on the image
#     reconstructed = []
#     reconstructed = image.reconstruct_from_patches_2d(mask, next_img.shape)
#     plt.imshow(reconstructed),plt.show();
#     mask2 = reconstructed.astype(np.bool)
#     silhouette = np.zeros_like(next_img)
#     silhouette[mask2] = next_img[mask2]
    
#     gray_sil = cv2.cvtColor(silhouette, cv2.COLOR_BGR2GRAY)
#     connected_mask = undesired_objects(gray_sil)
#     stacked_img = np.stack((connected_mask,)*3, axis=-1)
    
#     plt.imshow(stacked_img),plt.show();
    
#     mask = stacked_img.astype(np.bool)
#     silhouette = np.zeros_like(next_img)
#     silhouette[mask] = next_img[mask]
    
    cv2.imwrite(storepath+'silhouette_frame'+str(n)+'.png',silhouette);


