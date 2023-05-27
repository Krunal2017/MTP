
# coding: utf-8

# In[2]:


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
import matplotlib.patches as patches
from PIL import Image
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


# In[1]:


label=9
basepath = "training/"+str(label)+"/"
loadpath = basepath+"train/"
img = cv2.imread(loadpath+'frame.png');
im_max = max(img.flatten())
im_min = min(img.flatten())
img = (img - im_min)/(im_max - im_min)
sil = cv2.imread(loadpath+'gt.png');

plt.imshow(img),plt.show();
plt.imshow(sil*255),plt.show();


x1,y1 = indices[0][0,0],indices[1][0,0]


# im = img[0:8,0:8]
patches=[]
GT=[]
psize=8
patches = image.extract_patches_2d(img, (psize, psize))
GT = image.extract_patches_2d(sil, (psize, psize))
X=[]
Y=[]
i=0;
print("Total Patches:"+str(len(patches)));
while i < len(patches):
# for i in range(len(patches)):
    a = patches[i].flatten()
    if(np.count_nonzero(GT[i])>=(psize*psize/2)*3):
        Y.append(1)
    else:
        Y.append(0)

    X.append(a)
    i=i+1


# In[23]:


svc = SVC(C=1,kernel='rbf',gamma=1/(X[0].shape[0]*np.array(X).var()))
svc.fit(X,Y)
svc.save_to_file('usual-rbf-class9.svc')


# In[69]:


def trainVideo2(file,boxes,label,filename):
    vs = cv2.VideoCapture(file)
    
    # loop over frames from the video stream
    fno = 1
    disp_flag=False
    capture=True
    next_frame=[]
    corners=[]
    img=[]
    storepath = 'output/'+str(label)+"/"
    nos = 100
    if not os.path.exists(storepath):
        os.mkdir(storepath)
    fname = filename.strip('.mp4')
    writer = skvideo.io.FFmpegWriter(storepath+fname+"_"+str(nos)+".mp4")
    
    def mouse_click(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            corners.append([x,y])
            for i in corners:
                [x,y] = i
                cv2.circle(cropped_im,(x,y),3,255,-1)


    cur_no=0
    count = 1
    while True:
        # grab the current frame, then handle if we are using a
        # VideoStream or VideoCapture object
        frame1 = vs.read()
        frame = frame1[1] # if args.get("video", False) else frame1
        # frame = fgbg.apply(frame)

        # check to see if we have reached the end of the stream
        if frame is None:
            break

        # resize the frame (so we can process it faster) and grab the
        # frame dimensions
        frame = imutils.resize(frame, width=500)
        (H, W) = frame.shape[:2]

        # check to see if we are currently tracking an object
        if fno in boxes:
            (x1,y1,x2,y2) = boxes[fno]

            if x2<x1:
                temp=x2
                x2=x1
                x1=temp
            if y2<y1:
                temp=y2
                y2=y1
                y1=temp

            if x2>W:
                x2=W
            if y2>H:
                y2=H
            if x1<0:
                x1=0
            if y1<0:
                y1=0
            cropped_im = frame[y1:y2,x1:x2]

            print("Frame:"+str(fno));

            if(capture):
                next_img = predictSilhouette2(cropped_im)
                cv2.imwrite('silhouette_frame'+str(n)+'.png',next_img);
                capture=False
                continue

                [m,n,p] = next_img.shape
                diff = 400 - m
                top = int(diff/2)
                bot = diff -top
                diff = 500 - n
                left = int(diff/2)
                right = diff -left
                padded_img= cv2.copyMakeBorder(next_img,top,bot,left,right,cv2.BORDER_CONSTANT,value=[0,0,0])
                padded_img = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
                writer.writeFrame(padded_img)
#                 out.write(padded_img)
                cur_no = cur_no+1
                if(cur_no>=nos):
                    count=count+1
                    nos=100*count
                    writer.close()
                    writer = skvideo.io.FFmpegWriter(storepath+fname+"_"+str(nos)+".mp4")


        fno=fno+1

    vs.release()
    writer.close()

    return frame_array


# In[70]:


############################# Method for testing with Strides ###################################
def predictSilhouette2(next_img):

    i=0;
    TEST1=None
    im_max = max(next_img.flatten())
    im_min = min(next_img.flatten())
    next_img_normed = (next_img - im_min)/(im_max - im_min)
    next_patches = image.extract_patches_2d(next_img_normed, (8, 8))
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
    plt.imshow(reconstructed),plt.show();

    mask2 = reconstructed.astype(np.bool)
    silhouette = np.zeros_like(next_img)
    silhouette[mask2] = next_img[mask2]
    
    gray_sil = cv2.cvtColor(silhouette, cv2.COLOR_BGR2GRAY)
    connected_mask = undesired_objects(gray_sil)
    stacked_img = np.stack((connected_mask,)*3, axis=-1)
    
    edges = canny(connected_mask)
    fill_holes = morph.binary_fill_holes(edges)
    stacked_img2 = np.stack((fill_holes,)*3, axis=-1)
    stacked_img = stacked_img + stacked_img2
    
    mask = stacked_img.astype(np.bool)
    silhouette = np.zeros_like(next_img)
    silhouette[mask] = next_img[mask]
    
    
    
    plt.imshow(silhouette),plt.show();
    
#     ## Using maximally connected components on the image
#     reconstructed = []
#     reconstructed = image.reconstruct_from_patches_2d(mask, next_img.shape)
    
#     ret,thresh1 = cv2.threshold(reconstructed,0.6,1,cv2.THRESH_BINARY)
#     print("Thresholded Mask:");
#     plt.imshow(thresh1),plt.show();
    
# #     plt.imshow(reconstructed),plt.show();
#     mask2 = thresh1.astype(np.bool)
#     silhouette = np.zeros_like(next_img)
#     silhouette[mask2] = next_img[mask2]
    
#     gray_sil = cv2.cvtColor(silhouette, cv2.COLOR_BGR2GRAY)
#     connected_mask = undesired_objects(gray_sil)
#     stacked_img = np.stack((connected_mask,)*3, axis=-1)
    
#     plt.imshow(stacked_img),plt.show();
    
#     mask = stacked_img.astype(np.bool)
#     silhouette = np.zeros_like(next_img)
#     silhouette[mask] = next_img[mask]
#     plt.imshow(silhouette),plt.show();
    return silhouette


# In[71]:


def undesired_objects (image):
    image = image.astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
    sizes = stats[:, -1]

    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    img2 = np.zeros(output.shape)
    img2[output == max_label] = 1
    return img2



basePath = "../IndianBirds/"
frame_array=[]
for i in range(9,10):
    boxes = {}
    fullPath = basePath+str(i)+"/"
    allfiles = os.listdir(fullPath)
    f = open(fullPath+"bbox.pkl","rb")
    boxes=pickle.load(f)
    f.close()
    j=0;
#     box = boxes[allfiles[0]][221]
    for file in allfiles:
#     for file in ["C3wkLDsVUwM.mp4"]:
        if file.endswith(".mp4"):
#             if(j==1):
            frame_array = trainVideo2(fullPath+file,boxes[file],i,file)
            j=j+1
        if (j==1):
            break


