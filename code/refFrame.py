
# coding: utf-8

# In[3]:


import cv2
import argparse
import imutils
import time
import scipy.io
# from sklearn.svm import SVC
import skvideo.io
from sklearn.decomposition import PCA
# from sklearn.manifold import Isomap
# from sklearn.manifold import LocallyLinearEmbedding
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
from imutils.video import VideoStream
from imutils.video import FPS
import os
import pickle
import copy


# In[4]:


def trainVideo(file,boxes,label,filename):
    vs = cv2.VideoCapture(file)
    
    # loop over frames from the video stream
    fno = 1
    
#     X=None
#     Y=None
    disp_flag=False
    capture=True
    next_frame=[]
    corners=[]
    img=[]
    nos = 20
#     storepath = 'output/'+str(label)+"/"
# #     storepath="dataset/"+str(label)+"/"+filename
#     if not os.path.exists(storepath):
#         os.mkdir(storepath)
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter(storepath+filename,fourcc, 20.0, (400,500))
    def mouse_click(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            corners.append([x,y])
#             print("fno"+str(fno))
            for i in corners:
                [x,y] = i
#                 cropped_im[y,x]=255
                cv2.circle(cropped_im,(x,y),3,255,-1)
#     label=None
#     TEST=None
#     GT=None
#     storepath="dataset/"+str(label)+"/"+filename
#     if not os.path.exists("dataset/"+str(label)):
#         os.mkdir("dataset/"+str(label))
#     if not os.path.exists(storepath):
#         os.mkdir(storepath)
    cv2.namedWindow('Frame')
    cv2.setMouseCallback("Frame", mouse_click)
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
#             cv2.rectangle(frame, (x1, y1), (x2, y2),
#                         (0, 255, 0), 2)
            if x2<x1:
                temp=x2
                x2=x1
                x1=temp
            if y2<y1:
                temp=y2
                y2=y1
                y1=temp
#             print(frame.shape)
#             print(boxes[fno])
            if x2>W:
                x2=W
            if y2>H:
                y2=H
            if x1<0:
                x1=0
            if y1<0:
                y1=0
            cropped_im = frame[y1:y2,x1:x2]
            [m,n,p] = cropped_im.shape
            diff = 400 - m
            top = int(diff/2)
            bot = diff -top
            diff = 500 - n
            left = int(diff/2)
            right = diff -left
            constant= cv2.copyMakeBorder(cropped_im,top,bot,left,right,cv2.BORDER_CONSTANT,value=[0,0,0])
#             print(constant.shape);
            if(capture):
#                 next_img = predictSilhouette(cropped_im)
#                 [m,n,p] = next_img.shape
#                 diff = 400 - m
#                 top = int(diff/2)
#                 bot = diff -top
#                 diff = 500 - n
#                 left = int(diff/2)
#                 right = diff -left
#                 padded_img= cv2.copyMakeBorder(next_img,top,bot,left,right,cv2.BORDER_CONSTANT,value=[0,0,0])
#                 out.write(padded_img)
                storepath = "training/"+str(label)+"/"
                if not os.path.exists(storepath):
                    os.mkdir(storepath)
                
                storepath = "training/"+str(label)+"/test/"
                if not os.path.exists(storepath):
                    os.mkdir(storepath)
                cv2.imwrite(storepath+'next_frame'+str(20-nos+1)+'.png',cropped_im)
                nos = nos-1
                if(nos==0):
                    capture=False

#             print(fno)                

            # show the output frame
#             cv2.imwrite(storepath+"/"+str(fno)+".png",cropped_im)
            key = cv2.waitKey(20) & 0xFF
            
#         # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
            if key == ord('p'):
                corners=[]
                img = copy.copy(cropped_im)
                while True:

                    key2 = cv2.waitKey(1) or 0xff
                    cv2.imshow('Frame',cropped_im)
                    
                    # cv2.imshow('frame', frame)

                    if key2 == ord('p'):
                        disp_flag=True
                        break
            if disp_flag:
#                 img = frame[y1:y2,x1:x2]
#                 plt.imshow(img),plt.show()
#                 img = newim
                mask = np.zeros((img.shape[0], img.shape[1]))
                cv2.fillPoly(mask, [np.array(corners)], 1)
                
                storepath = "training/"+str(label)+"/"
                if not os.path.exists(storepath):
                    os.mkdir(storepath)
                storepath = "training/"+str(label)+"/train/"
                if not os.path.exists(storepath):
                    os.mkdir(storepath)
                f1 = filename.strip('.mp4')
                print(storepath+f1+'-frame.png')
                cv2.imwrite(storepath+f1+'-frame.png',img)
                cv2.imwrite(storepath+f1+'-gt.png',mask)
#                 cv2.imshow("Mask:", mask)
                mask = mask.astype(np.bool)
#                 cv2.imshow("Mask:", mask)

                silhouette = np.zeros_like(img)
                silhouette[mask] = img[mask]
#                 # delete zero columns
#                 silhouette= np.delete(silhouette,np.where(~silhouette.any(axis=0))[0], axis=1)
#                 # delete zero rows
#                 silhouette= np.delete(silhouette,np.where(~silhouette.any(axis=1))[0], axis=0)
                
    
#                 mask = np.zeros_like(frame,np.uint8)
#                 rect = (x1,y1,x2-x1,y2-y1)
#                 cv2.grabCut(frame,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

#                 mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
# #                 mask2.resize(frame.shape)
#                 img = np.zeros([frame.shape[0],frame.shape[1],1,3])
#                 img = img*mask2[:,:,np.newaxis]

#                 mask = mask.astype(np.int8)
#                 mask = np.where((mask==255)|(mask==0),0,1).astype('uint8')
#                 mask, bgdModel, fgdModel = cv2.grabCut(img,mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
#                 mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
#                 img = img*mask[:,:,np.newaxis]
#                 cv2.imshow("Grabcut:", img)
#                 cv2.namedWindow('Silhouette:')
                cv2.imwrite(storepath+f1+'-extracted_silhouette.png',silhouette)
#                 cv2.imshow("Silhouette:", silhouette)
                disp_flag=False
#                 capture = True
#             gray = cv2.cvtColor(cropped_im,cv2.COLOR_BGR2GRAY)

#             corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
#             corners = np.int0(corners)
#             print (corners)
#             cv2.drawContours(cropped_im, corners, -1, (0, 255, 0), 3)
#             for i in corners:
#                 [x,y] = i
#                 cv2.circle(cropped_im,(x,y),3,255,-1)

#             plt.imshow(img),plt.show()
            cv2.imshow("Frame", cropped_im)
            
#             key = cv2.waitKey(1) & 0xFF

# #         # if the `q` key was pressed, break from the loop
#             if key == ord("q"):
#                 break

        fno=fno+1

    vs.release()
#     out.release()
    
    # close all windows
    cv2.destroyAllWindows()
#     data={'X':X,'Y':Y}
#     return data


# In[6]:


basePath = "../IndianBirds/"
# X=None
# Y=None
# TEST=None
# GT=None
# a = {24 :["66ehMV8l17w", "h10IuNIRpgQ"],
# 22:["jSC8Wy3WhaI", "LkCuK5H0JSc"],
# 23:["-iHTKofq5cM", "8VWIz-yyYsw"]}
for i in range(25,30):
    boxes = {}
    fullPath = basePath+str(i)+"/"
    allfiles = os.listdir(fullPath)
    if not os.path.exists(fullPath+"bbox-2.pkl"):
    	continue
    f = open(fullPath+"bbox-2.pkl","rb")
    boxes=pickle.load(f)
    f.close()
#     allfiles.remove('C3wkLDsVUwM.mp4')
    j=0;
#     box = boxes[allfiles[0]][221]
    for file in boxes.keys():
#     for file in ["C3wkLDsVUwM.mp4"]:
#     for file in a[i]:
#         file = file+'.mp4'
#         if file.endswith(".mp4"):
        trainVideo(fullPath+file,boxes[file],i,file)
#             if i==1:
#             if X is not None:
#                 X.append(data['X'])
#             else:
#                 X=data['X']

#             if Y is not None:
#                 Y.append(data['Y'])
#             else:
#                 Y=data['Y']

        # j=j+1
        # if (j==2):
        #     break




