
# coding: utf-8

# In[71]:


import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy import ndimage
import math
from scipy.misc import imsave
import sys

# Input : python imageStitching.py img window_size std ksize_x ksize_y k tf

# img : Input image path
# window_size : dimension of gaussian kernel
# std : standard deviation of gaussian kernel
# ksize_x, ksize_y : dimensions of sobel operator
# k : Harris Detector free parameter
# tf : Thresholding factor

# Output : Stitched image


# In[72]:


# Read the image
def readImage(img):
    img = cv2.imread(img, 0)
    return img


# In[73]:


# Apply gaussian filter
def gaussianBlur(img, window_size, std):
    img = cv2.GaussianBlur(img,(window_size, window_size), std)
    return img


# In[74]:


# Generate gaussian kernel
def getGaussianKernel(window_size, sigma):
    offset = int(window_size / 2)
    gaussian_weights = np.zeros((window_size, window_size))
    for i in range(-offset, offset + 1):
        for j in range(-offset, offset + 1):
            gaussian_weights[i + offset][j + offset] = np.exp(-(i**2 + j**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    #Normalizing the gaussian kernel
    gaussian_weights = gaussian_weights / np.sum(gaussian_weights)
    return gaussian_weights


# In[75]:


#Use sobel operator to get the corner list
def getCornerList(img, ksize_x, ksize_y, window_size, std, k):
    dx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize_x)
    dy = cv2.Sobel(img,cv2.CV_64F,0,1,ksize_y)
    Ixx = dx**2
    Iyy = dy**2
    Ixy = dx*dy
    x, y = img.shape
    offset = int(window_size / 2)
    gaussian_weights = getGaussianKernel(window_size, std)
    corners = []
    for i in range(offset, x - offset):
        for j in range(offset, y - offset):
        
            #Getting size of gradient matrices as the window_size 
            Ixx_window = Ixx[i-offset : i+offset+1, j-offset : j+offset+1]
            Iyy_window = Iyy[i-offset : i+offset+1, j-offset : j+offset+1]
            Ixy_window = Ixy[i-offset : i+offset+1, j-offset : j+offset+1]
        
            #Calculating weighted gradient matrices
            Ixx_weights = Ixx_window * gaussian_weights
            Iyy_weights = Iyy_window * gaussian_weights
            Ixy_weights = Ixy_window * gaussian_weights
        
            #Summing up the weights
            sum_xx = np.sum(Ixx_weights)
            sum_yy = np.sum(Iyy_weights)
            sum_xy = np.sum(Ixy_weights)
        
            #Calculating determinant
            det = (sum_xx * sum_yy) - (sum_xy**2)
            #Calculating trace
            trace = sum_xx + sum_yy
            R = det - k*(trace**2)
            corners.append([i,j,R])
    return np.array(corners)


# In[76]:


# Mark corners
def markCorners(img, corners, tf, window_size):
    x, y = img.shape
    offset = int(window_size / 2)
    max_cornerVal = np.max(corners[:,2])
    threshold = tf * max_cornerVal
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    cnt = 0
    for i in range(offset, x - offset):
        for j in range(offset, y - offset):
            if (corners[cnt][2] > threshold):
                img.itemset((i,j,0), 0)
                img.itemset((i,j,1), 0)
                img.itemset((i,j,2), 255)
            cnt = cnt + 1
    return img


# In[77]:


img1 = "Inputs/1.jpg"
img2 = "Inputs/2.jpg"
window_size = 5
std = 1
ksize_x = ksize_y = 5
k = 0.05
tf = 0.1
img1 = readImage(img1)
img2 = readImage(img2)
img1 = gaussianBlur(img1, window_size, std)
img2 = gaussianBlur(img2, window_size, std)
corners1 = getCornerList(img1, ksize_x, ksize_y, window_size, std, k)
corners2 = getCornerList(img2, ksize_x, ksize_y, window_size, std, k)
corners_img1 = markCorners(img1, corners1, tf, window_size)
corners_img2 = markCorners(img2, corners2, tf, window_size)

plt.subplot(2,1,1)
plt.imshow(corners_img1)
cv2.imwrite("corners1.jpg", corners_img1)

plt.subplot(2,1,2)
plt.imshow(corners_img2)
cv2.imwrite("corners2.jpg", corners_img2)


# In[86]:


# original
# common_corners1 = np.array([(452,158),(478,144),(485,140),(474,230),(547,248),(575,250),(603,174),(617,178),(655,130),(752,176),(757,146),(760,132),(740,143)])
# common_corners2 = np.array([(73,186),(100,164),(107,160),(115,252),(185,257),(211,249),(216,173),(227,174),(258,125),(337,161),(338,133),(338,125),(330,133)])

# reverse
common_corners1 = np.array([(158,452),(144,478),(140,485),(230,474),(248,547),(250,575),(174,603),(178,617),(130,655),(176,752),(146,757),(132,760),(143,740),(116,762),(63,782),(54,647),(52,637),(44,638),(14,621),(134,458),(133,422),(134,491),(61,781),(42,620)])
common_corners2 = np.array([(186,73),(164,100),(160,107),(252,115),(257,185),(249,211),(173,216),(174,227),(125,258),(161,337),(133,338),(125,338),(133,330),(110,337),(61,346),(55,234),(53,224),(45,223),(24,218),(156,70),(159,23),(153,114),(61,346),(40,219)])


# In[176]:


def get_perspectiveTransformMatrix(common_corners1,common_corners2):
    #num = common_corners1.shape[0]
    num = 16
    A = np.zeros((2*num,9))
    common_corners1 = np.array(common_corners1)
    common_corners2 = np.array(common_corners2)
    for i in range(num):
        x,y = common_corners1[i][0],common_corners1[i][1]
        u,v = common_corners2[i][0],common_corners2[i][1]
        A[2*i] = [-x,-y,-1,0,0,0,u*x,u*y,u]
        A[2*i+1] = [0,0,0,-x,-y,-1,v*x,v*y,v]
    U,S,V = np.linalg.svd(A)
    print("This value should be close to zero: "+str(np.sum(np.dot(A,V[8]))))
    H = V[8].reshape((3,3))        
    return H 


# In[177]:


H = get_perspectiveTransformMatrix(common_corners1, common_corners2)


# In[178]:


H


# In[179]:


img1 = "Inputs/1.jpg"
img2 = "Inputs/2.jpg"
orig_image1 = readImage(img1)
orig_image2 = readImage(img2)
ref_image1 = orig_image1
ref_image2 = orig_image2


# In[180]:


def apply_transform(image1, image2):
#     plt.subplot(2,1,1)
#     plt.imshow(image1)
#     plt.subplot(2,1,2)
#     plt.imshow(image2)
    bX=[]; bY=[]
    tt = np.array([[0],[0],[1]])
    t = np.dot(H,tt)
    bX.append(t[0]/t[2])
    bY.append(t[1]/t[2])

    tt = np.array([[image1.shape[0]-1],[0],[1]])
    t = np.dot(H,tt)
    bX.append(t[0]/t[2])
    bY.append(t[1]/t[2])

    tt = np.array([[0],[image1.shape[1]-1],[1]])
    t = np.dot(H,tt)
    bX.append(t[0]/t[2])
    bY.append(t[1]/t[2])

    tt = np.array([[image1.shape[0]-1],[image1.shape[1]-1],[1]])
    t = np.dot(H,tt)
    bX.append(t[0]/t[2])
    bY.append(t[1]/t[2])

    refX1 = int(np.min(bunchX))
    refX2 = int(np.max(bunchX))
    refY1 = int(np.min(bunchY))
    refY2 = int(np.max(bunchY))
    print (refX1, refX2, refY1, refY2)

# Final image whose size is defined by the offsets previously calculated
    final = np.zeros((int(refX2-refX1),int(refY2-refY1)+image2.shape[1]))
# Iterate over the image to transform every pixel
    for i in range(image1.shape[0]):
            for j in range(image1.shape[1]):
                tt = np.array([[i],[j],[1]])
                tmp = np.dot(H,tt)
                x1=int(tmp[0]/tmp[2])-refX1
                y1=int(tmp[1]/tmp[2])-refY1

                if x1>0 and y1>0 and y1<refY2-refY1 and x1<refX2-refX1:
                        final[x1,y1]=ref_image1[i,j]
    for i in range(image2.shape[0]):
        for j in range(image2.shape[1]):
            if (final[i-refX1][j-refY1]==0):
                final[i-refX1][j-refY1]=image2[i][j]
                
    plt.imsave("final2.png",final)
    return final


# In[181]:


final = apply_transform(orig_image1, orig_image2)


# In[182]:


output = final[0:500, 0:1900]
plt.imshow(final,cmap = "gray")
cv2.imwrite("output16.jpg", final)

