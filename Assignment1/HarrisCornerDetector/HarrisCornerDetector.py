import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy import ndimage
import math
from scipy.misc import imsave
import sys

# Input : python HarrisCornerDetector.py img window_size std ksize_x ksize_y k tf

# img : Input image path
# window_size : dimension of gaussian kernel
# std : standard deviation of gaussian kernel
# ksize_x, ksize_y : dimensions of sobel operator
# k : Harris Detector free parameter
# tf : Thresholding factor

# Output : Image with corners detected using Harris Corner Detector


# Read the image
def readImage(img):
    img = cv2.imread(img, 0)
    return img

# Apply gaussian filter
def gaussianBlur(img, window_size, std):
    img = cv2.GaussianBlur(img,(window_size, window_size), std)
    return img

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

if __name__ == '__main__' :
    args = list(sys.argv)
    img = readImage(args[1])
    window_size = int(args[2])
    std = int(args[3])
    img = gaussianBlur(img, window_size, std)
    ksize_x = int(args[4])
    ksize_y = int(args[5])
    k = float(args[6])
    corners = getCornerList(img, ksize_x, ksize_y, window_size, std, k)
    tf = float(args[7])
    final_img = markCorners(img, corners, tf, window_size)
    imsave("output_img.jpg", final_img)
