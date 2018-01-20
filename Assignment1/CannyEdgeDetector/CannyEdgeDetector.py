import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy import ndimage
import math
from scipy.misc import imsave
import sys

# Input : python CannyEdgeDetector.py img window_size_x window_size_y std ksize_x ksize_y minVal maxVal

# img : Input image path
# window_size_x, window_size_y : dimensions of gaussian kernel
# std : standard deviation of gaussian kernel
# ksize_x, ksize_y : dimensions of sobel operator
# minVal, maxVal : minimum and maximum values of threshold

# Output : Image with edges detected using Canny Edge Detector


# Read the image
def readImage(img):
	img = cv2.imread(img, 0)
	return img

# Apply gaussian filter
def gaussianBlur(img, window_size_x, window_size_y, std):
	img = cv2.GaussianBlur(img,(window_size_x, window_size_y), std)
	return img

# Applying the sobel operator to get first derivative gradient
def sobel(img, ksize_x, ksize_y):
	sobel_x = cv2.Sobel(img,cv2.CV_64F,1,0,ksize_x)
	sobel_y = cv2.Sobel(img,cv2.CV_64F,0,1,ksize_y)
	grad = np.hypot(sobel_x, sobel_y)
	theta = np.arctan2(sobel_y, sobel_x)
	return grad, theta

#Rounding off angles
def roundOffAngles(img, theta):
	theta_rounded = np.zeros(img.shape)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			angle = np.rad2deg(theta[i][j])%180
			if (0 <= angle < 22.5) and (157.5 <= angle <180):
				theta_rounded[i][j] = 0
			elif 22.5 <= angle < 67.5:
				theta_rounded[i][j] = 45
			elif 67.5 <= angle < 112.5:
				theta_rounded[i][j] = 90
			else:
				theta_rounded[i][j] = 135
	return theta_rounded    

#Non-maximum suppression
def nonMaxSuppression(img, grad, theta_rounded):
	max_img = np.zeros(img.shape)
	x, y = img.shape
	for i in range(x):
		for j in range(y):
			if (theta_rounded[i][j] == 0 and j>0 and j<y-1 and grad[i][j] >= grad[i][j-1] and grad[i][j] >= grad[i][j+1]):
				max_img[i][j] = grad[i][j]
			elif(theta_rounded[i][j] == 45 and j>0 and j<y-1 and i>0 and i<x-1 and grad[i][j] >= grad[i+1][j-1] and grad[i][j] >= grad[i-1][j+1]):
				max_img[i][j] = grad[i][j]
			elif(theta_rounded[i][j] == 90 and i>0 and i<x-1 and grad[i][j] >= grad[i+1][j] and grad[i][j] >= grad[i-1][j]):
				max_img[i][j] = grad[i][j]
			elif(theta_rounded[i][j] == 135 and j>0 and j<y-1 and i>0 and i<x-1 and grad[i][j] >= grad[i+1][j+1] and grad[i][j] >= grad[i-1][j-1]):
				max_img[i][j] = grad[i][j]
	return max_img

#Hysteresis thresholding
def hysteresisThresholding(max_img, minVal, maxVal):
	img_minVal = max_img > minVal
	img_maxVal = max_img > maxVal
	#label connected components in img_minVal
	labels_minVal, num_labels = ndimage.label(img_minVal)
	#Mark connected components which contain pixels from img_maxVal
	sums = ndimage.sum(img_maxVal, labels_minVal, np.arange(num_labels + 1))
	edges = sums > 0
	thresholded_img = edges[labels_minVal]
	return thresholded_img

#HSV coloring
def hsvColoring(img, thresholded_img, grad):
	x, y = img.shape
	hsv_img = np.zeros((x, y, 3), dtype = np.uint8)
	max_grad = np.max(grad)
	min_grad = np.min(grad)
	for i in range(x):
		for j in range(y):
			if(thresholded_img[i][j]):
				v = int(255*((grad[i][j] - min_grad)/(max_grad - min_grad)))
				hsv_img[i][j] = [theta_rounded[i][j]+100, 255, v]
	final_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
	return final_img

if __name__ == '__main__' :
	args = list(sys.argv)
	img = readImage(args[1])
	window_size_x = int(args[2])
	window_size_y = int(args[3])
	std = int(args[4])
	img = gaussianBlur(img, window_size_x, window_size_y, std)
	ksize_x = int(args[5])
	ksize_y = int(args[6])
	grad, theta = sobel(img, ksize_x, ksize_y)
	theta_rounded = roundOffAngles(img, theta)
	max_img = nonMaxSuppression(img, grad, theta_rounded)
	minVal = int(args[7])
	maxVal = int(args[8])
	thresholded_img = hysteresisThresholding(max_img, minVal, maxVal)
	final_img = hsvColoring(img, thresholded_img, grad)
	imsave("output_img.jpg", final_img)






