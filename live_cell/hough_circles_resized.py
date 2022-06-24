# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 04:27:47 2022

@author: sdeni
"""
import cv2
import numpy as np


img = cv2.imread("../input_images/cells/cells_1.jfif")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#####blur/smoothing


scale_percent = 20 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
  
# resize image
resized = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)
blur = cv2.GaussianBlur(resized, (3,3), 0)
print('Resized Dimensions : ',resized.shape)

minDist = 6
param1 = 20 #500
param2 = 10 #200 #smaller value-> more false circles
minRadius = 0
maxRadius = 7

circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

count = 0

if circles is not None:
    circles = np.uint8(np.around(circles))
    for i in circles[0,:]:
        cv2.circle(resized, (i[0], i[1]), i[2], (0, 255, 0), 2)
        count += 1


print(count)


cv2.imshow("orb", blur)
cv2.imshow("rb", resized)
cv2.waitKeyEx(0)
cv2.destroyAllWindows()