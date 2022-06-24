# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 05:51:56 2022

@author: sdeni
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import img_as_float

img = cv2.imread("../input_images/cells/cells_1.jfif")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("clahse", gray)
median = cv2.medianBlur(gray, 15)
clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
res = clahe.apply(gray)

sigma = 0.3
median = np.median(img)

lower = int(max(0,(1.0-sigma)*median))
upper = int(min(255, (1.0 +sigma)*median))

canny = cv2.Canny(img, lower, upper)
ret, canny_b = cv2.threshold(canny, 0, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("canny",canny_b)

minDist = 300
param1 = 250 #500
param2 = 10 #200 #smaller value-> more false circles
minRadius = 0
maxRadius = 20

circles = cv2.HoughCircles(canny_b, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

count = 0

if circles is not None:
    circles = np.uint8(np.around(circles))
    for i in circles[0,:]:
        cv2.circle(canny_b, (i[0], i[1]), i[2], (0, 255, 0), 2)
        count += 1


print(count)
cv2.imshow("clahe", res)
#cv2.imshow("circled",canny_b)
cv2.waitKey(0)
cv2.destroyAllWindows()
