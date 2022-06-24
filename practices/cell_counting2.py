# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 23:40:20 2022

@author: sdeni
"""
import cv2
import numpy as np 

img = cv2.imread("../input_images/Cells.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

cv2.imshow("original1",gray)

minDist = 10
param1 = 20 #500
param2 = 17 #200 #smaller value-> more false circles
minRadius = 4
maxRadius = 8

circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 3, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

count = 0

if circles is not None:
    circles = np.uint8(np.around(circles))
    for i in circles[0,:]:
        cv2.circle(gray, (i[0], i[1]), i[2], (0, 255, 0), 2)
        count += 1


print(count)
cv2.imshow("circled",gray)

cv2.waitKey(0)
cv2.destroyAllWindows()