# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 22:18:22 2022

@author: sdeni
"""

import cv2
import numpy as np


img = cv2.imread("../input_images/Cells.jpg")


##plt.hist(img.flat, bins=100, range=(0,255))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


minDist = 10
param1 = 80 #500
param2 = 30 #200 #smaller value-> more false circles
minRadius = 4
maxRadius = 9

circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 3, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

count = 0

if circles is not None:
    circles = np.uint8(np.around(circles))
    for i in circles[0,:]:
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        count += 1


print(count)
cv2.imshow("original",img)


cv2.waitKey(0)
cv2.destroyAllWindows()