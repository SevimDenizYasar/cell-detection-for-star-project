# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 04:03:02 2022

@author: sdeni
"""

import cv2
import numpy as np

img = cv2.imread("../input_images/cells/cells_1.jfif")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#####blur/smoothing
blur = cv2.GaussianBlur(gray, (3,3), 0)
ret, th =cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)

orb = cv2.ORB_create(200)
kp = orb.detect(blur, None)

kp, des = orb.compute(blur, kp)

img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)

cv2.imshow("orb", img2)
cv2.waitKeyEx(0)
cv2.destroyAllWindows()