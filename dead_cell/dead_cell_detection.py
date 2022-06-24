# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 09:03:53 2022

@author: sdeni
"""

import cv2
import numpy as np

########## DEAD CELL DETECTION

#### reading/ gray sclae
img = cv2.imread("../input_images/cells/cells_1.jfif")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#####blur/smoothing
blur = cv2.GaussianBlur(gray, (3,3), 0)
ret, th =cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

params = cv2.SimpleBlobDetector_Params()

params.minThreshold = 0;

params.maxThreshold = 110;



detector = cv2.SimpleBlobDetector_create(params)


keypoints = detector.detect(blur)
im_with_keypoints = cv2.drawKeypoints(blur, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKeyEx(0)
cv2.destroyAllWindows()