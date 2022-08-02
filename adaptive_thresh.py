# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 12:05:12 2022

@author: deno
"""

import cv2
import numpy as np

img = cv2.imread("../cell_images/11.bmp")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 1)

kernel = np.ones((5, 5), np.uint8)
img_dilation = cv2.dilate(img, kernel, iterations=1)
gray_dilated = cv2.cvtColor(img_dilation, cv2.COLOR_BGR2GRAY)

hpf = gray - cv2.GaussianBlur(gray_dilated, (3, 3), 3)+127

ret, thresh = cv2.threshold(hpf, 115, 255, cv2.THRESH_BINARY)
clahe = cv2.createCLAHE(clipLimit =5)
CLAHE_img = clahe.apply(hpf)

blur2 = cv2.GaussianBlur(CLAHE_img, (5,5), 2)
ret, thresh2 = cv2.threshold(blur2, 100, 255, cv2.THRESH_BINARY)

kernel2 = np.ones((3,3), np.uint8)
closing = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel2)

params = cv2.SimpleBlobDetector_Params()
##### filter by convexity
params.filterByConvexity = True
params.minConvexity = 0.1
params.maxConvexity = 1
##### filter by area
params.filterByArea = True
params.minArea = 160

params.minThreshold = 0
params.maxThreshold = 110

params.filterByColor = True
params.blobColor = 0


detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(closing)
blank = np.zeros((1, 1))
blobs = cv2.drawKeypoints(img, keypoints, blank, (0, 0, 255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)



cv2.imshow("keypoints", blobs)


cv2.waitKey(0)
cv2.destroyAllWindows()