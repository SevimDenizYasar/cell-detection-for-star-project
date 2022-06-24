# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 01:34:51 2022

@author: sdeni
"""

###### Dead cell detection 10 images

import cv2 
import numpy as np
import glob


path = "C:/Users/sdeni/OneDrive/Belgeler/cell_detection/input_images/cells/*.*" #give your folder path
img_number =1 
total_dead_cells=0

for file in glob.glob(path):
    
    a = cv2.imread(file)
    gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    
    ###### simple blob detector
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 0;
    params.maxThreshold = 110;

    detector = cv2.SimpleBlobDetector_create(params)


    keypoints = detector.detect(blur)
    
    im_with_keypoints = cv2.drawKeypoints(blur, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    print("number of dead cells on image", img_number,":",len(keypoints))
    total_dead_cells += len(keypoints)
    
    cv2.imshow("Keypoints", im_with_keypoints)
    cv2.waitKeyEx(500)
    cv2.destroyAllWindows()
    
    img_number +=1
    
print("total number of dead cells", total_dead_cells)
