import cv2
import matplotlib.pyplot as plt
import numpy as np

### GETTING INPUT IMAGES
img = cv2.imread("../cell_images/2.bmp")
img = cv2.GaussianBlur(img, (5,5), 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kernel = np.ones((3, 3), np.uint8)



hpf = gray - cv2.GaussianBlur(gray, (3, 3), 3)+127

thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 199, 5)

thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 199, 5)
opening = cv2.morphologyEx(thresh2, cv2.MORPH_OPEN, kernel, iterations=2)

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#plt.imshow(rgb)
plt.show()

lower_rgb = np.array([195, 179, 139])
upper_rgb = np.array([255, 252, 234])
mask_rgb = cv2.inRange(rgb, lower_rgb, upper_rgb)
kernel1 = np.ones((5, 5), np.uint8)
opening_rgb = cv2.morphologyEx(mask_rgb, cv2.MORPH_OPEN, kernel1, iterations = 2)


contours,a = cv2.findContours(opening_rgb,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# finding bounding rectangle using contours data points
rect = cv2.boundingRect(contours[0])
pt1 = (rect[0],rect[1])
pt2 = (rect[0]+rect[2],rect[1]+rect[3])
cv2.rectangle(img,pt1,pt2,(100,100,100),thickness=2)

# extracting the rectangle
text = img[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]


cv2.imshow('Adaptive Mean', opening_rgb)


cv2.waitKey(0)
cv2.destroyAllWindows()