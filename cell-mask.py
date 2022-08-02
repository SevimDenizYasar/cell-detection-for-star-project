import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("../cell_images/12.bmp")
kernel = np.ones((3,3), np.uint8)
img_dilated = cv2.dilate(img, kernel, iterations=2)

plt.imshow(img_dilated)
plt.show()

lower = np.array([145, 201, 212])
upper = np.array([206, 248, 253])

mask = cv2.inRange(img_dilated, lower, upper)

result = cv2.bitwise_and(img, img, mask)

kernel2 = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel2, iterations = 3)

cnts = cv2.findContours(opening, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]


## filter by area
s1= 1
s2 = 200
xcnts = []
for cnt in cnts:
    if s1<cv2.contourArea(cnt) <s2:
        xcnts.append(cnt)

print("Dots number: {}".format(len(xcnts)))


cv2.imshow("image", opening)
cv2.imshow("or", result)

cv2.waitKey(0)
cv2.destroyAllWindows()

