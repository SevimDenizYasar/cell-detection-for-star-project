import cv2
import matplotlib.pyplot as plt
import numpy as np

### GETTING INPUT IMAGES
img = cv2.imread("../cell_images/2.bmp")
img = cv2.GaussianBlur(img, (5,5), 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_final = img.copy()


### MASK THE OUTSIDE OF THE CHAMBER
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(rgb)
plt.show()
lower_rgb = np.array([195, 179, 139])
upper_rgb = np.array([255, 252, 234])
mask_rgb = cv2.inRange(rgb, lower_rgb, upper_rgb)

kernel1 = np.ones((5,5), np.uint8)
opening_rgb = cv2.morphologyEx(mask_rgb, cv2.MORPH_OPEN, kernel1, iterations = 2)

chamberContours, chamberHierarchy = cv2.findContours(image=opening_rgb, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
minArea = 5

for cnt in chamberContours:
    if cv2.contourArea(cnt) < minArea:
        rgb = cv2.bitwise_and(rgb, rgb, opening_rgb)
        indices = np.where(opening_rgb == 255)
        rgb[indices[0], indices[1], :] = [191, 191, 191]

img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

## PURPLE MASK TO CLEAN TRYPAN BLUE RESIDUES
lower_purple = np.array([57, 44, 46])
upper_purple = np.array([168, 164, 171])

mask_purple = cv2.inRange(img_bgr, lower_purple, upper_purple)

result_purple = cv2.bitwise_and(img_bgr, img_bgr, mask_purple)
indices = np.where(mask_purple==255)
img_bgr[indices[0], indices[1], :] = [191, 191, 191]

###### CELL DETECTION USING MASK
kernel = np.ones((3,3), np.uint8)
img_dilated = cv2.dilate(img_bgr, kernel, iterations=2)
plt.imshow(img_dilated)
plt.show()

lower_blue = np.array([160, 198, 208])
upper_blue = np.array([206, 248, 253])

mask_blue = cv2.inRange(img_dilated, lower_blue, upper_blue)
result = cv2.bitwise_and(img_bgr, img_bgr, mask_blue)

kernel2 = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel2, iterations = 2)

####### CELL COUNTING WITH CONTOURS
contours, hierarchy = cv2.findContours(image=opening, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

#cv2.drawContours(image=img_final, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
## filter by area
s1= 5
s2 = 200
xcnts = []
for cnt in contours:
    if s1<cv2.contourArea(cnt) <s2:
        xcnts.append(cnt)
        cv2.drawContours(image=img_final, contours=cnts, contourIdx=-1, color=(0, 255, 0), thickness=1,
                         lineType=cv2.LINE_AA)

print("Dots number: {}".format(len(xcnts)))

#cv2.imshow("image", opening)
cv2.imshow("or", img_final)
cv2.waitKey(0)
cv2.destroyAllWindows()
