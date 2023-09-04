import cv2
import math
import numpy as np

path = "/home/amr/Desktop/redline5.jpg"
img = cv2.imread(path)
img = cv2.GaussianBlur(cv2.imread(path), (15, 15), cv2.BORDER_DEFAULT)
# img = cv2.bilateralFilter(cv2.imread(path), 9,75,75)


img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


lowerb1 = np.array([0, 75, 20])
upperb1 = np.array([10, 255, 255])

lowerb2 = np.array([160, 75, 20])
upperb2 = np.array([180, 255, 255])


lower_mask  = cv2.inRange(img_hsv, lowerb1, upperb1)
upper_mask  = cv2.inRange(img_hsv, lowerb2, upperb2)

mask = lower_mask + upper_mask
# mask = cv2.GaussianBlur(lower_mask + upper_mask, (5, 5), cv2.BORDER_DEFAULT)

moments = cv2.moments(mask)
cX = int(moments['m10'] / moments['m00'])
cY = int(moments['m01'] / moments['m00'])

print(f"cY is {cY}")
print(f"cX is {cX}")
cv2.arrowedLine(img_hsv, (int(img_hsv.shape[1]/2), int(img_hsv.shape[0])), (cX, cY), (255, 0, 0), 10)

print(f'Vector to be followed is:({cX - img_hsv.shape[1]/2}, {cY - img_hsv.shape[0]})')