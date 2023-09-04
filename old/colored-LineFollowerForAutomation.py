import cv2
import math
import numpy as np

path = r"test_images/red4.webp"
img = cv2.imread(path)
print(f"Image dimensions: {img.shape}")

cv2.imshow("img", img)
cv2.waitKey(0)
img = cv2.bilateralFilter(cv2.imread(path), 9, 75, 75)
img = cv2.GaussianBlur(cv2.imread(path), (15, 15), cv2.BORDER_DEFAULT)

# Converting to HSV color space for better red-color segmentation
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Lower and upper HSV bounds for red (at both ends of the spectrum)
lowerb1 = np.array([0, 75, 20])
upperb1 = np.array([10, 255, 255])

lowerb2 = np.array([160, 75, 20])
upperb2 = np.array([180, 255, 255])


# Masking the path to be followed
lower_mask = cv2.inRange(img_hsv, lowerb1, upperb1)
upper_mask = cv2.inRange(img_hsv, lowerb2, upperb2)

mask = lower_mask + upper_mask


img_edges = cv2.Canny(mask, 50, 200, None, 3)


lines = cv2.HoughLines(img_edges, 1, np.pi / 180, 150, None, 0, 0)
# lines = cv2.HoughLinesP(img_edges, 1, np.pi / 180, 50, None, 50, 10)

# if it is not none condition is missing
rho = lines[0][0][0]
theta = lines[0][0][1]

theta_line = 90 - math.degrees(theta)
theta_norm_line = math.atan(-1 / math.tan(math.radians(theta_line)))  # in radians
thickness = abs(
    np.count_nonzero(mask[len(mask) - 15]) / math.cos(math.pi - theta_norm_line)
)

# Distance from center calculations
a = math.cos(theta)
b = math.sin(theta)
x0 = a * rho
y0 = b * rho
pt1 = np.array((int(x0 + 1000 * (-b)), int(y0 + 1000 * (a))))
pt2 = np.array((int(x0 - 1000 * (-b)), int(y0 - 1000 * (a))))
center = (img.shape[0] / 2, img.shape[1] / 2)
distanceFromOrigin = np.cross(pt2 - pt1, pt1 - center) / np.linalg.norm(pt1 - pt2)


print(f"The number of hough lines is: {len(lines)}")
print(
    f"First Hough Line is: rho({lines[0][0][0]}), theta({90-math.degrees(lines[0][0][1])})"
)
print(f"Thicnkess (in pixels): {thickness}")
print(f"Distance from center (in pixels): {distanceFromOrigin} ")


cv2.waitKey(0)
cv2.destroyAllWindows()
