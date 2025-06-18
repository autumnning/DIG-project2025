import cv2
import numpy as np

img = cv2.imread('Q2_image/test.png', cv2.IMREAD_GRAYSCALE)
height, width = img.shape

# turn to binary image
for i in range(height):
    for j in range(width):
        if img[i][j] > 128:
            img[i][j] = 1
        else:
            img[i][j] = 0

# Erosion
se = np.ones((3, 3), dtype=np.uint8)
img_erosion = img.copy()
for i in range(1, height - 1):
    for j in range(1, width - 1):
        if np.all(img[i - 1:i + 2, j - 1:j + 2] == 1):
            img_erosion[i][j] = 1
        else:
            img_erosion[i][j] = 0
img = img - img_erosion

# recovery to 0~255
img = img * 255
cv2.imwrite('Q2_image/output_2.png', img)
