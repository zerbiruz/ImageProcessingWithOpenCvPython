import numpy as np 
import cv2 as cv 


def nothing(x):
    pass

# create black image, a window
img = np.zeros((300, 512, 3), np.uint8)
cv.namedWindow("image")

# create trackbar for color change
cv.createTrackbar('R', "image", 0, 255, nothing)
cv.createTrackbar('G', "image", 0, 255, nothing)
cv.createTrackbar('B', "image", 0, 255, nothing)

# create switch for ON/OFF functionality
switch = "0 : OFF \n 1 : ON"
cv.createTrackbar(switch, "image", 0, 1, nothing)

while True:
    cv.imshow("image", img)

    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break

    # get current position of four trackbar
    r = cv.getTrackbarPos('R', "image")
    g = cv.getTrackbarPos('G', "image")
    b = cv.getTrackbarPos('B', "image")
    s = cv.getTrackbarPos(switch, "image")

    if s == 0:
        img[:] = 0
    else:
        img[:] = [b, g, r]

cv.destroyAllWindows()
        