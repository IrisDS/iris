import numpy as np
import cv2

def rotate(img, direction):
    orientation = {"NORTH" : 0, "EAST" : -90, "SOUTH" : 180, "WEST" : 90}
    rows,cols = img.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), orientation[direction], 1)
    dst = cv2.warpAffine(img, M, (cols,rows))
    return dst

left = cv2.imread("img/left.jpg", 0)
right = cv2.imread("img/right.jpg", 0)
right = rotate(right, "SOUTH")
# Initiate the SIFT detector
orb= cv2.ORB()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(left, None)
kp2, des2 = orb.detectAndCompute(right, None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match Descriptors
matches = bf.match(des1, des2)

#Sort them in order of distance
matches = sorted(matches, key = lambda x:x.distance)

#draw first 10 matches
img3 = None
img3 = cv2.drawMatches(left, kp1, right, kp2, matches[:10], outImg=img3, flags=2)

for match in matches:
    x, y = kp1[match.trainIdx].pt
    cv2.circle(left,(int(x), int(y)), 2, 3)

cv2.imwrite( "Test.jpg", img3)
cv2.imwrite( "left.jpg", left)
