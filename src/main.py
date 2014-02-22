import numpy as np
import cv2
from matplotlib import pyplot as plt

left = cv2.imread("img/left.jpg", 0)
right = cv2.imread("img/right.jpg", 0)

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
img3 = cv2.drawMatches(left, kp1, right, kp2, matches[:10], flags=2)

plt.imshow(img3),plt.show()
