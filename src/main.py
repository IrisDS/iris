import numpy as np
import cv2
from math import sqrt, pow

def rotate(img, direction):
    orientation = {"NORTH" : 0, "EAST" : -90, "SOUTH" : 180, "WEST" : 90}
    rows,cols = img.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), orientation[direction], 1)
    dst = cv2.warpAffine(img, M, (cols,rows))
    return dst

def analyze(key1, key2, matches, scale): 
    scorex = 0
    scorey = 0
    for match in matches:
        x1, y1 = key1[match.queryIdx].pt
        x2, y2 = key2[match.trainIdx].pt
        print (x1, y1), (x2*scale[0], y2*scale[1])
        if x1>x2*scale[0]:
            scorex += 1
        else:
            scorex -= 1
        if y1>y2*scale[1]:
            scorey += 1
        else:
            scorey -= 1
    return scorex, scorey

def scale(key1, key2, matches, amount):
    totalx = 0.0
    totaly = 0.0
    for i in range(amount):
        x1, y1 = key1[matches[i].queryIdx].pt
        x2, y2 = key1[matches[i+1].queryIdx].pt
        x3, y3 = key2[matches[i].trainIdx].pt
        x4, y4 = key2[matches[i+1].trainIdx].pt
        totalx += (x2-x1)/(x4-x3)
        totaly += (y2-y1)/(y4-y3)
    return abs(totalx/amount), abs(totaly/amount)

def score(first, second):
    alpha = cv2.imread(first["path"], 0)
    alpha = rotate(alpha, first["direction"])
    beta = cv2.imread(second["path"], 0)
    beta = rotate(alpha, second["direction"])
    
    # Initiate the SIFT detector
    orb= cv2.ORB()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(alpha, None)
    kp2, des2 = orb.detectAndCompute(beta, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match Descriptors
    matches = bf.match(des1, des2)

    #Sort them in order of distance
    matches = sorted(matches, key = lambda x:x.distance)
    
    scale = scale(kp1, kp2, matches, 1)
    return analyze(kp1, kp2, matches, scale)
