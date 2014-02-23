import numpy as np
import cv2
from math import sqrt, pow, acos
sample = 5

def rotate(imgl, key1, key2, matches):
    x1, y1 = key1[matches[1].queryIdx].pt
    x2, y2 = key1[matches[2].queryIdx].pt
    x3, y3 = key2[matches[1].trainIdx].pt
    x4, y4 = key2[matches[2].trainIdx].pt

    dotproduct = ((x2-x1)*(x4-x3)) + ((y2-y1)*(y4-y3))
    dist_l = sqrt(pow((x2-x1),2) + pow((y2-y1),2))
    dist_r = sqrt(pow((x4-x3),2) + pow((y4-y3),2))
    cos_theta = dotproduct/(dist_l*dist_r)
    theta = acos(cos_theta)
    theta = (theta/3.14159265359)*180
    
    rowsl,colsl = imgl.shape
    M = cv2.getRotationMatrix2D((colsl/2, rowsl/2), theta, 1)
    dst = cv2.warpAffine(imgl, M, (colsl,rowsl))
    return dst, theta

def analyze(key1, key2, matches, scale): 
    totalx, totaly = 0, 0
    for i in range(sample):
        x1, y1 = key1[matches[i].queryIdx].pt
        x2, y2 = key2[matches[i].trainIdx].pt
        totalx += x2-x1*scale[0]
        totaly += y2-y1*scale[1]
    return totalx/sample, totaly/sample

def scale(key1, key2, matches, amount):
    totalx = 0
    totaly = 0
    for i in range(sample):
        x1, y1 = key1[matches[i].queryIdx].pt
        x2, y2 = key1[matches[i+1].queryIdx].pt
        x3, y3 = key2[matches[i].trainIdx].pt
        x4, y4 = key2[matches[i+1].trainIdx].pt
        if abs(x4-x3)>1:
            totalx += abs(x2-x1)/abs(x4-x3)
        if abs(y4-y3)>1:
            totaly += abs(y2-y1)/abs(y4-y3)
    print totalx/sample, totaly/sample
    return totalx/sample, totaly/sample

def json():
    myfile="manfest.json"
    data = open(myfile,'r')
    f=json.load(data)

    #head wrapper
    t=0
    fatdump={}
    scrs={}
    for i in f["nodes"]:
        ips=f["nodes"][t]['ip']
        imgs=f["nodes"][t]["image"]
        fatdump[ips]=t
        fatdump[imgs]=t
        #score wrapper
        h=0
        for j in f["nodes"]:
            scrs[t,h]=score[t,h]
            #scrs[t,h]=t+h
            fatdump["scr"]=scrs
            h+=1
        t+=1

    print fatdump

def score(first, second):
    alpha = cv2.imread(first["path"], 0)
    beta = cv2.imread(second["path"], 0)
    
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
    
    alpha, angle = rotate(alpha, kp1, kp2, matches)

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
    
    imgscale = scale(kp1, kp2, matches, 1)
    return analyze(kp1, kp2, matches, imgscale)
