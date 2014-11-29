import numpy as np
import cv2
from matplotlib import pyplot as plt

def filter_matches(kp1, kp2, matches, ratio = 0.75):
    """
    Keep only matches that have distance ratio to 
    second closest point less than 'ratio'.
    """
    mkp1, mkp2 = [], []
    for m in matches:
        if m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append( kp1[m.queryIdx] )
            mkp2.append( kp2[m.trainIdx] )
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, kp_pairs

def find_Fmatrix(matches):
    print(matches[0][0].pt[0])
    f1,f2,f3,f4,f5,f6,f7,f8,f9 = [],[],[],[],[],[],[],[],[]
    f = matches[0][0].pt[0]

    for i in range(9): f1.append(matches[i*2][0].pt[0]*matches[i*2][1].pt[0])   # xx'
    for i in range(9): f2.append(matches[i*2][0].pt[0]*matches[i*2][1].pt[1])   # xy'
    for i in range(9): f3.append(matches[i*2][0].pt[0])                       # x
    for i in range(9): f4.append(matches[i*2][0].pt[1]*matches[i*2][1].pt[0])   # yx'
    for i in range(9): f5.append(matches[i*2][0].pt[1]*matches[i*2][1].pt[1])   # yy'
    for i in range(9): f6.append(matches[i*2][0].pt[1])                       # y
    for i in range(9): f7.append(matches[i*2][1].pt[0])                       # x'
    for i in range(9): f8.append(matches[i*2][1].pt[1])                       # y'
    for i in range(9): f9.append(1.0)                                       # 1s

    a = np.array([f1,f2,f3,f4,f5,f6,f7,f8,f9])
    b = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    x = np.linalg.solve(a, b)
    print(f1)
    print(f2)
    print(f3)
    print(f4)
    print(f5)
    print(f6)
    print(f7)
    print(f8)
    print(f9)
    print(a)
    print(b)
    print(x)


# read in two images in grayscale
img1 = cv2.imread('dino/dino0001.png',0)
img2 = cv2.imread('dino/dino0002.png',0)


# Initiate SIFT detector
sift = cv2.SIFT()

# find the keypoints and descriptors with SIFT
print "Detecting features in images."
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)


# Create new images with keypoints displayed
print "Creating keypoint images."
img3=cv2.drawKeypoints(img1,kp1,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img4=cv2.drawKeypoints(img2,kp2,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# BFMatcher with default params
print "Finding matches."
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)
print "Found {} matches.".format(len(matches))

# Apply ratio test
print "Finding good matches."
p1, p2, kp_pairs = filter_matches(kp1, kp2, matches)

find_Fmatrix(kp_pairs)

# Display images with keypoints to help understand OpenCV SIFT operations
cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
cv2.imshow('image1',img3)

cv2.namedWindow('image2', cv2.WINDOW_NORMAL)
cv2.imshow('image2',img4)

cv2.waitKey(0)
cv2.destroyAllWindows()
