import numpy as np
import cv2

def RANSAC(img1, img2):
    """
        Algorithm for robust feature matching between images
    """

    # Initialize SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # Find keypoints and descriptors in both images
    kp1, desc1 = sift.detectAndCompute(img1, None)
    kp2, desc2 = sift.detectAndCompute(img2, None)

    # Initialize brute-force matcher
    bf = cv2.BFMatcher(cv2.NORM_L2)

    # Match descriptors
    matches = bf.knnMatch(desc1, desc2, k=2)

    # Apply ratio test to select only good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Convert keypoints to numpy arrays
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    # Apply RANSAC algorithm to estimate homography matrix
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

    result = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    return result
