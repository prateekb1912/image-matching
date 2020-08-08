### We will perform homography with FLANN-based matches on two images

import cv2
import numpy as np

def Homography(img1, img2):
    """ 
    Performs homography between two images by making use of the FLANN- based matching.
    """

    # Extract features from both images to be matched using SIFT algorithm
    orb = cv2.xfeatures2d.SIFT_create()
    keypts1, desc1 = orb.detectAndCompute(img1, None)
    keypts2, desc2 = orb.detectAndCompute(img2, None)

    #Define FLANN-based matching parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    #Perform FLANN-based matching
    flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann_matcher.knnMatch(desc1, desc2, k=2)
    
    #Find all good matches as per Lowe's ratio test
    good_matches = []
    for m,n in matches:
        if m.distance < 0.7 * n.distance :
            good_matches.append(m)
    
    MIN_GOOD_MATCHES = 10

    if len(good_matches) >= MIN_GOOD_MATCHES:
        src_pts = np.float32(
            [keypts1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [keypts2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        #Find the homography
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        mask_matches = mask.ravel().tolist()

        #Perform perspective transform
        h, w = img1.shape
        src_corners = np.float32(
            [[0, 0], [0, h-1], [w-1, h-1], [w-1, 0] ]).reshape(-1, 1, 2)
        dst_corners = cv2.perspectiveTransform(src_corners, M)
        dst_corners = dst_corners.astype(np.int32)

        cv2.polylines(img2, [dst_corners], True, 0, 3, cv2.LINE_AA)
    
    else:
        print("Not enough matches found - {}/{}".format(len(good_matches), MIN_GOOD_MATCHES))

    img_matches = cv2.drawMatches(img1, keypts1, img2, keypts2, good_matches, None, 
    matchColor=(255, 0, 0), singlePointColor = (0, 0, 0), flags=0)
    return img_matches