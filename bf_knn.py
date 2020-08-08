### We will use the ORB feature detector and then using those features from the logo,
### Brute-Force-KNN match the logo with another picture.
import cv2

def BruteForceKNN(img1, img2):
    """ 
    Matches features from an image to another using the brute-force-knn algorithm.
    
    """

    # Extract features from both images to be matched using ORB
    orb = cv2.ORB_create()
    keypts1, desc1 = orb.detectAndCompute(img1, None)
    keypts2, desc2 = orb.detectAndCompute(img2, None)

    #Perform brute-force-knn matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = False)
    match_pairs = bf.knnMatch(desc1, desc2, k=2)

    #Apply the ratio test (threshold = 0.8)
    matches = [x[0] for x in match_pairs
    if len(x) > 1 and x[0].distance < 0.8*x[1].distance]
    
    #Draw the 25 best matches
    img_matches = cv2.drawMatches(
        img1, keypts1, img2, keypts2, matches[:25], img2, flags = cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    
    return img_matches
