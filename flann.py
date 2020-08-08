### We will use the SIFT feature detector and then using those features from the logo,
### match the logo with another picture using the FLANN algorithm.
import cv2

def FLANN(img1, img2):
    """ 
    Matches features from an image to another using the FLANN algorithm.
    
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

    #Preparing a mask to draw good matches
    mask = [ [0, 0] for i in range(len(matches)) ]

    #Fill the mask based on Lowe's ratio test
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            mask[i] = [1, 0]

    #Draw the matches that passed the ratio test
    img_matches = cv2.drawMatchesKnn(
        img1, keypts1, img2, keypts2, matches, None, matchColor = (255, 0, 0), singlePointColor=(0, 0, 255), matchesMask = mask, flags = 0)
    
    return img_matches
