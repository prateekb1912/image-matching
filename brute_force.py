### We will use the ORB feature detector and then using those features from the logo,
### brute-force match the logo with another picture.
import cv2

def BruteForce(img1, img2):
    """ 
    Matches features from an image to another using the brute-force algorithm.
    
    """

    # Extract features from both images to be matched using ORB
    orb = cv2.ORB_create()
    keypts1, desc1 = orb.detectAndCompute(img1, None)
    keypts2, desc2 = orb.detectAndCompute(img2, None)

    #Perform brute-force matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    matches = bf.match(desc1, desc2)

    #Sort the matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    #Draw the 25 best matches
    img_matches = cv2.drawMatches(
        img1, keypts1, img2, keypts2, matches[:25], img2, flags = cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    
    return img_matches
