### The main script which will invoke other matching functions on the input images.

import cv2
import matplotlib.pyplot as plt
from brute_force import BruteForce
from bf_knn import BruteForceKNN
from flann import FLANN
from homography import Homography
from ransac import RANSAC
import argparse

#Constructing the argument parser and parsing the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-L', '--logo', required=True, help='Path to the logo or image you want to find')
ap.add_argument('-I', '--image', required=True, help='Path to the input image containing the logo/image')
ap.add_argument('-M', '--match', help='Type of matching algorithm to use', default='F')
args = vars(ap.parse_args())

#Getting the images ready
img1 = cv2.imread(args['logo'], cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(args['image'], cv2.IMREAD_GRAYSCALE)

match_type = args['match']
img_matches = None

if match_type == 'B':
    #Matching the images using the Brute-Force algorithm.
    img_matches = BruteForce(img1, img2)
elif match_type == 'R':
    #Matching the images using BruteForce KNN algorithm with Ratio Test.
    img_matches = BruteForceKNN(img1, img2)
elif match_type == 'F':
    img_matches = FLANN(img1, img2)
elif match_type == 'H':
    img_matches = Homography(img1, img2)
elif match_type == 'R':
    img_matches = RANSAC(img1, img2)

#Displaying the results and storing them
plt.imshow(img_matches)
plt.show()
cv2.imwrite('matched.png', img_matches)