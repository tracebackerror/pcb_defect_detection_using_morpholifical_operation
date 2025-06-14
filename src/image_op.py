# Python example program for image subtraction
import cv2
from PIL import Image

import numpy as np

# Paths of two image frames

image1Path = "./tests/pcb1_original.jpg";

image2Path = "./tests/pcb1_missingpinhole_defective.jpg";

# Open the images




# Get the image buffer as ndarray
image1 = cv2.imread(image1Path)
image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
#cv2.imshow("Original Image 1", image1)
#cv2.imshow("Original Image Gray 1", image1_gray)


image2 = cv2.imread(image2Path)
image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
#cv2.imshow("Original Image 2", image2)
#cv2.imshow("Original Image Gray 2", image2_gray)




for i in range(4, 8):
    dilated_im1 = cv2.dilate(image1_gray.copy(), None, iterations=i + 1)
    dilated_im2 = cv2.dilate(image2_gray.copy(), None, iterations=i + 1)

    #cv2.imshow("Dilated {} times".format(i + 1), dilated_im1)
    #cv2.imshow("Dilated Im 2 {} times".format(i + 1), dilated_im2)


    eroded_im1 = cv2.erode(dilated_im1.copy(), None, iterations=i + 5)
    eroded_im2 = cv2.erode(dilated_im2.copy(), None, iterations=i + 5)

    # concatenate image Horizontally
    Hori_im1 = np.concatenate((dilated_im1, eroded_im1), axis=1)
    Hori_im2 = np.concatenate((dilated_im2, eroded_im2), axis=1)



    buffer3 = eroded_im1 - eroded_im2
    # concatenate image Vertically
    Verti = np.concatenate((Hori_im1, Hori_im2), axis=0)

    # Display all the images including the difference image
    cv2.imshow("Image 1 and 2", Verti)
    cv2.imshow("Difference Image", buffer3)
    cv2.waitKey(0)
