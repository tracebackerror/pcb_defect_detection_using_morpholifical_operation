from skimage.metrics import structural_similarity
import cv2
import numpy as np
from matplotlib import pyplot as plt

rows = 2
columns = 2

# image1Path = "./tests/pcb1_original.jpg";
#
# image2Path = "./tests/pcb1_missingpinhole_defective.jpg";

def pcb_defect_detection_algorithm(image1Path=None, image2Path=None):
        # Load images
        before = cv2.imdecode(np.frombuffer(image1Path.read(), np.uint8), -1)
        after = cv2.imdecode(np.frombuffer(image2Path.read(), np.uint8), -1)
        # before = cv2.imread("input1.jpg")
        # after = cv2.imread("input2.jpg")

        # Convert images to grayscale
        before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
        after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

        # Compute SSIM between the two images
        (score, diff) = structural_similarity(before_gray, after_gray, full=True)
        print("Image Similarity: {:.4f}%".format(score * 100))

        # The diff image contains the actual image differences between the two images
        # and is represented as a floating point data type in the range [0,1]
        # so we must convert the array to 8-bit unsigned integers in the range
        # [0,255] before we can use it with OpenCV
        diff = (diff * 255).astype("uint8")
        diff_box = cv2.merge([diff, diff, diff])

        # Threshold the difference image, followed by finding contours to
        # obtain the regions of the two input images that differ
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        mask = np.zeros(before.shape, dtype='uint8')
        filled_after = after.copy()

        for c in contours:
            area = cv2.contourArea(c)
            if area > 40:
                x,y,w,h = cv2.boundingRect(c)
                cv2.rectangle(before, (x, y), (x + w, y + h), (36,255,12), 2)
                cv2.rectangle(after, (x, y), (x + w, y + h), (36,255,12), 2)
                cv2.rectangle(diff_box, (x, y), (x + w, y + h), (36,255,12), 2)
                cv2.drawContours(mask, [c], 0, (255,255,255), -1)
                cv2.drawContours(filled_after, [c], 0, (0,255,0), -1)
        #import pdb; pdb.set_trace()
        cv2.imwrite("./static/first_image.jpg", before)
        cv2.imwrite("./static/second_image.jpg", after)
        cv2.imwrite("static/diff.jpg", diff)
        cv2.imwrite("static/diff_box.jpg", diff_box)
        cv2.imwrite("static/mask.jpg", mask)
        cv2.imwrite("static/filled_after.jpg", filled_after)


#cv2.waitKey()