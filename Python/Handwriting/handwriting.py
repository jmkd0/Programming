import cv2
import numpy as np
#https://pysource.com/2018/08/26/knn-handwritten-digits-recognition-opencv-3-4-with-python-3-tutorial-36
#https://www.nist.gov/srd/nist-special-database-19

digits = cv2.imread("digits.png", cv2.IMREAD_GRAYSCALE)

cv2.imshow("digits", digits)
cv2.waitKey(0)
cv2.destroyAllWindows()