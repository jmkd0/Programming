import cv2
import imutils
from imutils.contours import sort_contours
import numpy as np

## (1) read
image = cv2.imread("image3.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  

ret,thresh1 = cv2.threshold(gray, 0, 255,cv2.THRESH_OTSU|cv2.THRESH_BINARY_INV)
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)

contoursLines, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contoursLines = sorted(contoursLines, key=lambda contoursLines: cv2.boundingRect(contoursLines)[0] + cv2.boundingRect(contoursLines)[1] * image.shape[1] )
im2 = image.copy()
for cntLine in contoursLines:
    xLine, yLine, wLine, hLine = cv2.boundingRect(cntLine)
    cropLine = image[yLine:yLine+hLine, xLine:xLine+wLine]

    line = cropLine.copy()
    gray = cv2.cvtColor(cropLine, cv2.COLOR_BGR2GRAY)  #convert to gray
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)     #remove all noise
    limitEdge = cv2.Canny(blurred, 30, 150)         #limit the contours
    findEdge = cv2.findContours(limitEdge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #get all contours
    findEdge = imutils.grab_contours(findEdge)
    #findEdge = sort_contours(findEdge, method="left-to-right")[0]
    getChartRectangle = []
    for c in findEdge:
        (x, y, w, h) = cv2.boundingRect(c)
        if (w >= 5 and w <= 150) and (h >= 15 and h <= 120):
            crop = gray[y:y+h, x:x+w]   #crop the erea of the character
            crop = cv2.resize(crop, (32, 32))
            thresh = cv2.threshold(crop, 0, 225, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            #
            #if tW > tH:
            #    thresh = imutils.resize(thresh, width=32)
            #else:
            #    thresh = imutils.resize(thresh, height=32)
            print(thresh.shape)
            cv2.imshow("image", thresh)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.rectangle(line, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #cv2.rectangle(line, (xLine, yLine), (xLine + wLine, yLine + hLine), (0, 255, 0), 2)
    #cv2.imshow("image", line)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()





cv2.imshow("image", im2)
cv2.waitKey(0)
cv2.destroyAllWindows()