import os
import numpy as np
import cv2   # pip install opencv-python
import pytesseract
from imutils.contours import sort_contours
from imutils.object_detection import non_max_suppression
import imutils #pip install imutils         #helps to operate on edge contours in image
from sklearn.model_selection import train_test_split # pip install -U scikit-learn            
#import tensorflow as tf   # pip install tensorflow==2.2.0 tensorflow-gpu==2.2.0
#from tensorflow.keras.layers import Conv2D
#from tensorflow.keras import Model
#from tensorflow import keras   # pip install Keras
#import matplotlib.pyplot as plt # pip install -U matplotlib
################################
#Download Pb file for EAST
#https://github.com/ZER-0-NE/EAST-Detector-for-text-detection-using-OpenCV 
#################################
#https://pysource.com/2018/08/26/knn-handwritten-digits-recognition-opencv-3-4-with-python-3-tutorial-36/
class Handwriting:

    allImageLetters = []        
    labels = []   
    allWordInInputImage = []

    def __init__(self, size):
        self.size = size
        self.imageInput = "test2.jpeg"#  "receipt.png"
        self.pathDigits = "/home/komlan/datas/handwrite/hsf_0/digit/"
        self.pathLowLetter = "/home/komlan/datas/handwrite/hsf_0/lower"
        self.pathUppLetter = "/home/komlan/datas/handwrite/hsf_0/upper"
        self.eastFile = "/home/komlan/anaconda3/envs/anaconda/lib/frozen_east_text_detection.pb"
        self.sizeWidth = 640 #Must be a multiple of 32
        self.sizeHeight = 640
        self.padding = 0.05
        self.rateFind = 0.5

    def manager(self):
        #Gharge Input image and process to get all word erea
        self.preprocessInputImage()

        for i in range(len(self.allWordInInputImage)):
            wordImage = self.allWordInInputImage[i][0]
            letterImage = self.getLettersImage(wordImage)
            #print("HAAA")
            #break
        #print(self.allWordInInputImage[0][1])

        #self.getListofImages(self.pathDigits)
        #self.getListofImages(self.pathLowLetter)
        #self.getListofImages(self.pathUppLetter)
        ##convert to numpy array
        #self.allImageLetters = np.array(self.allImageLetters)
        #self.labels = np.array(self.labels)
        #print(self.allImageLetters.shape)
        #print(self.labels.shape)

    def preprocessInputImage(self):
        image = cv2.imread(self.imageInput)
        orig = image.copy()
        imageShow  = orig.copy()
        (height, width) = image.shape[:2]
        #Get the rate to modify bound after
        rateHeight = height / float(self.sizeHeight)
        rateWidth = width / float(self.sizeWidth)
        image = cv2.resize(image, (self.sizeWidth, self.sizeHeight))
        
        #Prediction of text ereas
        layerNames = ["feature_fusion/Conv_7/Sigmoid","feature_fusion/concat_3"]
        net = cv2.dnn.readNet(self.eastFile)
        blob = cv2.dnn.blobFromImage(image, 1.0, (self.sizeWidth, self.sizeHeight), (123.68, 116.78, 103.94), swapRB=True, crop=False) #contruct blob from the image
        net.setInput(blob)
        (scores, geometry) = net.forward(layerNames)
        (rectangles, confidences) = self.decodePrediction(scores, geometry)
        boxes = non_max_suppression(np.array(rectangles), probs=confidences)
        
        rects = []
        for (startX, startY, endX, endY) in boxes:
            startX = int(startX * rateWidth)
            startY = int(startY * rateHeight)
            endX = int(endX * rateWidth)
            endY = int(endY * rateHeight)

            dX = int((endX - startX) * self.padding)
            dY = int((endY - startY) * self.padding)

            startX = max(0, startX - dX)
            startY = max(0, startY - dY)
            endX = min(width, endX + (dX * 2))
            endY = min(height, endY + (dY * 2))

            rects.append([startX, endX, startY, endY])
        
        rectsSort = self.sortRectangle(rects)
        print(rectsSort)
        for rect in rectsSort:
            roi = orig[rect[2]:rect[3], rect[0]:rect[1]]
            self.allWordInInputImage.append([roi, [rect[0], rect[1], rect[2], rect[3]]])

            cv2.rectangle(imageShow, (rect[0], rect[2]), (rect[1], rect[3]), (0, 255, 0), 2)
            #roi = orig[startY:endY, startX:endX]
            #self.allWordInInputImage.append([roi, [startX, endX, startY, endY]])
            
        cv2.imshow("image", imageShow)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def sortRectangle(self, rects):
        lenRec = len(rects)
        #Sort Y values
        for i in range(1,lenRec):
            k = i-1 
            while k >= 0 and rects[k][2] > rects[k+1][2]:
                t = rects[k+1]
                rects[k+1] = rects[k]
                rects[k] = t 
                k -= 1
        #Get the range of the same Y
        i = 0
        while i < lenRec:
            init = i
            end = i
            x = 1
            if i+x < lenRec and rects[i][2] == rects[i+x][2]:
                x += 1
                while i+x < lenRec and rects[i][2] == rects[i+x][2]:
                    x += 1
                end += x-1
                i += x
            else:
                i += 1
            #If there is no repetition for Y value continue
            if end - init == 0:
                continue
            
            #sort X values
            for j in range(init + 1, end+1):
                k = j-1 
                while k >= init and rects[k][0] > rects[k+1][0]:
                    t = rects[k+1]
                    rects[k+1] = rects[k]
                    rects[k] = t 
                    k -= 1
        return rects
        
    def decodePrediction(self, scores, geometry):
        (numbRows, numbCols) = scores.shape[2:4]
        rectangles = []
        confidences = []
        for j in range(numbRows):
            #get datas of the boxs in those lines
            scoreData = scores[0,0,j]
            xData0 = geometry[0,0,j]
            xData1 = geometry[0,1,j]
            xData2 = geometry[0,2,j]
            xData3 = geometry[0,3,j]
            angleData = geometry[0,4,j]
            for i in range(numbCols):
                if scoreData[i] < self.rateFind:
                    continue
                (offsetX, offsetY) = (i*4.0, j*4.0)

                angle = angleData[i]
                cos = np.cos(angle)
                sin = np.sin(angle)

                h = xData0[i] + xData2[i]
                w = xData1[i] + xData3[i]

                endX = int(offsetX + (cos * xData1[i]) + (sin * xData2[i]))
                endY = int(offsetY - (sin * xData1[i]) + (cos * xData2[i]))

                startX = int(endX - w)
                startY = int(endY - h)

                rectangles.append((startX, startY, endX, endY))
                
                confidences.append(scoreData[i])
            
        return (rectangles, confidences)






    def getLettersImage(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  #convert to gray
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)     #remove all noise
        limitEdge = cv2.Canny(blurred, 30, 150)         #limit the contours
        findEdge = cv2.findContours(limitEdge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #get all contours
        findEdge = imutils.grab_contours(findEdge)
        findEdge = sort_contours(findEdge, method="left-to-right")[0]

        getChartRectangle = []
        for c in findEdge:
            (x, y, w, h) = cv2.boundingRect(c)
            if (w >= 5 and w <= 150) and (h >= 15 and h <= 120):
                crop = gray[y:y+h, x:x+w]   #crop the erea of the character
                thresh = cv2.threshold(crop, 0, 225, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                tH, tW = thresh.shape
                if tW > tH:
                    thresh = imutils.resize(thresh, width=32)
                else:
                    thresh = imutils.resize(thresh, height=32)
                #cv2.imshow("image", thresh)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
                #cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow("image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()




        #cv2.imshow("image", findEdge)


        


    

    def getListofImages(self, repository):
        listDir = os.listdir(repository)
        for d in listDir:
            paths = os.listdir(repository+"/"+str(d))
            for x in paths:
                currentImg = cv2.imread(repository+"/"+str(d)+"/"+x)
                currentImg = cv2.resize(currentImg, (self.size,self.size))
                self.allImageLetters.append(currentImg)
                self.labels.append(d)


hand = Handwriting(32)
#hand.preprocessInputImage()
#hand.getLettersImage()
hand.manager()
