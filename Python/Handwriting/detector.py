import cv2 #conda install -c conda-forge opencv
import io
import os

# Imports the Google Cloud client library
from google.cloud import vision #pip install --upgrade google-cloud-vision

# Instantiates a client
client = vision.ImageAnnotatorClient()

# The name of the image file to annotate
file_name = os.path.abspath('resources/wakeupcat.jpg')

# Loads the image into memory
with io.open(file_name, 'rb') as image_file:
    content = image_file.read()

image = vision.Image(content=content)

# Performs label detection on the image file
response = client.label_detection(image=image)
labels = response.label_annotations

print('Labels:')
for label in labels:
    print(label.description)

#import io
#import cv2 #conda install -c conda-forge opencv
#import pytesseract #conda install -c conda-forge pytesseract
#import tesserocr #conda install -c conda-forge tesserocr
#from PIL import Image
#
#
#
#image = cv2.imread('image4.png')
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#_, gray = cv2.threshold(gray, 10, 255,cv2.THRESH_BINARY)
#
#img = Image.fromarray(gray)
#txt = pytesseract.image_to_string(img)
#print(txt)
#
#with tesserocr.PyTessBaseAPI() as api:
#    #req_image = cv2.imread('bienvenue.bmp')
#    image = Image.open('image1.bmp')
#    api.SetImage(image)
#    api.Recognize()  # required to get result from the next line
#    iterator = api.GetIterator()
#    print (iterator.WordFontAttributes())
