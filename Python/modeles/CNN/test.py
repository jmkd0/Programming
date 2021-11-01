import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2

def load_dataset():
    #path = "/home/komlan/Datasets/Character_hand/Hnd/Img"
    path = "/home/komlan/Datasets/myData"
    dir_chars = os.listdir(path)
    dir_chars.sort()
    labels = [chr(i) if i>=10 else i for i in list(range(10))+list(range(65,91))+list(range(97,123))]

    data_images = []
    for label, dir_label in enumerate(dir_chars):
        imgs_name_list = os.listdir(path+"/"+dir_label)
        for img_name in imgs_name_list:
            current_image = cv2.imread(path+"/"+dir_label+"/"+img_name)
            current_image = cv2.resize(current_image, (32,32))
            current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
            data_images.append([current_image, label])
    return np.array(data_images)
data_images = load_dataset()
data_images = pd.DataFrame(data_images, columns=["images", "labels"])
data_images.shape
data_images = load_dataset()
data_images = pd.DataFrame(data_images, columns=["images", "labels"])
data_images.shape
X = data_images["images"].values
y = data_images["labels"].values
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.8, test_size=0.2, random_state=42)