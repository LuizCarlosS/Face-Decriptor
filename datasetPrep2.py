# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 20:44:26 2018

@author: Luiz Carlos
"""
#from PIL import Image
import cv2
import numpy as np
import pandas as pd
from keras.utils import to_categorical
#img = Image.open("00063_931230_fa.ppm").convert('LA')
#img.save("00063_931230_fa.png")

import os
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        images.append(img)
    return images

imgset = load_images_from_folder(r"C:\Users\luido\OneDrive\Documentos\GitHub\Face-Decriptor\batman")
img = imgset[0]
for img in imgset:
  img.resize((768, 512), refcheck = False)

imgset_array = np.array(imgset)

#dados categoricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_y = LabelEncoder()
oneHotEncoder = OneHotEncoder(categorical_features = [0])
y = pd.read_csv('OUT.csv')
y = labelEncoder_y.fit_transform(y)
y = y.reshape(-1, 1)
y = oneHotEncoder.fit_transform(y).toarray()
#i = 0
#for img in imgset:
    #cv2.imwrite("C://Users//luido//Desktop//batman//"+str(i)+".ppm", img)
    #i = i+1
    
#imgset = np.array(imgset)
#testimg = Image.open(r"C:\Users\7\Desktop\Desktop\new shit\Grayscale\213.png")
