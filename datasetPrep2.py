# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 20:44:26 2018

@author: Luiz Carlos
"""
#from PIL import Image
import cv2
import numpy as np
#img = Image.open("00063_931230_fa.ppm").convert('LA')
#img.save("00063_931230_fa.png")

import os
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename), cv2.IMGREAD_GRAYSCALE)
        images.append(img)
    return images

imgset = load_images_from_folder(r"C:\Users\7\Desktop\Desktop\faces\colorida")
i = 0
for img in imgset:
    cv2.imwrite("C://Users//7//Desktop//Desktop//new shit//take2//"+str(i)+".ppm")
    i = i+1
    
imgset = np.array(imgset)
testimg = Image.open(r"C:\Users\7\Desktop\Desktop\new shit\Grayscale\213.png")