# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 20:44:26 2018

@author: Luiz Carlos
"""
from PIL import Image
import cv2

#img = Image.open("00063_931230_fa.ppm").convert('LA')
#img.save("00063_931230_fa.png")

import os
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder,filename)).convert('LA')
        images.append(img)
    return images

imgset = load_images_from_folder(r"C:\Users\7\Desktop\faces\colorida")
i = 0
for img in imgset:
    img.save(str(i)+".png")
    i = i+1
    

testimg = cv2.imread(r"C:\Users\7\Desktop\Desktop\new shit\Grayscale\112.png")
xd =  cv2.cvtColor(testimg, cv2.COLOR_BGR2GRAY)
zulul = cv2.equalizeHist(xd)
cv2.imwrite('result2.jpg',zulul)
