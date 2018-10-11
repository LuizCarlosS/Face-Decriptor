# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 14:10:20 2018

@author: luido
"""

import cv2
import os
import numpy as np

img = cv2.imread(r"C:\Users\luido\OneDrive\Documentos\1 PAIC\Grayscale\296.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_equal = cv2.equalizeHist(img_gray)
#cv2.imwrite(r"C:\Users\luido\OneDrive\Documentos\1 PAIC\Ijamens\result.ppm", img_equal)
cv2.imwrite(r"C:\Users\luido\OneDrive\Documentos\1 PAIC\Ijamens\result296.png", img_equal)

test = cv2.imread(r"C:\Users\luido\OneDrive\Documentos\1 PAIC\Ijamens\311.png")
