# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 20:44:26 2018

@author: Luiz Carlos
"""
#from PIL import Image
import cv2
import numpy as np
import pandas as pd
#img = Image.open("00063_931230_fa.ppm").convert('LA')
#img.save("00063_931230_fa.png")

import os
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        r = 100.0 / img.shape[1]
        dim = (100, int(img.shape[0] * r))
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        images.append(resized)
    return images

imgset = load_images_from_folder(r"C:\Users\7\Desktop\Desktop\Face Decriptor\batman")
img = imgset[0]
for img in imgset:
  img.resize((150, 100), refcheck = False)

imgset_array = np.array(imgset)

#dados categoricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_y = LabelEncoder()
oneHotEncoder = OneHotEncoder(categorical_features = [0])
y = pd.read_csv(r'C:\Users\7\Desktop\Desktop\Face Decriptor\label.csv')
y = labelEncoder_y.fit_transform(y)
y = y.reshape(-1, 1)
y = oneHotEncoder.fit_transform(y).toarray()

#começo da rede neural
import keras
from keras.models import Sequential,Input,Model,load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

#transformar cada imagem em uma matriz 28x28x1
X = imgset_array.reshape(-1, 150, 100, 1)
#converter de uint8 para float32
X = X.astype('float32')
#valores dos pixels para valores entre 0 e 1 inclusive.
X = X / 255.
#split
from sklearn.model_selection import train_test_split
train_X,valid_X,train_label,valid_label = train_test_split(X, y, test_size=0.2, random_state=13)

batch_size = 64
epochs = 20
num_classes = 5

#modelo da rede neural
ethn_model = Sequential()
ethn_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear', input_shape=(150,100,1),padding='same'))
ethn_model.add(LeakyReLU(alpha=0.1))
ethn_model.add(MaxPooling2D((2, 2),padding='same'))
ethn_model.add(Dropout(0.25))

ethn_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
ethn_model.add(LeakyReLU(alpha=0.1))
ethn_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
ethn_model.add(Dropout(0.25))

ethn_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
ethn_model.add(LeakyReLU(alpha=0.1))                  
ethn_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
ethn_model.add(Dropout(0.4))
ethn_model.add(Flatten())

ethn_model.add(Dense(128, activation='linear'))
ethn_model.add(LeakyReLU(alpha=0.1))
ethn_model.add(Dropout(0.3))                  
ethn_model.add(Dense(output_dim = num_classes, activation='softmax'))

ethn_model.summary()

#otimizador adam
ethn_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

fashion_train = ethn_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))

ethn_model.save('ethn_model_dropout.h5py')

model = load_model(r'C:\Users\7\Desktop\Desktop\Face Decriptor\ethn_model_dropout.h5py')
teste = cv2.imread(r'C:\Users\7\Desktop\Desktop\Face Decriptor\yo.jpg')
teste = cv2.cvtColor(teste, cv2.COLOR_BGR2GRAY)
reTeste = cv2.resize(teste, (100, 150), interpolation = cv2.INTER_AREA)
reTeste.resize((150,100), refcheck = False)

#label_teste = y[5]
#label_teste = label_teste.reshape(-1, 1)
bacate = reTeste.reshape(-1, 150, 100, 1)
teste_eval = ethn_model.predict(bacate)


