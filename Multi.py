# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 19:03:30 2021

@author: abhir
"""


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.style.use('dark_background')

from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten
from keras.datasets import cifar10
from keras.utils import normalize,to_categorical


(X_train,y_train),(X_test,y_test)=cifar10.load_data()
X_train=normalize(X_train,axis=1)
X_test=normalize(X_test,axis=1)
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

train_datagen=ImageDataGenerator(rotation_range=45,
                                 width_shift_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True)
train_datagen.fit(X_train)

train_generator=train_datagen.flow(
    X_train,
    y_train,
    batch_size=32)
