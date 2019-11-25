import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
import numpy as np
import pandas as pd
from keras.models import load_model
import cv2
def loadweight():
    #cnn4=load_model()
    cnn4 = cnn4.load_model('myModel.h5')
    return cnn4
def datapreprocess(img):
    img = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
    img=cv2.resize(img, (28, 28))
    img = img.reshape((28, 28,1))

    img = img / 255.0
    img=np.expand_dims(img, 0)
    return img
def predict(img):
    img=datapreprocess(img)
    cnn4=loadweight()
    predictions = cnn4.predict_classes(img)
    # Define the text labels
    fashion_mnist_labels = ["T-shirt/top",  # index 0
                        "Trouser",      # index 1
                        "Pullover",     # index 2 
                        "Dress",        # index 3 
                        "Coat",         # index 4
                        "Sandal",       # index 5
                        "Shirt",        # index 6 
                        "Sneaker",      # index 7 
                        "Bag",          # index 8 
                        "Ankle boot"]   # index 9


    return fashion_mnist_labels[predictions]
predict('/home/system/test/tshirt.jpg')