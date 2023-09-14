import numpy as np 
import pandas as pd
import random as rd
import os
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from keras.metrics import accuracy
from keras.applications.mobilenet import MobileNet
from keras.applications.vgg16 import VGG16
# setting seed for reproducability
from numpy.random import seed
seed(25)
tf.random.set_seed(50)
from google.colab import drive
drive.mount('/content/drive/')
	
# 0 - Normal
# 1 - Tumor
	
data = [] #creating a list for images
paths = [] #creating a list for paths
labels = [] #creating a list to put our 0 or 1 labels

# staring with the images that have tumors
for r, d, f in os.walk(r'/content/drive/MyDrive/akkj/yes'):
    for file in f:
        if '.jpg' in file:
            paths.append(os.path.join(r, file))
for path in paths:
    img = Image.open(path)
    img = img.resize((128,128))
    img = np.array(img)
    if(img.shape == (128,128,3)):
        data.append(np.array(img))
        labels.append(1)
        #now working with the images with no tumors        
paths = []
for r, d, f in os.walk(r"/content/drive/MyDrive/akkj/no"):
    for file in f:
        if '.jpg' in file:
            paths.append(os.path.join(r, file))
for path in paths:
    img = Image.open(path)
    img = img.resize((128,128))
    img = np.array(img)
    if(img.shape == (128,128,3)):
        data.append(np.array(img))
        labels.append(0)
        data = np.array(data)
data.shape
labels = np.array(labels)
labels = labels.reshape(2891,1)
print('data shape is:', data.shape)
print('labels shape is:', labels.shape)

print(data.shape)

# getting the max of the array
print(np.max(data))
# getting the min of the array
print(np.min(data))
# reducing the data to between 1 and 0
data = data / 255.00
# getting the max of the array
print(np.max(data))
# getting the min of the array
print(np.min(data))

for i in range(50):
    fig = plt.figure(figsize=(50,50))
    plt.subplot(50,50,i+1)
    image = plt.imshow(data[i])
    plt.show(image)
	
x_train,x_test,y_train,y_test = train_test_split(data, labels, test_size=0.3, shuffle=True, random_state=7)

print("shape of our training data:",x_train.shape)
print("shape of our training labels:",y_train.shape)
print("shape of our test data:",x_test.shape)
print("shape of our test labels:",y_test.shape)

# model_generating(MobileNet)
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Flatten,Dense
from keras.models import Model, load_model
from keras.applications.mobilenet import MobileNet
import keras

base_model=VGG16(input_shape=(128,128,3),include_top=False,pooling='max',weights='imagenet')

for layer in base_model.layers:
  layer.trainable=False

base_model.summary()

X=Flatten()(base_model.output)
X=Dense(units=256,activation='relu')(X)
X=Dense(units=1,activation='sigmoid')(X)
model=Model(base_model.input,X)

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(epsilon=0.01),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# including early stopping to revent overfitting
early_stopping = keras.callbacks.EarlyStopping(
    patience=40,
    min_delta=0.001,
    restore_best_weights=True,
)

history = model.fit(
    x = x_train,
    y = y_train,
    validation_data= (x_test,y_test),
    batch_size = 64,
    epochs=200,
    callbacks=[early_stopping],
    verbose=(2),
)

history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss','val_loss']].plot()
history_frame.loc[:, ['accuracy','val_accuracy']].plot();

pred = model.predict(x_test)

def names(number):
    if number==0:
        return 'Its a Tumor'
    else:
        return 'No, Its not a tumor'

from matplotlib.pyplot import imshow
img = Image.open(r"/content/drive/MyDrive/akkj/yes/y1032.jpg")
x = np.array(img.resize((128,128)))
x = x.reshape(1,128,128,3)
res = model.predict_on_batch(x)
classification = np.where(res == np.amax(res))[1][0]
imshow(img)
print(str(res[0][classification]*100) + '% Confidence This Is ' + names(classification))

for i in range(len(pred)):
    if pred[i] > 0.5:
        pred[i] = 1
    else:
        pred[i] = 0

pred = pred.astype(int)
# creating a classification report
classification_report(y_test, pred)
