'''
VGG 16 model for Jaffe emotion recognition

'''
__author__ = 'Kaustubh Sakhalkar'
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model
from keras.optimizers import SGD
import keras
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
from os import listdir
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

batch_size = 32
num_classes = 8
epochs = 200
data_augmentation = True

def find_label(temp):
	temp = temp[3:5]
	if temp == 'NE':
		return 1
	elif temp == 'HA':
		return 2
	elif temp == 'SA':
		return 3
	elif temp == 'SU':
		return 4
	elif temp == 'FE':
		return 5
	elif temp == 'DI':
		return 6
	elif temp == 'AN':
		return 7
	else:
		return 8

path_to_data = '/home/kaustubh/kaustubh_imp/deepLearning/datasets/jaffe'

files = listdir(path_to_data)
data = []
label = []
for i in files:
	img = cv2.imread(path_to_data+ "/" + i)
	data.append(img)
	label.append(find_label(i))

data = np.array(data)
label = np.array(label)

x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.33, stratify=label)
print x_train.shape, x_test.shape, y_train.shape, y_test.shape

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(8, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(x_train, y_train, batch_size=8, epochs=20)
score = model.evaluate(x_test, y_test, batch_size=8)
print score
  