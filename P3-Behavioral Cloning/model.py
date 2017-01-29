import os
import cv2
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mimg
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Convolution2D, Dense, Flatten, Lambda, ELU, Dropout, MaxPooling2D, Activation
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf
tf.python.control_flow_ops = tf

##Read the data
driving_log = pd.read_csv('./data/driving_log.csv')
image_path = './data/'
np.random.seed(111)

##Variables to be used in processing pipeline
driving_log = shuffle(driving_log, random_state = 111)
train_data, val_data = train_test_split(driving_log, test_size=0.2, random_state=111)
img_rows, img_cols, ch = 66, 200,3    ## As per Nvidia's convention for thier model
augmentation_min_value = 0.2
augmentation_max_value = 1.25
shift = 0.25


##Brightness augmentation
def augment_brightness(image):
    brightness = 0.25 + np.random.uniform()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image[:,:,2] = image[:,:,2]*brightness
    ##Clip the image so that no pixel has value greater than 255
    image[:,:,2] = np.clip(image[:,:,2], a_min=0, a_max=255)
    return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)


##Cropping of images and changing of color space
def roi(image):
    image = image[64:295, :]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    return cv2.resize(image, (200,66), interpolation=cv2.INTER_AREA)



## Define the steps here for how the pipeline is going to be exceuted
def preprocess_trainData(idx, augment=True, bias=0.8):
    steering_angle = driving_log.iloc[idx].steering
    j = np.random.randint(3) ##Choose a ranodm number 
    if j == 0:
        image = cv2.imread(os.path.join(image_path, driving_log.iloc[idx].left.strip())) ##Left image
        steering_angle = steering_angle + shift
    elif j == 1:    
        image = cv2.imread(os.path.join(image_path, driving_log.iloc[idx].center.strip())) ##Center image
    else:
        image = cv2.imread(os.path.join(image_path, driving_log.iloc[idx].right.strip()))  ##Right image
        steering_angle = steering_angle - shift
        
    ##Chaneg from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    ##Apply brightness augmentation
    image = augment_brightness(image)
    
    ##Apply flipping whenever required as there are too many left turns
    flip = np.random.randint(2)
    if flip == 0:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
        
    ##Apply roi
    image = roi(image)
    return image, steering_angle    


##Preprocessing for validation data
def preprocess_valid_data(row_data):
    image = cv2.imread(os.path.join(image_path, row_data['center'].strip()))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = roi(image)
    return image   
    

## Define custom trainImageDataGenerator
def train_image_data_generator(data, batch_size=32):
    batch_images = np.zeros((batch_size, img_rows, img_cols, ch), dtype=np.float32)
    batch_steering_angles = np.zeros((batch_size,),dtype=np.float32)
    
    while 1:
        i = 0
        while i < batch_size:
            idx = np.random.randint(data.shape[0])
            img, str_angle = preprocess_trainData(idx)
            batch_images[i] = img
            batch_steering_angles[i] = str_angle
            i = i + 1
        
        yield batch_images, batch_steering_angles



##Validation image generator
def valid_image_data_generator(data):
    while 1:
        for i in range(len(data)):
            row_data = data.iloc[i]
            img = preprocess_valid_data(row_data)
            img = np.array(img.reshape(1, img.shape[0], img.shape[1], img.shape[2]),dtype=np.float32)
            steer = np.array([row_data['steering']],dtype=np.float32)
            yield img, steer                   


##Generators
train_data_gen = train_image_data_generator(train_data)
val_data_gen = valid_image_data_generator(val_data)


##Build Nvidia's architecture with modifications
model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(img_rows, img_cols,ch)))

model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="valid", init="he_normal"))
model.add(Activation('relu'))

model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="valid", init="he_normal"))
model.add(Activation('relu'))

model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="valid", init="he_normal"))
model.add(Activation('relu'))

model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", init="he_normal"))
model.add(Activation('relu'))

model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", init="he_normal"))
model.add(Activation('relu'))

model.add(Flatten())

#model.add(Activation('relu'))

model.add(Dense(100,init="he_normal"))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(50,init="he_normal"))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(10,init="he_normal"))
model.add(Activation('relu'))
model.add(Dense(1))
model.compile(loss="mse", optimizer='adam')

#model.summary()

##Checkpoint the model during training
#filepath="weights.best.hdf5"
#checkpoint = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, mode='min')
#early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')


##Train the model
model.fit_generator(train_data_gen, samples_per_epoch=23296,validation_data= val_data_gen, 
                                             nb_val_samples=1000, nb_epoch=10)

##Save the model weights and json file
model_json = model.to_json()
model_name = 'model'
with open(model_name+'.json', "w") as json_file:
    json_file.write(model_json)
model.save_weights(model_name+'.h5')

