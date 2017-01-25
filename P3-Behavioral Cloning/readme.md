# P3 : Behavioral Cloning

The goal of this project was to train a deep neural network in order to mimic the human driving so that the car
can drive autonomously on the simulator provided by Udacity.

## Dataset
Though the data can be collected by recording the data on the simulator provided by Udacity, I preferred to use the dataset provided by Udacity itself.
Udacity training set is constituted by 8036 samples. For each sample, two main information are provided:

* three frames from the frontal, left and right camera respectively
* the corresponding steering direction

## Data Augmentation
Because 8K samples were not enough to achieve a good performance, data augmentation techniques were applied in order
to get around 25K samples for training. The following augmentation techniques were used:

**Brightness augmentation** : New images were obtained from a single sample by changing the brightness in the HSV
color space.

**Resizing** : The images were resized to (66,220,3) as per the Nvidia's architecture convention.

**Color space** : The color space was changed to YUV from RGB as per the convention of Nvidia's architecture.

**Shift in steering angle** : A shift of 0.25 was added to the steering angel for left images and a shift of -0.25
was added to the steering angle for right images. This was done in order to make sure that the car drives in center of the path.

## Model architecture 
For the purpose of this project Nvidia's architecture was chosen and little modifications were done in order to overcome
overfitting issues. The final model looks like this:

```python
model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(img_rows, img_cols,ch)))

model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="valid", init='glorot_uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="valid", init='glorot_uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="valid", init='glorot_uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", init='glorot_uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", init='glorot_uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(100,init='glorot_uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(50,init='glorot_uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(10,init='glorot_uniform'))
model.add(Activation('relu'))
model.add(Dense(1))
model.compile(loss="mse", optimizer='adam')
```
