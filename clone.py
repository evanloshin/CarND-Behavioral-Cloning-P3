from typing import List
import matplotlib
matplotlib.use('Agg')
import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

samples = []
with open('driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

def moving_average(angles, width):
    cumsum_vec = np.cumsum(np.insert(angles, 0, 0))
    mvavg_vec = (cumsum_vec[width:] - cumsum_vec[:-width]) / width
    return np.array(mvavg_vec)

# take a moving average of the angle measurement
center_angles = []
left_angles = []
right_angles = []
for sample in samples:
    center_angles.append(float(sample[3]))
    left_angles.append(float(sample[4]))
    right_angles.append(float(sample[5]))
n_period = 10
center_angles_mv = moving_average(center_angles, n_period)
left_angles_mv = moving_average(left_angles, n_period)
right_angles_mv = moving_average(right_angles, n_period)
del samples[-n_period+1:]

# for idx in range(len(samples)):
#     samples[idx][3] = center_angles_mv[idx]
#     samples[idx][4] = left_angles_mv[idx]
#     samples[idx][5] = right_angles_mv[idx]


for idx, sample in enumerate(samples):
    sample[3] = center_angles_mv[idx]
    sample[4] = left_angles_mv[idx]
    sample[5] = right_angles_mv[idx]

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=35):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for idx, batch_sample in enumerate(batch_samples):
                #if idx % 20 == 0:
                    center_path = './IMG/' + batch_sample[0].split('/')[-1]
                    left_path = './IMG/' + batch_sample[1].split('/')[-1]
                    right_path = './IMG/' + batch_sample[2].split('/')[-1]
                    center_image_BGR = cv2.imread(center_path)
                    left_image_BGR = cv2.imread(left_path)
                    right_image_BGR = cv2.imread(right_path)
                    center_image = cv2.cvtColor(center_image_BGR, cv2.COLOR_BGR2RGB)
                    left_image = cv2.cvtColor(left_image_BGR, cv2.COLOR_BGR2RGB)
                    right_image = cv2.cvtColor(right_image_BGR, cv2.COLOR_BGR2RGB)
                    center_angle = float(batch_sample[3])
                    left_angle = float(batch_sample[4])
                    right_angle = float(batch_sample[5])
                    center_image_aug = cv2.flip(center_image, 1)
                    center_angle_aug = center_angle * -1
                    images.append(center_image)
                    angles.append(center_angle)
                    images.append(center_image_aug)
                    angles.append(center_angle_aug)
                    # images.append(left_image)
                    # angles.append(left_angle)
                    # images.append(right_image)
                    # angles.append(right_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

from keras.models import Sequential
from keras.layers import Flatten, Activation, Cropping2D, Dense, Lambda, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

batch_size = 30
train_generator = generator(train_samples, batch_size)
validation_generator = generator(validation_samples, batch_size)

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu', name='first_convolution'))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu', name='second_convolution'))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu', name='third_convolution'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, \
                                     epochs=1, \
                                     steps_per_epoch=len(train_samples)/batch_size*2, \
                                     validation_data=validation_generator, \
                                     validation_steps=len(validation_samples)/batch_size*2)

### plot the training and validation loss for each epoch

# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.savefig('loss.png')

model.save('model.h5')