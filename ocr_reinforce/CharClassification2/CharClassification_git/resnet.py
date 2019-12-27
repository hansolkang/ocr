# import cv2
# import time
# import tensorflow as tf
# import matplotlib.pyplot as plt
# tf.logging.set_verbosity(tf.logging.ERROR)
# from keras.applications.resnet50 import ResNet50, decode_predictions
# resnet = ResNet50()
#
# img = cv2.imread('rafd.jpg', -1)
# img = cv2.resize(img, (224, 224))
#
# start = time.time()
# yhat = resnet.predict(img.reshape(-1, 224, 224, 3))
# time = time.time() - start
# label = decode_predictions(yhat)
# label = label[0][0]
#
# print("Test time {}".format(time))
# print('%s (%.2f%%)' % (label[1], label[2]*100))
# img = img[:,:,::-1]
# plt.figure(figsize=(11,11))
# plt.imshow(img)
# plt.axis("off")
# plt.show()
import numpy as np
import pandas as pd

import os
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D, BatchNormalization
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array

resnet_weights_path = './check/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

data_generator = ImageDataGenerator(horizontal_flip=True,
                                   width_shift_range = 0.4,
                                   height_shift_range = 0.4,
                                   zoom_range=0.3,
                                   rotation_range=20,
                                   )

image_size = 224
batch_size = 10
train_generator = data_generator.flow_from_directory(
        './train/English/Fnt/',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical')

num_classes = len(train_generator.class_indices)

model = Sequential()

model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(2048, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(num_classes, activation='softmax'))

model.layers[0].trainable = True
checkpoint = ModelCheckpoint(filepath='./check/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
            monitor='loss',
            mode='min',
            save_best_only=True, verbose=1)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

count = sum([len(files) for r, d, files in os.walk("./train/English/Fnt/")])

model.fit_generator(
        train_generator,
        steps_per_epoch=int(count/batch_size) + 1,
        epochs=50)
model.save_weights('./check/renet_0509_fnt.h5')
from IPython.display import Image, display

import os, random
img_locations = []
for d in os.listdir("./train/English/Fnt/"):
    directory = "./train/English/Fnt/" + d
    sample = [directory + '/' + s for s in random.sample(
        os.listdir(directory), int(random.random()*10))]
    img_locations += sample
def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size):
    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]
    img_array = np.array([img_to_array(img) for img in imgs])
    return preprocess_input(img_array)

random.shuffle(img_locations)
imgs = read_and_prep_images(img_locations)
predictions = model.predict_classes(imgs)
classes = dict((v,k) for k,v in train_generator.class_indices.items())

for img, prediction in zip(img_locations, predictions):
    display(Image(img))
    print(classes[prediction])