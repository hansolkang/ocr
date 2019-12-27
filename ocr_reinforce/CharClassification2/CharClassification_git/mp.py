import cv2  # for reading and writing or showing image
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import keras
import tensorflow as tf
from keras.preprocessing import image
from keras.models import load_model
# from net import Net
from resnet_build import Net
import sys
# from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D, BatchNormalization

def load_image(img_path, show=False):
    img = image.load_img(img_path, target_size=(32, 32))  # Path of test image
    # show the image if show=True
    if show:
        plt.imshow(img)
        plt.axis('off')

    # converting image to a tensor
    img_tensor = image.img_to_array(img)  # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.

    # return converted image
    return img_tensor


def predict(weights_path, image_path):
    model = Net.build(32, 32, 3, weights_path)

    image = load_image(image_path, show=False)  # load image, rescale to 0 to 1
    class_ = model.predict(image)  # predict the output, returns 36 length array
    # print("Detected: ", class_[0])  # print what is predicted
    output_indice = 0  # set it initially to -1
    global max
    # get class index having maximum predicted score
    for i in range(71):
        if (i == 0):
            max = class_[0][i]
            output_indice = 0
        else:
            if (class_[0][i] > max):
                max = class_[0][i]
                # print("asdfasdf", class_[0][i])
                output_indice = i

    # append 72 characters to list characters
    characters = []
    for i in range(35,38):
        characters.append((chr(i)))
    for i in range(40, 43):
        characters.append((chr(i)))
    characters.append(chr(45))
    for j in range(48, 48 + 10):
        characters.append(chr(j))
    characters.append(chr(58))
    characters.append(chr(64))
    for i in range(65, 65 + 26):
        characters.append(chr(i))
    for i in range(97, 97 + 26):
        characters.append(chr(i))

    # if output indice > 9 (means characters)
    # if (output_indice >= 20):
    final_result = characters[(output_indice)]
    print("Predicted: ", final_result)
    print("value: ", max)  # print predicted score
    return final_result

# test_image = "0.jpg"
#
# predict("check/renet_0509_fnt.h5", test_image)  # Specify weights file and Test image
