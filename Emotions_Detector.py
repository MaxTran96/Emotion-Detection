import tensorflow as tf
import numpy as np
from PIL import Image
import glob
from keras.models import Sequential
from keras.layers import Reshape, Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras import backend as K
K.set_image_dim_ordering('th')
K.set_learning_phase(1)

def load_and_format_images(dir_name):
    output_images = []
    for image_name in glob.glob(dir_name + '/*'):
        image = Image.open(image_name).resize((32,32))
        output_image_data = np.asarray(image, dtype='float32')/255.0
        output_image_data = np.transpose(output_image_data, (2,0,1))
        if output_image_data.shape == (3,32,32):
            output_images.append(output_image_data)
    return np.asarray(output_images)

def separate_data(happy_images, sad_images):
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    for i in range(5):
        test_images.append(happy_images[i])
        test_labels.append([1,0])
        test_images.append(sad_images[i])
        test_labels.append([0,1])
    for i in range(len(happy_images)-5):
        train_images.append(happy_images[i+5])
        train_labels.append([1,0])
        train_images.append(sad_images[i+5])
        train_labels.append([0,1])
    return np.asarray(train_images), np.asarray(train_labels), np.asarray(test_images), np.asarray(test_labels)

def reshape_images(input_array):
    output_array = []
    for image in input_array:
        output_array.append(image.reshape(-1))
    return np.asarray(output_array)

def create_model():
    model = Sequential()
    model.add(Reshape(target_shape=(3,32,32), input_shape=(3072, )))
    model.add(Conv2D(32, (2,2),input_shape=(3,32,32), padding='same', activation='relu',
                     kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    return model




    


















