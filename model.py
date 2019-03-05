"""
Neural network for steering angle prediction.
The model is based on NVIDIA's "End to End Learning for Self-Driving Cars" paper.
Source:  https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, MaxPooling2D, LeakyReLU, PReLU, Cropping2D, Reshape
from keras.layers.convolutional import Conv2D
import numpy as np
import utils

np.random.seed(42)  # for reproducibility

n_epochs = 8
n_samples_per_epoch = 20032
n_valid_samples = 6400
learning_rate = 1e-4
batch_size = 64


def get_model():
    row, col, ch = 128, 128, 3
    model = Sequential()
    # Normalize incoming data, centered around zero with small standard deviation
    model.add(Lambda(lambda x: x / 127.5 - 1.,
                     input_shape=(row, col, ch),
                     output_shape=(row, col, ch)))

    # discussion: strided convolutions or max pooling layers
    # try different activations

    # Five convolutional and maxpooling layers
    model.add(Conv2D(24, (5, 5), padding='valid', activation='relu', strides=(2, 2)))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Conv2D(36, (5, 5), padding='valid', activation='relu', strides=(2, 2)))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Conv2D(48, (5, 5), padding='valid', activation='relu', strides=(2, 2)))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    #
    model.add(Conv2D(64, (3, 3), padding='valid', activation='relu'))
    #
    model.add(Conv2D(64, (3, 3), padding='valid', activation='relu'))
    #
    model.add(Flatten())

    # Next, five fully connected layers for uncropped images: 27456
    # for cropped images: 6336 neurons, Nvidia: 1164
    model.add(Dense(5184, activation='relu'))  # adapt this if you have a different input size. the rest should be fine
    #
    model.add(Dense(100, activation='relu'))
    #
    model.add(Dense(50, activation='relu'))
    #
    model.add(Dense(10, activation='relu'))
    #
    model.add(Dense(1, activation='relu'))

    model.compile(optimizer="adam", loss="mse")

    return model


if __name__ == "__main__":
    model = get_model()
    model.summary()

    # # create two generators for training and validation
    # train_gen = utils.get_next_batch(batch_size)
    # validation_gen = utils.get_next_batch(batch_size)
    #
    # model.fit_generator(train_gen,
    #                     steps_per_epoch=n_samples_per_epoch // batch_size,
    #                     epochs=n_epochs,
    #                     validation_data=validation_gen,
    #                     validation_steps=n_valid_samples // batch_size,
    #                     verbose=1)
    #
    # # save model
    # model.save('model.h5')
