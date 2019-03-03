"""
Neural network for steering angle prediction.
The model is based on NVIDIA's "End to End Learning for Self-Driving Cars" paper.
Source:  https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
"""

from keras.models import Sequential
from keras.activations import relu
from keras.layers import Activation
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, MaxPooling2D, LeakyReLU, PReLU
from keras.layers.convolutional import Conv2D
import numpy as np
import utils
import data_load

np.random.seed(42)  # for reproducibility

n_epochs = 8
n_samples_per_epoch = 20032
n_valid_samples = 6400
learning_rate = 1e-4


def get_model():
    row, col, ch = 66, 200, 3  # Trimmed image format 80 320

    model = Sequential()
    # Preprocess incoming data, centered around zero with small standard deviation
    model.add(Lambda(lambda x: x / 127.5 - 1.,
                     input_shape=(row, col, ch),
                     output_shape=(row, col, ch)))

    # discussion: strided convolutions or max pooling layers
    # try different activations

    # Five convolutional and maxpooling layers
    model.add(Conv2D(24, (5, 5), padding='valid', activation='relu', strides=(2, 2)))  # subsample=(2, 2)
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Conv2D(36, (5, 5), padding='valid', activation='relu', strides=(2, 2)))  # subsample =(2, 2)
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Conv2D(48, (5, 5), padding='valid', activation='relu', strides=(2, 2)))  # subsample =(2, 2)
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    #
    model.add(Conv2D(64, (3, 3), padding='valid', activation='relu'))
    #
    model.add(Conv2D(64, (3, 3), padding='valid', activation='relu'))
    #
    model.add(Flatten())
    #
    # Next, five fully connected layers
    model.add(Dense(1164, activation='relu'))  # adapt this if you have a different input size. the rest should be fine

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
    #     parser = argparse.ArgumentParser(description='Training neural network for steering angle prediction')
    #     parser.add_argument('--batch', type=int, default=64, help='Batch size.')
    #     parser.add_argument('--epoch', type=int, default=200, help='Number of epochs.')
    #     parser.add_argument('--epochsize', type=int, default=10000, help='How many frames per epoch.')
    #     parser.add_argument('--skipvalidate', dest='skipvalidate', action='store_true', help='Multiple path output.')
    #     parser.set_defaults(skipvalidate=False)
    #     parser.set_defaults(loadweights=False)
    #     args = parser.parse_args()
    #

    # does fit_generator also have shuffle, validation_split parameters?

    model = get_model()
    model.summary()
    # create two generators for training and validation
    train_gen = helper.generate_next_batch()
    validation_gen = helper.generate_next_batch()

    history = model.fit_generator(train_gen,
                                  samples_per_epoch=n_samples_per_epoch,
                                  nb_epoch=n_epochs,
                                  validation_data=validation_gen,
                                  nb_val_samples=n_valid_samples,
                                  verbose=1)

    # finally save our model and weights
    utils.save_model(model)
    # save weights
#   model.save_weights("./outputs/steering_model/steering_angle.keras", True)
#   with open('./outputs/steering_model/steering_angle.json', 'w') as outfile:
#       json.dump(model.to_json(), outfile)
