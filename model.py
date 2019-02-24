"""
Neural network for steering angle prediction.
The model is based on NVIDIA's "End to End Learning for Self-Driving Cars" paper.
Source:  https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
"""

from keras.models import Sequential
from keras.activations import relu
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, MaxPooling2D, LeakyReLU, PReLU
from keras.layers.convolutional import Conv2D
import numpy as np
import data_load

np.random.seed(42)  # for reproducibility

n_epochs = 8
n_samples_per_epoch = 20032
n_valid_samples = 6400
learning_rate = 1e-4


def get_model():
    ch, row, col = 3, 80, 320  # Trimmed image format

    model = Sequential()
    # Preprocess incoming data, centered around zero with small standard deviation
    model.add(Lambda(lambda x: x / 127.5 - 1.,
                     input_shape=(ch, row, col),
                     output_shape=(ch, row, col)))

    # Five convolutional and maxpooling layers
    model.add(Conv2D(24, 5, 5, border_mode='same', activation='relu', subsample=(2, 2)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Activation(ELU))

    model.add(Conv2D(36, 5, 5, border_mode='same', activation='relu', subsample=(2, 2)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Conv2D(48, 5, 5, border_mode='same', activation='relu', subsample=(2, 2)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu', subsample=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu', subsample=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Flatten())

    # Next, five fully connected layers
    model.add(Dense(1164))
    model.add()

    model.add(Dense(100))
    model.add(activation)

    model.add(Dense(50))
    model.add(activation)

    model.add(Dense(10))
    model.add(activation)

    model.add(Dense(1))

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
    # model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator,
    #                     nb_val_samples=len(validation_samples), nb_epoch=3)

#     model.save_weights("./outputs/steering_model/steering_angle.keras", True)
#     with open('./outputs/steering_model/steering_angle.json', 'w') as outfile:
#         json.dump(model.to_json(), outfile)
