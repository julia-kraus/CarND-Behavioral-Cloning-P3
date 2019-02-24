"""
Neural network for steering angle prediction. Implementation of Nvidia research paper.
"""
import os
from keras.models import Sequential
from keras.activations import relu
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, MaxPooling2D, LeakyReLU, PReLU
from keras.layers.convolutional import Conv2D
import numpy as np
import data_load

np.random.seed(42)  # for reproducibility


# def gen(hwm, host, port):
#   for tup in client_generator(hwm=hwm, host=host, port=port):
#     X, Y, _ = tup
#     Y = Y[:, -1]
#     if X.shape[1] == 1:  # no temporal context
#       X = X[:, -1]
#     yield X, Y

def get_model(activation=ELU):
    ch, row, col = 3, 80, 320  # Trimmed image format

    model = Sequential()
    # Preprocess incoming data, centered around zero with small standard deviation
    model.add(Lambda(lambda x: x / 127.5 - 1.,
                     input_shape=(ch, row, col),
                     output_shape=(ch, row, col)))
    model.add(Conv2D(16, 8, 8, border_mode="same"))
    model.add(MaxPooling2D((4, 4), padding='same'))
    model.add(activation())
    model.add(Conv2D(32, 5, 5, border_mode="same"))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(activation())
    model.add(Conv2D(64, 5, 5, border_mode="same"))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(activation())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
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
    model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data = validation_generator,
                        nb_val_samples = len(validation_samples), nb_epoch = 3)

#     )
#     print("Saving model weights and configuration file.")
#
#     if not os.path.exists("./outputs/steering_model"):
#         os.makedirs("./outputs/steering_model")
#
#     model.save_weights("./outputs/steering_model/steering_angle.keras", True)
#     with open('./outputs/steering_model/steering_angle.json', 'w') as outfile:
#         json.dump(model.to_json(), outfile)
