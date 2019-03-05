import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import errno
from scipy.ndimage import rotate
from keras.preprocessing.image import ImageDataGenerator

DRIVING_LOG_FILE = './data/driving_log.csv'
STEERING_PLUS_MINUS = 0.229
IMG_FOLDER = './data/'


def remove_existing_file(filename):
    """removes existing files of the name filename"""
    try:
        os.remove(filename)
    except OSError as e:  # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT:  # errno.ENOENT = no such file or directory
            raise  # re-raise exception if a different error occurred


def save_model(model, model_filename='model.json', weights_filename='weights.h5'):
    """
    Save the model onto disk
    :param model:
        Model to be saved
    :param model_filename:
        Name of file where model is stored
    :param weights_filename:
        Name of file where weights are stored
    :return:
        None
    """

    # remove existing files of the same name
    remove_existing_file(model_filename)
    remove_existing_file(weights_filename)

    json_string = model.to_json()
    with open(model_filename, 'w') as outfile:
        json.dump(json_string, outfile)

    model.save_weights(weights_filename)


def get_next_img_file(batch_size=64):
    """Extracts driving log file to image-label pairs in batches"""
    data = pd.read_csv(DRIVING_LOG_FILE)
    # pick random images for batch
    batch_indices = np.random.randint(0, len(data), batch_size)

    images = []
    angles = []
    for index in batch_indices:
        # pick either left, right or center image
        image_position = np.random.randint(0, 3)
        if image_position == 0:
            # if image is left image, add the STEERING_PLUS_MINUS coefficient to the steering angle
            img = data.iloc[index]['left'].strip()
            angle = data.iloc[index]['steering'] + STEERING_PLUS_MINUS
            images.append(img)
            angles.append(angle)

        elif image_position == 1:
            # if image is central image, leave it as it is
            img = data.iloc[index]['center'].strip()
            angle = data.iloc[index]['steering']
            images.append(img)
            angles.append(angle)
        else:
            # if image is right image, subtract the STEERING_PLUS_MINUS coefficient
            img = data.iloc[index]['right'].strip()
            angle = data.iloc[index]['steering'] - STEERING_PLUS_MINUS
            images.append(img)
            angles.append(angle)

    X_train = np.array(images)
    y_train = np.array(angles)

    return X_train, y_train


def process_image(image, steering_angle):
    return image, steering_angle


def get_next_batch(batch_size=64):
    """
    Yields the next batch of training examples
    :param batch_size:
        Number of training examples in a batch
    :return:

    """
    while True:
        X_batch = []
        y_batch = []
        images, angles = get_next_img_file(batch_size)
        # HERE: TOO MANY_VALUES_TO UNPACK!
        for img_file, angle in zip(images, angles):
            # plt.imread returns image in RGB, opencv in BGR
            raw_image = plt.imread(IMG_FOLDER + img_file)
            raw_angle = angle
            new_image, new_angle = process_image(raw_image, raw_angle)
            X_batch.append(new_image)
            y_batch.append(new_angle)

        yield np.array(X_batch), np.array(y_batch)
#
#
# def rotate_img(image, steering_angle, rotation_amount=15):
#     """
#     rotate the image for image augmentation
#     """
#     angle = np.random.uniform(-rotation_amount, rotation_amount + 1)
#     rad = (np.pi / 180.0) * angle
#     return rotate(image, angle, reshape=False), steering_angle + (-1) * rad
#
#
# def flip_img(image, steering_angle, flipping_prob=0.5):
#     """
#     flips images with probability of 0.5
#
#     """
#     head = bernoulli.rvs(flipping_prob)
#     if head:
#         return np.fliplr(image), -1 * steering_angle
#     else:
#         return image, steering_angle
#
#
# def crop(img, top_pct, bottom_pct):
#     """
#     Crops an image
#     """
#
#     top = int(np.ceil(img.shape[0] * top_pct))
#     bottom = img.shape[0] - int(np.ceil(img.shape[0] * bottom_pct))
#
#     return img[top:bottom, :]
