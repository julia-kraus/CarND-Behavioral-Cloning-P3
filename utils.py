import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import errno
from scipy.ndimage import rotate
import cv2

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


def rotate_img(image, steer_angle, max_rotation=15):
    """
    rotate the image for image augmentation
    """
    rot_angle = np.random.uniform(-max_rotation, max_rotation + 1)
    rot_rad = (np.pi / 180.0) * rot_angle
    return rotate(image, rot_angle, reshape=False), steer_angle + (-1) * rot_rad


#
def vertical_flip(image, angle):
    """
    flips images with probability of 0.5

    """
    flip = np.random.randint(0, 1)
    if flip:
        return np.fliplr(image), -1 * angle
    else:
        return image, angle
