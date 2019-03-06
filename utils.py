import numpy as np
import pandas as pd
import os
from PIL import Image
np.random.seed(0)
import matplotlib.pyplot as plt

DATA_HOME = './data_mine'
DRIVING_LOG_FILE = os.path.join(DATA_HOME, 'driving_log.csv')
STEERING_CORRECTION = 0.229


def get_images_angles(batch_size=64):
    """Extracts driving log file to image-label pairs in batches"""
    data = pd.read_csv(DRIVING_LOG_FILE)
    batch_indices = np.random.randint(0, len(data), batch_size)

    images = []
    angles = []

    for index in batch_indices:
        center_path = data.iloc[index]['center'].strip()
        left_path = data.iloc[index]['left'].strip()
        right_path = data.iloc[index]['right'].strip()

        steering_center = data.iloc[index]['steering']
        steering_left = steering_center + STEERING_CORRECTION
        steering_right = steering_center - STEERING_CORRECTION

        img_center, steering_center = process_image(np.asarray(Image.open(os.path.join(DATA_HOME, center_path))),
                                                    steering_center)
        img_left, steering_left = process_image(np.asarray(Image.open(os.path.join(DATA_HOME, left_path))),
                                                steering_left)
        img_right, steering_right = process_image(np.asarray(Image.open(os.path.join(DATA_HOME, right_path))),
                                                  steering_right)

        images.extend([img_center, img_left, img_right])
        angles.extend([steering_center, steering_left, steering_right])

    X_train = np.array(images)
    y_train = np.array(angles)

    return X_train, y_train


def process_image(img, angle):
    img = crop(img)
    img = resize(img, (64, 64))
    img, angle = horizontal_flip(img, angle)
    return img, angle


def get_next_batch(batch_size=64):
    """
    Yields the next batch of training examples

    """
    while True:
        images, angles = get_images_angles(batch_size)

        yield images, angles


def horizontal_flip(image, angle):
    """
    flips images with probability of 0.5

    """
    flip = np.random.randint(0, 2)

    if flip:
        return np.fliplr(image), -1 * angle
    else:
        return image, angle


def resize(image, new_dim):
    """Resize a given image according the the new dimension"""

    return np.array(Image.fromarray(image).resize(new_dim))


def crop(image, top_crop=60, bottom_crop=20):
    """
    Crops an image according to the given parameters

    """
    return image[top_crop:-bottom_crop, :]

