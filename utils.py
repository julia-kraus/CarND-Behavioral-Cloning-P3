import numpy as np
import pandas as pd
import os
from PIL import Image

DATA_HOME = './data'
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

        img_center = process_image(np.asarray(Image.open(os.path.join(DATA_HOME, center_path))), steering_center)
        img_left = process_image(np.asarray(Image.open(os.path.join(DATA_HOME, left_path))), steering_right)
        img_right = process_image(np.asarray(Image.open(os.path.join(DATA_HOME, right_path))), steering_right)

        images.extend([img_center, img_left, img_right])
        angles.extend([steering_center, steering_left, steering_right])

    X_train = np.array(images)
    y_train = np.array(angles)

    return X_train, y_train


def process_image(img, angle):
    img = crop(img)
    # img = resize(img, (128, 128))
    img, angle = horizontal_flip(img, angle)
    return img, angle


def get_next_batch(batch_size=64):
    """
    Yields the next batch of training examples
    :param batch_size:
        Number of training examples in a batch
    :return:

    """
    while True:
        images, angles = get_images_angles(batch_size)

        yield images, angles


# horizontal or vertical flip??
def horizontal_flip(image, angle):
    """
    flips images with probability of 0.5

    """
    flip = np.random.randint(0, 1)
    if flip:
        return np.fliplr(image), -1 * angle
    else:
        return image, angle


def resize(image, new_dim):
    """
    Resize a given image according the the new dimension
    :param image:
        Source image array
    :param new_dim:
        new dimensions
    :return:
        Resize image
    """

    return np.array(Image.fromarray(image).resize(new_dim))


def crop(image, top_crop=60, bottom_crop=20):
    """
    Crops an image according to the given parameters
    :param image: source image
    :param top_percent:
        The percentage of the original image will be cropped from the top of the image
    :param bottom_percent:
        The percentage of the original image will be cropped from the bottom of the image
    :return:
        The cropped image
    """

    return image[top_crop:-bottom_crop, :]

# multiple cameras can be done better like that
# steering_center = float(row[3])
#
#             # create adjusted steering measurements for the side camera images
#             correction = 0.2 # this is a parameter to tune
#             steering_left = steering_center + correction
#             steering_right = steering_center - correction
#
#             # read in images from center, left and right cameras
#             path = "..." # fill in the path to your training IMG directory
#             img_center = process_image(np.asarray(Image.open(path + row[0])))
#             img_left = process_image(np.asarray(Image.open(path + row[1])))
#             img_right = process_image(np.asarray(Image.open(path + row[2])))
#
#             # add images and angles to data set
#             car_images.extend(img_center, img_left, img_right)
#             steering_angles.extend(steering_center, steering_left, steering_right)
