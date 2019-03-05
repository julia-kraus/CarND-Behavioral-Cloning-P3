import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.ndimage import rotate
import cv2

DATA_HOME = './data'
DRIVING_LOG_FILE = os.path.join(DATA_HOME, 'driving_log.csv')
STEERING_PLUS_MINUS = 0.229


def get_next_img_file(batch_size=64):
    """Extracts driving log file to image-label pairs in batches"""
    data = pd.read_csv(DRIVING_LOG_FILE)
    # pick random images for batch
    batch_indices = np.random.randint(0, len(data), batch_size)

    img_files = []
    angles = []
    for index in batch_indices:
        img_center = data.iloc[index]['center'].strip()
        img_left = data.iloc[index]['left'].strip()
        img_right = data.iloc[index]['right'].strip()

        steering_center = data.iloc[index]['steering']
        steering_left = steering_center + STEERING_PLUS_MINUS
        steering_right = steering_center - STEERING_PLUS_MINUS

        img_files.extend([img_center, img_left, img_right])
        angles.extend([steering_center, steering_left, steering_right])

    X_train = np.array(images)
    y_train = np.array(angles)

    return X_train, y_train

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
            raw_image = plt.imread(os.path.join(DATA_HOME, img_file))
            raw_angle = angle
            new_image, new_angle = process_image(raw_image, raw_angle)
            X_batch.append(new_image)
            y_batch.append(new_angle)

        yield np.array(X_batch), np.array(y_batch)


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
