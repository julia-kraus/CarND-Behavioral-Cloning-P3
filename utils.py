import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import errno

DRIVING_LOG_FILE = './data/driving_log.csv'
STEERING_PLUS_MINUS = 0.229


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
    """Extracts driving log file to """
    data = pd.read_csv(DRIVING_LOG_FILE)
    rnd_indices = np.random.randint(0, len(data), batch_size)

    image_files_and_angles = []
    for index in rnd_indices:
        rnd_image = np.random.randint(0, 3)
        if rnd_image == 0:
            # if image is left image, add the STEERING_PLUS_MINUS coefficient to the steering angle
            img = data.iloc[index]['left'].strip()
            angle = data.iloc[index]['steering'] + STEERING_PLUS_MINUS
            image_files_and_angles.append((img, angle))

        elif rnd_image == 1:
            # if image is central image, leave it as it is
            img = data.iloc[index]['center'].strip()
            angle = data.iloc[index]['steering']
            image_files_and_angles.append((img, angle))
        else:
            # if image is right image, subrtract the STEERING_PLUS_MINUS coefficient
            img = data.iloc[index]['right'].strip()
            angle = data.iloc[index]['steering'] - STEERING_PLUS_MINUS
            image_files_and_angles.append((img, angle))

    return image_files_and_angles


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
        images = get_next_img_file(batch_size)
        for img_file, angle in images:
            # plt.imread returns image in RGB, opencv in BGR
            raw_image = plt.imread(IMG_PATH + img_file)
            raw_angle = angle
            new_image, new_angle = process_image(raw_image, raw_angle)
            X_batch.append(new_image)
            y_batch.append(new_angle)

        assert len(X_batch) == batch_size, 'len(X_batch) == batch_size should be True'

        yield np.array(X_batch), np.array(y_batch)
