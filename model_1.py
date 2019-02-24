import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense

# loads images and models from file

lines = []

with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

imgs = []
measurements = []

for sample in lines:
    source_path = sample[0]
    filename = source_path.split('/')[-1]
    current_path = '../data/IMG/' + filename
    img = cv2.imread(current_path)
    imgs.append(img)
    measurement = float(sample[3])
    measurements.append(measurement)

# without generator
X_train = np.array(imgs)
y_train = np.array(measurements)

model = Sequential()
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True)
model.save('model.h5')
