"""
ref: udacity's lecture
https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/46a70500-493e-4057-a78e-b3075933709d/concepts/b602658e-8a68-44e5-9f0b-dfa746a0cc1a
"""

import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

df = pd.read_csv("driving_log.csv").values
d_train, d_valid = train_test_split(df, test_size=0.2, random_state=2525)

def generator(samples, batch_size=128):
    num_samples = len(samples)
    while True:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

train_generator = generator(d_train)
validation_generator = generator(d_valid)


if __name__ == "__main__":
    from keras.models import Sequential
    from keras.layers import Dense, Lambda, Flatten

    ch, row, col = 3, 160, 320
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=(row, col, ch),
            output_shape=(row, col, ch)))
    model.add(Flatten())
    model.add(Dense(1, activation='tanh'))
    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(train_generator,
                        samples_per_epoch= len(d_train),
                        validation_data=validation_generator,
                        nb_val_samples=len(d_valid),
                        nb_epoch=2)
