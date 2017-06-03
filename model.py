from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers import Convolution2D, BatchNormalization, Cropping2D
from keras.optimizers import Adam

from data_generator import d_train, d_valid, train_generator, validation_generator

BATCH_SIZE = 192
NUM_EPOCHS = 10
NROWS = 160
NCOLS = 320

###################################
# train CNN (NVIDIA model)
# ref.:
# - arxiv: https://arxiv.org/abs/1604.07316
# - keras impl.: https://github.com/0bserver07/Nvidia-Autopilot-Keras/blob/master/model.py#L45
###################################
model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(NROWS, NCOLS, 3),
        output_shape=(NROWS, NCOLS, 3)))
model.add(Cropping2D(cropping=((50,20), (0,0))))
#model.add(BatchNormalization(epsilon=0.001, mode=2,
#                             axis=1, input_shape=(3, NROWS, NCOLS)))
model.add(Convolution2D(24, 5, 5, border_mode='valid',
                        activation='relu', subsample=(2, 2)))
model.add(Convolution2D(36, 5, 5, border_mode='valid',
                        activation='relu', subsample=(2, 2)))
model.add(Convolution2D(48, 5, 5, border_mode='valid',
                        activation='relu', subsample=(2, 2)))
model.add(Convolution2D(64, 3, 3, border_mode='valid',
                        activation='relu', subsample=(1, 1)))
model.add(Convolution2D(64, 3, 3, border_mode='valid',
                        activation='relu', subsample=(1, 1)))
model.add(Flatten())
#model.add(Dense(1164, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='tanh'))

model.summary()

model.compile(loss='mse',
              optimizer=Adam(lr=0.0001))
history = model.fit_generator(train_generator,
                                steps_per_epoch= int(len(d_train)/BATCH_SIZE*6),
                                validation_data=validation_generator,
                                validation_steps=int(len(d_valid)/BATCH_SIZE*6),
                                epochs=NUM_EPOCHS, verbose=1)
#history = model.fit(X_train, Y_train,
#                    batch_size=BATCH_SIZE, nb_epoch=NUM_EPOCHS,
#                    verbose=1, validation_data=(X_test, Y_test))
#score = model.evaluate(X_test, Y_test, verbose=0)
#print('Test score:', score[0])
#print('Test accuracy:', score[1])

###################################
# save model
###################################
open('models/net.yaml', 'w').write(model.to_yaml())
model.save_weights('models/weight.h5')

###################################
# visualize model
###################################
from keras.utils import plot_model
plot_model(model, to_file="fig/model_cnn.png", show_shapes=True, show_layer_names=True)

###################################
# visualize log
###################################
import matplotlib.pyplot as plt

fig = plt.figure()
plt.plot(history.history['loss'],"o-",label="loss",)
plt.plot(history.history['val_loss'],"o-",label="val_loss")
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='upper right')
fig.savefig("fig/log.png")
