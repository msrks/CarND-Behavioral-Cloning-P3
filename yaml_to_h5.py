from keras.models import model_from_yaml

model = model_from_yaml(open('models/net.yaml').read())
model.load_weights("models/weight.h5")
model.compile(loss='mse', optimizer='adam')

model.save("model.h5")
