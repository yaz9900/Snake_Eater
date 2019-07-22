import json
import os
import glob
import numpy as np
import keras
from keras.models import Model, load_model, model_from_json
from keras.layers import Dense, Input, Flatten
from keras.utils import Sequence
from snake_engine import game_table
from utils import make_dir


modelPath = "models/conv_v1/"
size = 40
make_dir(modelPath)

model_input = Input(shape=(size,size,1))

conv = keras.layers.Conv2D(32, (3,3), strides=(1, 1), padding='valid', activation = 'tanh')(model_input)
conv = keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv)
conv = keras.layers.Conv2D(32, (3,3), strides=(1, 1), padding='valid', activation = 'tanh')(conv)
conv = keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv)
conv = keras.layers.Conv2D(32, (3,3), strides=(1, 1), padding='valid', activation = 'tanh')(conv)
conv = keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv)
conv_out = keras.layers.Flatten()(conv)

x = keras.layers.Dense(256, activation = 'relu')(conv_out)
x = keras.layers.Dense(128, activation = 'relu')(x)
predictions = keras.layers.Dense(4, activation = 'relu')(x)

model = Model(model_input, predictions)
model.compile(optimizer = 'adam', loss='mean_squared_error')
print(model.summary())



model_json = model.to_json()
with open(modelPath+"model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(modelPath+"model_1.h5")
print("Saved model to disk")



# list_of_files = glob.glob('models/test/*.h5') # * means all if need specific format then *.csv
# latest_file = maconv(list_of_files, key=os.path.getctime)
# print(latest_file)
