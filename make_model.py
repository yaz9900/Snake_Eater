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


modelPath = "models/test/"
size = 10
make_dir(modelPath)

conv_input = Input(shape=(size,size,2))
conv = keras.layers.Conv2D(16, (3,3), strides=(1, 1), padding='valid', activation = 'tanh')(conv_input)
conv = keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv)
conv = keras.layers.Conv2D(16, (3,3), strides=(1, 1), padding='valid', activation = 'tanh')(conv)
conv = keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv)
conv = keras.layers.Conv2D(16, (3,3), strides=(1, 1), padding='valid', activation = 'tanh')(conv_input)
conv = keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv)
conv = keras.layers.Conv2D(16, (3,3), strides=(1, 1), padding='valid', activation = 'tanh')(conv)
conv = keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv)
conv_out = keras.layers.Flatten()(conv)
conv_model = Model(conv_input, conv_out)

model_input = Input(shape=(None,size,size,2))
state_input = Input(shape=(128,))

x = keras.layers.TimeDistributed(conv_model)(model_input)
x, state_output = keras.layers.GRU(128, activation = 'relu', return_sequences = True, return_state = True)(x, initial_state = state_input)
x = keras.layers.Dense(64, activation = 'relu')(x)
predictions = keras.layers.Dense(4, activation = 'softmax')(x)

model = Model([model_input] + [state_input], [predictions] + [state_output])
model.compile(optimizer = 'adam', loss='mean_squared_error')
print(model.summary())



model_json = model.to_json()
with open(modelPath+"model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(modelPath+"model.h5")
print("Saved model to disk")



# list_of_files = glob.glob('models/test/*.h5') # * means all if need specific format then *.csv
# latest_file = maconv(list_of_files, key=os.path.getctime)
# print(latest_file)
