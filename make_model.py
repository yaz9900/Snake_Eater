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


model_input = Input(shape=(None,size,size))
state_input = Input(shape=(128,))

model_reshape_input = Input(shape=(size,size))
model_reshape = keras.layers.Reshape((size**2,))(model_reshape_input)
model_reshape = Model(model_reshape_input, model_reshape)

x = keras.layers.TimeDistributed(model_reshape)(model_input)
print(x.shape)
x, state_output = keras.layers.GRU(128, activation = 'tanh', return_sequences = True, return_state = True)(x, initial_state = state_input)
x = keras.layers.Dense(64, activation = 'tanh')(x)
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
