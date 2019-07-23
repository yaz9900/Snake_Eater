import json
import numpy as np
import keras
import tensorflow as tf
from keras.models import Model, load_model, model_from_json
from keras.layers import Dense, Input, Flatten
from keras.utils import Sequence
from snake_engine import game_table
from utils import game_table, game_generator, make_dir, NeuralNetwork
import utils



modelPath= "models/test/"
make_dir(modelPath)
filepath = modelPath + "model.h5"
epochs = 200
length = 10
fr = 0.95
size = 10

generator = utils.generator(modelPath, size, fr, length)

model = NeuralNetwork(modelPath)
model.load()

callbacks = [keras.callbacks.ModelCheckpoint(filepath, monitor="epoch", 
                                             verbose=1, save_best_only=False, 
                                             save_weights_only=True, mode='auto', 
                                             period=1)]
# loss_down = keras.callbacks.ModelCheckpoint(filepath, monitor="loss", 
#                                              verbose=1, save_best_only=False, 
#                                              save_weights_only=True, mode='min', 
#                                              period=1)                            
# callbacks = [loss_up, loss_down]                 
model.fit_generator(generator,
                    epochs=epochs,
                    steps_per_epoch=length,
                    max_queue_size = 1,
                    use_multiprocessing=False,
                    workers = 0,
                    callbacks = callbacks)




# def save_model():
#     model_json = model.to_json()
#     with open(modelPath+"model.json", "w") as json_file:
#         json_file.write(model_json)
#     model.save_weights(model_json+"model.h5")
#     print("Saved model to disk")