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



modelPath= "models/conv_v1/"
make_dir(modelPath)

epochs = 10
length = 100
fr = 0.5
size = 40

generator = utils.generator(modelPath, size, fr, length)

model = NeuralNetwork(modelPath)
model.load()

model.fit_generator(generator,
                    epochs=epochs,
                    steps_per_epoch=length,
                    max_queue_size = 2,
                    use_multiprocessing=False,
                    workers = 0)




# def save_model():
#     model_json = model.to_json()
#     with open(modelPath+"model.json", "w") as json_file:
#         json_file.write(model_json)
#     model.save_weights(model_json+"model.h5")
#     print("Saved model to disk")