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
model = NeuralNetwork(modelPath)
model.load()   
game  = game_table(size = 40,
                fr = 0.5)    
game.reset_board()
alive = True
counter = 0

#play the game
while alive:
    counter += 1
    board = game.get_board()
    board = board[np.newaxis, :]
    move = model.predict(board)
    alive, fed = game.take_turn(move)
    print(game.get_snake())
