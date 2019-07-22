import json
import os
import glob
import numpy as np
import keras
import tensorflow as tf
from keras.models import Model, load_model, model_from_json
from keras.layers import Dense, Input, Flatten
from keras.utils import Sequence
from snake_engine import game_table
from keras import backend as K

class game_generator():
    
    def __init__(self, modelPath, size, fr, epochLength):
        self.model = NeuralNetwork(modelPath)
        self.size = size
        self.modelPath = modelPath
        self.fr = fr
        self.epochLength = epochLength
        self.on_epoch_end()

    def __len__(self):
        return self.epochLength

    def get_item(self):
        #initiate a game
        alive = True
        game  = game_table(size = self.size,
                            fr = self.fr)
        #play the game
        counter = 0
        while alive:
            counter += 1
            board = game.get_board()
            board = board[np.newaxis, :]
            move = self.model.predict(board)
            alive, fed = game.take_turn(move)
            if fed:
                counter = 0
            else:
                counter += 1
                if counter >= 500:
                    break

        #augment data
        game.adjust_reward()
        game.salt_moves()
        x,y = game.get_training_data()
        score = game.get_score()
        return x, y, score

    def on_epoch_end(self):
        self.model.load()

def load_model(modelPath):

    model_name = "model.json"
    #find latest weights
    path = modelPath + "/*.h5"
    print("PATH", path)
    list_of_files = glob.glob(path) # * means all if need specific format then *.csv
    weights_name = max(list_of_files, key=os.path.getctime)
    print("LIST OF FILES", list_of_files)
    print("WEIGHTS NAME", weights_name)

    if model_name is not None:
        # load the model
        json_file_path = os.path.join(modelPath, model_name)
        json_file = open(json_file_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
    if weights_name is not None:
        # load the weights
        weights_path = os.path.join(weights_name)
        model.load_weights(weights_path)
    return model

def make_dir(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        # directory already exists
        pass

class NeuralNetwork:
    def __init__(self, model_folder):
        # the folder in which the model and weights are stored
        self.model_folder = model_folder
        self.model = None
        self.modelIndex = 0

    def load(self):
        try:
            #set model name
            model_name = "model.json"
            #find latest weights
            path = self.model_folder + "*.h5"
            list_of_files = glob.glob(path) # * means all if need specific format then *.csv
            self.modelIndex = len(list_of_files) + 1
            weights_name = max(list_of_files, key=os.path.getctime)
            if model_name is not None:
                # load the model
                json_file_path = os.path.join(self.model_folder, model_name)
                json_file = open(json_file_path, 'r')
                loaded_model_json = json_file.read()
                json_file.close()
                self.model = model_from_json(loaded_model_json)
            if weights_name is not None:
                # load the weights
                weights_path = os.path.join(weights_name)
                self.model.load_weights(weights_path)
            self.model.compile(optimizer = 'Adam', loss='mean_squared_error')
            return True
        except Exception as e:
            print(e)
            return False

    def save(self):
        model_json = self.model.to_json()
        with open(self.model_folder+"model.json", "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(self.model_folder+"model_" + str(self.modelIndex) + ".h5")
        print("Saved model to disk")
    
    def fit(self,x,y):
        self.model.fit(x, y, verbose = 0)

    def fit_generator(self, generator, steps_per_epoch=None,
                            epochs=1, verbose=1,
                            callbacks=None, validation_data=None, 
                            validation_steps=None, validation_freq=1,
                            class_weight=None, max_queue_size=10,
                            workers=1, use_multiprocessing=False, 
                            shuffle=True, initial_epoch=0):
            self.model.fit_generator(generator, steps_per_epoch=steps_per_epoch,
                                        epochs=epochs, verbose=verbose,
                                        callbacks=callbacks, validation_data=validation_data,
                                        validation_steps=validation_steps,
                                        class_weight=class_weight, max_queue_size=max_queue_size,
                                        workers=workers, use_multiprocessing=use_multiprocessing,
                                        shuffle=shuffle, initial_epoch=initial_epoch)


    def predict(self, x):
        y = self.model.predict(x)
        return y

def generator(modelPath, size, fr, epochLength):
    model = NeuralNetwork(modelPath)
    game  = game_table(size = size,
                    fr = fr)                
    while True:
        #initiate a game
        model.load()
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
            if fed:
                counter = 0
            else:
                counter += 1
                if counter >= 500:
                    break

        #augment data
        x,y = game.parse_data()
        yield x, y
