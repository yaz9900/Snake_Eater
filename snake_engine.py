# import keras

import numpy as np
import random
# from numpy.random import randint
class game_table:

    def __init__(self, size = 3, fr = 0.5):
        self.size = size
        self.fr = fr
        self.emptyBoard = np.zeros((size, size,1))
        self.gameStates  = []
        self.boardStates = []
        self.moveStates = []
        self.startPoint = (int(self.size/2),int(self.size/2),0)
        self.score = 0
        #set empty tiles
        emptyTiles = {}
        for a in range(self.size):
            for b in range(self.size):
                emptyTiles[(a,b,0)] = [a,b,0]
        snake = [self.startPoint]
        for tile in snake:
            emptyTiles.pop(tile)
        #generate food
        food= random.sample( emptyTiles.keys(), 1 )[0]
        emptyTiles.pop(food)
        #setup initial game state
        boardState = np.zeros((self.size, self.size,1))
        for tile in snake:
            boardState[tile] = 1. 
        boardState[food] = -1
        self.boardStates.append(boardState)
        gameState = {'snake' : snake,
                     'food' : food,
                     'emptyTiles' : emptyTiles,
                     'reward' : 0,
                     'alive' : True
        }
        self.gameStates.append(gameState)


    def take_turn(self, moveState):
        #get last game state
        #find snake head and tail
        snake = self.gameStates[-1]["snake"]
        food = self.gameStates[-1]["food"]
        emptyTiles = self.gameStates[-1]["emptyTiles"].copy()
        snakeHead = snake[-1]
        #find direction
        direction = np.argmax(moveState)
        #move snake head
        alive = True
        fed = False
        x, y, _= snakeHead 
        if direction == 0:
            y += 1
        elif direction == 1:
            y += -1
        elif direction == 2:
            x -= 1
        elif direction == 3:
            x += 1
        #check for walls
        if x >= self.size or x < 0:
            # print("killed by hitting a side wall")
            alive = False
        if y >= self.size or y < 0:
            # print("killed by hitting a top wall")
            alive = False
        #check for snake
        if (x,y,1) in snake:
            alive = False
            # print("killed by hitting itself")
        #check for food
        if (x,y,0) == self.gameStates[-1]["food"]:
            fed = True
        #update snake
        if alive:
            snake.append((x,y,0))
            if (x,y) in emptyTiles:
                emptyTiles.pop((x,y))
            #update empty tiles
            if not fed:
                tail = snake.pop(0)
                emptyTiles[tail] = [tail[0],tail[1]]
        #make new food
        if fed: 
            self.score += 1
            food = random.sample( emptyTiles.keys(), 1 )[0]
            emptyTiles.pop(food)
        #calcuate reward
        reward = 0 
        if fed:
            reward += 1
        if not alive:
            reward -= 1
        #assemble game state
        gameState = {'snake' : snake,
                     'food' : food,
                     'emptyTiles' : emptyTiles,
                     'reward' : reward,
                     'alive' : alive
        }
        #assemble board state
        boardState = np.zeros((self.size, self.size,1))
        for tile in snake:
            boardState[tile] = 1
        boardState[food] = -1
        #append data
        if alive:
            self.boardStates.append(boardState)
        self.gameStates.append(gameState)
        self.moveStates.append(np.array(moveState, copy=True))
        return alive, fed

    def get_board(self, index = -1):
        return self.boardStates[index]

    def get_rewards(self):
        return [d["reward"]for d in self.gameStates]

    def get_moves(self):
        # if not index:
        #     return self.moveStates[-1]
        return self.moveStates

    def get_score(self):
        return self.score

    def reset_board(self):
        self.gameStates  = []
        self.boardStates = []
        self.moveStates = []
        self.score = 0
        #set empty tiles
        emptyTiles = {}
        for a in range(self.size):
            for b in range(self.size):
                emptyTiles[(a,b,0)] = [a,b,0]
        snake = [self.startPoint]
        for tile in snake:
            emptyTiles.pop(tile)
        #generate food
        food= random.sample( emptyTiles.keys(), 1 )[0]
        emptyTiles.pop(food)
        #setup initial game state
        boardState = np.zeros((self.size, self.size,1))
        for tile in snake:
            boardState[tile] = 1. 
        boardState[food] = -1
        self.boardStates.append(boardState)
        gameState = {'snake' : snake,
                     'food' : food,
                     'emptyTiles' : emptyTiles,
                     'reward' : 0,
                     'alive' : True
        }
        self.gameStates.append(gameState)
    def parse_data(self):
        #adjust rewards
        running_reward = 0
        for state in reversed(self.gameStates):
            running_reward += state["reward"]
            state["reward"] = running_reward
            running_reward = running_reward * self.fr
        #salt moves
        for move, reward in zip(reversed(self.moveStates), reversed([d["reward"]for d in self.gameStates])):
            index = np.argmax(move)
        move_value = move[0][index]
        move[0][index] = move_value+reward
        #get training data
        if len(self.boardStates) > len(self.moveStates):
            del self.boardStates[-1]
        return  np.stack(self.boardStates), np.vstack(self.moveStates)