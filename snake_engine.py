# import keras

import numpy as np
import random
# from numpy.random import randint
class game_table:

    def __init__(self, size = 3, fr = 0.5):
        self.size = size
        self.fr = fr
        self.startPoint = (int(self.size/2),int(self.size/2))
        #set empty tiles
        emptyTiles = {}
        for a in range(self.size):
            for b in range(self.size):
                emptyTiles[(a,b)] = [a,b]
        #set snake
        snake = [self.startPoint]
        for tile in snake:
            emptyTiles.pop(tile)
        #generate food
        food= random.sample( emptyTiles.keys(), 1 )[0]
        emptyTiles.pop(food)
        #set game state
        self.gameStates = []
        gameState = {'snake' : snake,
                     'food' : food,
                     'emptyTiles' : emptyTiles,
                     'reward' : 0,
                     'alive' : True
        }
        self.gameStates.append(gameState)
        self.fedCounter = 50
        self.score = 0


    def take_turn(self, direction):
        #get last game state
        #find snake head and tail
        snake = self.gameStates[-1]["snake"]
        food = self.gameStates[-1]["food"]
        emptyTiles = self.gameStates[-1]["emptyTiles"].copy()
        snakeHead = snake[-1]
        #move snake head
        alive = True
        fed = False
        x, y = snakeHead 
        if direction == 0:
            y += 1
        elif direction == 1:
            y += -1
        elif direction == 2:
            x -= 1
        elif direction == 3:
            x += 1
        else:
            alive = False
            print("failed to make a move")
        #check for walls
        if x >= self.size or x < 0:
            # print("killed by hitting a side wall")
            alive = False
        if y >= self.size or y < 0:
            # print("killed by hitting a top wall")
            alive = False
        #check for snake
        if (x,y) in snake:
            alive = False
            # print("killed by hitting itself")
        #check for food
        if (x,y) == self.gameStates[-1]["food"]:
            fed = True
        #update snake
        if alive:
            snake.append((x,y))
            if (x,y) in emptyTiles:
                emptyTiles.pop((x,y))
            #update empty tiles
            if not fed:
                tail = snake.pop(0)
                emptyTiles[tail] = [tail[0],tail[1]]
        #make new food
        if fed: 
            self.score += 1
            self.fedCounter += 20
            food = random.sample( emptyTiles.keys(), 1 )[0]
            emptyTiles.pop(food)
        #calcuate reward
        reward = 0 
        if fed:
            reward += 0.1
        if not alive:
            reward -= 0.1
        #assemble game state
        gameState = {'snake' : snake,
                     'food' : food,
                     'emptyTiles' : emptyTiles,
                     'reward' : reward,
                     'alive' : alive
        }
        self.gameStates.append(gameState)
        #update fed counter, kill snake if not fed
        self.fedCounter -= 1
        if self.fedCounter <= 0:
            alive = False

        return alive, fed

    def get_rewards(self):
        return [d["reward"]for d in self.gameStates]

    def get_score(self):
        return self.score

    def get_snake(self):
        return self.gameStates[-1]['snake']

    def get_tiles(self, last = True):
        if last:
            return self.size, self.gameStates[-1]["snake"], self.gameStates[-1]["food"]
        else:
            return self.size, [d["snake"]for d in self.gameStates], [d["food"]for d in self.gameStates]           
