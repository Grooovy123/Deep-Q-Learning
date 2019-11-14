import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from tqdm import tqdm
import numpy as np
from PIL import ImageGrab
import cv2
from grabscreen import grab_screen
import time
from collections import deque
import random
import math
import pygame
import tkinter as tk
from tkinter import messagebox

### REINFORCEMENT LEARNING ###
EPISODES = 25000
FOOD_REWARD = 50
OOB_PUNISHMENT = 200
EAT_YOURSELF_PUNISHMENT = 150
epsilon = 0.9
EPS_DECAY = 0.99
SHOW_EVERY = 2500

Start_q_table = None # or existing file

LEARNING_RATE = 0.1

### FOR NETWORK ###
REPLAY_MEMORY_SIZE = 50000
MIN_REPLAY_MEMORY_SIZE = 1000
MODEL_NAME = "256x2"
MINIBATCH_SIZE = 64
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 5

class cube(object):
    rows = 30
    w = 60
    def __init__(self,start,dirnx=1,dirny=0,color=(255,0,0)):
        self.pos = start        
        self.dirnx = random.randint(0,1)
        if self.dirnx == 1:            
            self.dirny = 0
        else:
            self.dirny = 1            
        self.color = color
       
    def move(self, dirnx, dirny):
        self.dirnx = dirnx
        self.dirny = dirny
        self.pos = (self.pos[0] + self.dirnx, self.pos[1] + self.dirny)
 
    def draw(self, surface, eyes=False):
        dis = self.w // self.rows
        i = self.pos[0]
        j = self.pos[1]
 
        pygame.draw.rect(surface, self.color, (i*dis+1,j*dis+1, dis-2, dis-2))
 
class snake(object):
    body = []
    turns = {}
    def __init__(self, color, pos):
        self.color = color
        self.head = cube(pos)
        self.body.append(self.head)
        self.dirnx = 0
        self.dirny = 1
 
    def move(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
 
            keys = pygame.key.get_pressed()
 
            for key in keys:
                if keys[pygame.K_LEFT]:
                    self.dirnx = -1
                    self.dirny = 0
                    self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
 
                elif keys[pygame.K_RIGHT]:
                    self.dirnx = 1
                    self.dirny = 0
                    self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
 
                elif keys[pygame.K_UP]:
                    self.dirnx = 0
                    self.dirny = -1
                    self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
 
                elif keys[pygame.K_DOWN]:
                    self.dirnx = 0
                    self.dirny = 1
                    self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]

        for i, c in enumerate(self.body):
            p = c.pos[:]
            if p in self.turns:
                turn = self.turns[p]
                c.move(turn[0],turn[1])
                if i == len(self.body)-1:
                    self.turns.pop(p)
            else:
                if c.dirnx == -1 and c.pos[0] <= 0:
                    print('Score: ', len(s.body))
                    reward -= OOB_PUNISHMENT
                    s.reset((random.randint(8,11),random.randint(8,11)))              
                elif c.dirnx == 1 and c.pos[0] >= c.rows-1:
                    print('Score: ', len(s.body))
                    reward -= OOB_PUNISHMENT
                    s.reset((random.randint(8,11),random.randint(8,11)))
                elif c.dirny == 1 and c.pos[1] >= c.rows-1:
                    print('Score: ', len(s.body))
                    reward -= OOB_PUNISHMENT
                    s.reset((random.randint(8,11),random.randint(8,11)))
                elif c.dirny == -1 and c.pos[1] <= 0:
                    print('Score: ', len(s.body))
                    reward -= OOB_PUNISHMENT
                    s.reset((random.randint(8,11),random.randint(8,11)))                    
                else: 
                    c.move(c.dirnx,c.dirny)
 
    def reset(self, pos):
        self.head = cube(pos)
        self.body = []
        self.body.append(self.head)
        self.turns = {}
        self.dirnx = 0
        self.dirny = 1
 
    def addCube(self):
        tail = self.body[-1]
        dx, dy = tail.dirnx, tail.dirny
 
        if dx == 1 and dy == 0:
            self.body.append(cube((tail.pos[0]-1,tail.pos[1])))
        elif dx == -1 and dy == 0:
            self.body.append(cube((tail.pos[0]+1,tail.pos[1])))
        elif dx == 0 and dy == 1:
            self.body.append(cube((tail.pos[0],tail.pos[1]-1)))
        elif dx == 0 and dy == -1:
            self.body.append(cube((tail.pos[0],tail.pos[1]+1)))
 
        self.body[-1].dirnx = dx
        self.body[-1].dirny = dy
 
    def draw(self, surface):
        for i, c in enumerate(self.body):
            if i ==0:
                c.draw(surface, True)
            else:
                c.draw(surface)
 
def drawGrid(w, rows, surface):
    sizeBtwn = w // rows
 
    x = 0
    y = 0
    for l in range(rows):
        x = x + sizeBtwn
        y = y + sizeBtwn
 
        pygame.draw.line(surface, (10,10,10), (x,0),(x,w))
        pygame.draw.line(surface, (10,10,10), (0,y),(w,y))
 
def redrawWindow(surface):
    global rows, width, s, snack
    surface.fill((0,0,0))    
    s.draw(surface)
    snack.draw(surface)    
    pygame.display.update()  
    #pixelArray = np.array(pygame.PixelArray(surface), dtype=np.uint8)/255    
    pixelArray = pygame.PixelArray(surface)
    return pixelArray
 
def randomSnack(rows, item):
 
    positions = item.body
 
    while True:
        x = random.randrange(rows)
        y = random.randrange(rows)
        if len(list(filter(lambda z:z.pos == (x,y), positions))) > 0:            
            continue
        else:
            break
       
    return (x,y)

def main():    
    global width, rows, s, snack, sk
    width = 60
    rows = 30
    win = pygame.display.set_mode((width, width))
            
    s = snake((255,0,0), (random.randint(6,12),random.randint(6,12)))
    snack = cube(randomSnack(rows, s), color=(0,255,0))
    flag = True

    clock = pygame.time.Clock()

    while flag:
        pygame.time.delay(50)
        clock.tick(12)
        s.move()
        if s.body[0].pos == snack.pos:
            s.addCube()
            snack = cube(randomSnack(rows, s), color=(0,255,0))
            reward += FOOD_REWARD

        for x in range(len(s.body)):
            if s.body[x].pos in list(map(lambda z:z.pos,s.body[x+1:])):
                print('Score: ', len(s.body))
                reward -= EAT_YOURSELF_PUNISHMENT
                s.reset((random.randint(6,12), random.randint(6,12)))
                break            
           
        #print(redrawWindow(win))
        redrawWindow(win)
    pass
        


# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

class DQNAgent:
	def __init__(self):
		# Main model # Gets trained every step
		self.model = self.create_model()

		# Target model this is what we predict against every step
		self.target_model = self.create_model()
		self.target_model.set_weights(self.model.get_weights())

		self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

		self.tensorBoard = ModifiedTensorBoard(log_dir = f"logs/{MODEL_NAME}-{int(time.time())}")
		self.target_update_counter = 0

	def create_model(self):		
		model = Sequential()

		model.add(Conv2D(64, (3, 3), input_shape=(40,40,3)))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.2))

		model.add(Conv2D(256,(3, 3)))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))		
		model.add(Dropout(0.2))

		model.add(Flatten())
		model.add(Dense(64))		
		model.add(Dense(4))
		model.add(Activation('softmax'))

		model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])

		return model

	def update_replay_memory(self,transition):
		self.replay_memory.append(transition)

	def get_qs(self, terminal_state, state):
		return self.model_predict(np.array(state).reshape(-1, *state.shape)/255)[0]

	def train(self, terminal_state, state, step):
		if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
		    return

		minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

		current_states = np.array([transition[0] for transition in minibatch])/255
		current_qs_list = self.model.predict(current_states)

		new_current_states = np.array([transition[3] for transition in minibatch])/255
		future_qs_list = self.target_model.predict(new_current_states)

		x=[]
		y=[]

		for index, (current_states, action, reward, new_current_states, done) in enumerate(minibatch):
                    if not done:
                        max_future_q = np.max(future_qs_list[index])
                        new_q = reward + DISCOUNT * max_future_q
                    else:
                        new_q = reward

                    current_qs = current_qs_list[index]
                    current_qs[action] = new_q

                    x.append(current_states)
                    y.append(current_qs)

		self.model.fit(np.array(x)/255, np.array(y), batch_size = MINIBATCH_SIZE,
			verbose=0, shuffle=False, callbacks=[self.Tensorboard] if terminal_state else None)

		# updating to determine if we want to update target_model yet
		if terminal_state:
		    self.target_update_counter +=1

		if self.target_update_counter > UPDATE_TARGET_EVERY:
		    self.taget_model.set_weights(self.model.get_weights())
		    self.target_update_counter = 0

Agent = DQNAgent()

for episode in tqdm(range(1, EPISODES+1), ascii=True, unit='episode'):
	Agent.tensorBoard.step = episode

	episode_reward = 0
	step = 1
	s.reset((random.randint(8,11),random.randint(8,11)))
	current_states = redrawWindow(win)

	done = False

	while not done:
		if np.random.random() > epsilon:
		    action = np.argmax(Agent.get_qs(current_states))
		    ### implement key presses
                        
		else:
		    action = np.random.randint(0, 4)

		new_state = redrawWindow(win)
		episode_reward += reward

		Agent.update_replay_memory((current_state, action, reward, new_state, done))
		Agent.train(done, step)

		current_state = new_state
		step += 1

main()
