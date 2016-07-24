#!/usr/bin/env python

# ale_python_test_pygame_player.py
# Author: Ben Goodrich
#
# This modified ale_python_test_pygame.py to provide a fully interactive experience allowing the player
# to play. RAM Contents, current action, and reward are also displayed.
# keys are:
# arrow keys -> up/down/left/right
# z -> fire button
import sys
from ale_python_interface import ALEInterface
import numpy as np
import random
import pygame
import matplotlib.pyplot as plt
from scipy.misc import imresize

import util
from model_runner import ModelRunner

class DeepRLPlayer:
    def __init__(self, settings):
        self.settings = settings
        self.grayPixels = np.zeros((84, 84), np.float)

        self.greedyEpsilon = 1.0
        
        self.testModel = ModelRunner('test model',
                                     prototxt = settings['PROTOTXT'])
        self.trainModel = ModelRunner('train model', 
                                      solver_prototxt = settings['TRAIN_MODEL'], 
                                      maxReplayMemory = settings['MAX_REPLAY_MEMORY'],
                                      discountFactor = settings['DISCOUNT_FACTOR'],)
        

    def displayInfo(self, screen, ram, a, total_reward):
            font = pygame.font.SysFont("Ubuntu Mono",32)
            text = font.render("RAM: " ,1,(255,208,208))
            screen.blit(text,(330,10))
        
            font = pygame.font.SysFont("Ubuntu Mono",25)
            height = font.get_height()*1.2
        
            line_pos = 40
            ram_pos = 0
            while(ram_pos < 128):
                ram_string = ''.join(["%02X "%ram[x] for x in range(ram_pos,min(ram_pos+16,128))])
                text = font.render(ram_string,1,(255,255,255))
                screen.blit(text,(340,line_pos))
                line_pos += height
                ram_pos +=16
                
            #display current action
            font = pygame.font.SysFont("Ubuntu Mono",32)
            text = font.render("Current Action: " + str(a) ,1,(208,208,255))
            height = font.get_height()*1.2
            screen.blit(text,(330,line_pos))
            line_pos += height
        
            #display reward
            font = pygame.font.SysFont("Ubuntu Mono",30)
            text = font.render("Total Reward: " + str(total_reward) ,1,(208,255,255))
            screen.blit(text,(330,line_pos))

    def display(self, rgb, gray=False):
        if (gray):
            plt.imshow(rgb, cmap='gray')
        else:
            plt.imshow(rgb)
        plt.show()

    def getScreenPixels(self, ale, screen, game_surface):
        
        numpy_surface = np.frombuffer(game_surface.get_buffer(),dtype=np.int32)
        ale.getScreenRGB(numpy_surface)
        
        #del numpy_surface
        

        if (self.settings['SHOW_SCREEN']):
            screen.fill((0,0,0))
            screen.blit(pygame.transform.scale2x(game_surface),(0,0))
            #screen.blit(game_surface.copy(),(0,0))

        data = numpy_surface.view(np.uint8).reshape(numpy_surface.shape + (4,))
        rgba = np.reshape(data, (self.screen_height, self.screen_width, 4))
        rgb = rgba[:, :, (2, 1, 0)]
        resized = imresize(rgb, (110, 84, 3))
        cropped = resized[26:110, :, :].astype(np.float) / 256
        
        #if self.step > 200:        
        #    self.display(cropped)
        
        self.grayPixels = np.dot(cropped[...,:3], [0.299, 0.587, 0.114])        
        #self.display(self.grayPixels, gray=True)

        return self.grayPixels
            
    def checkExit(self, pressed): 
        #process pygame event queue
        exit=False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit=True
                break;
        if(pressed[pygame.K_q]):
            exit = True
        return exit
        
    def getActionFromModel(self, state):
        if self.step <= 10**6:
            # DJDJ
            #self.greedyEpsilon = 1.0 - 0.9 / 10**6 * self.step
            self.greedyEpsilon = 0.1 

        if random.random() < self.greedyEpsilon:
            return random.randrange(0, settings['MAX_ACTION_NO'])
        else:
            actionValues = self.testModel.test(state)
            
            #return np.argmax(actionValues)
            return 1
    
        
    def gogo(self):
        ale = ALEInterface()
        
        max_frames_per_episode = ale.getInt("max_num_frames_per_episode");
        ale.set("random_seed",123)
        
        random_seed = ale.getInt("random_seed")
        print("random_seed: " + str(random_seed))
        
        ale.loadROM(sys.argv[1])
        legal_actions = ale.getMinimalActionSet()
        print legal_actions
        
        (self.screen_width,self.screen_height) = ale.getScreenDims()
        print("width/height: " +str(self.screen_width) + "/" + str(self.screen_height))
        
        (display_width,display_height) = (1024,420)
        
        pygame.init()
        game_surface = pygame.Surface((self.screen_width,self.screen_height))

        if self.settings['SHOW_SCREEN']:
            screen = pygame.display.set_mode((display_width,display_height))
            pygame.display.set_caption("Arcade Learning Environment Player Agent Display")
            
            pygame.display.flip()
        else:
            screen = None
        
        ram_size = ale.getRAMSize()
        ram = np.zeros((ram_size),dtype=np.uint8)
        clock = pygame.time.Clock()
        
        episode = 0
        total_reward = 0.0 
        self.step = 0
        newState = None
        
        state = self.getScreenPixels(ale, screen, game_surface)
        while(episode < 10):

            if self.settings['USE_KEYBOARD']:
                action, pressed = util.getActionFromKeys()
                reward = ale.act(action);
            else:

                if newState != None:
                    self.trainModel.addData(state, action, reward, newState)
                    self.trainModel.train()
                    state = newState
                    
                action = self.getActionFromModel(state)
                reward = ale.act(action);
                
                if reward > 0:
                    reward = 1
                elif reward < 0:
                    reward = -1

                newState = self.getScreenPixels(ale, screen, game_surface)

            
                if self.step > 0 and self.step % self.settings['UPDATE_STEP'] == 0:
                    self.testModel.copy(self.trainModel)
            
            
            
            ale.getRAM(ram)
            
            if self.settings['SHOW_SCREEN']:
                self.displayInfo(screen, ram, action, total_reward)
                pygame.display.flip()
        
            if(self.settings['USE_KEYBOARD'] and self.checkExit(pressed)):
                break
        
            self.step +=1 
            total_reward += reward
            
            #delay to 60fps
            #clock.tick(60.)
        
            if(ale.game_over()):
                episode_frame_number = ale.getEpisodeFrameNumber()
                frame_number = ale.getFrameNumber()
                print("Frame Number: " + str(frame_number) + " Episode Frame Number: " + str(episode_frame_number))
                print("Episode " + str(episode) + " ended with score: " + str(total_reward))
                ale.reset_game()
                total_reward = 0.0 
                episode = episode + 1
        
    
if __name__ == '__main__':    
    if(len(sys.argv) < 2):
        print("Usage ./ale_python_test_pygame_player.py <ROM_FILE_NAME>")
        sys.exit()
    
    settings = {}

    settings['SHOW_SCREEN'] = True
    #settings['SHOW_SCREEN'] = False
    settings['USE_KEYBOARD'] = False

    settings['TRAIN_MODEL'] = 'models/solver.prototxt'
    settings['PROTOTXT'] = 'models/test.prototxt'
    settings['MAX_REPLAY_MEMORY'] = 10000
    settings['DISCOUNT_FACTOR'] = 0.9
    settings['LEARNING_RATE'] = 0.01
    settings['MAX_ACTION_NO'] = 18
    settings['UPDATE_STEP'] = 100
    
    player = DeepRLPlayer(settings)
    player.gogo()
