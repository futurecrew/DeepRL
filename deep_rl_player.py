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
import multiprocessing
import cv2
import matplotlib.pyplot as plt
from scipy.misc import imresize
import pickle
import threading
import time
import util
from model_runner import ModelRunner
from replay_memory import ReplayMemory

class DeepRLPlayer:
    def __init__(self, settings):
        self.settings = settings
        self.grayPixels = np.zeros((84, 84), np.float)
        self.zeroHistory = []
        for i in range(4):
            self.zeroHistory.append(np.zeros((84, 84), dtype=np.uint8))

        self.greedyEpsilon = 1.0
        self.sendQueue = multiprocessing.Queue()

        self.replayMemory = ReplayMemory(self.settings['MAX_REPLAY_MEMORY'], self.settings)
        # DJDJ        
        #DebugInput(self).start()
        

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

    """
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
        
        #self.grayPixels = np.dot(cropped[...,:3], [0.299, 0.587, 0.114])
        self.grayPixels = np.dot(cropped[...,:3], [29.9, 58.7, 11.4]).astype(np.uint8)
        #self.display(self.grayPixels, gray=True)

        return self.grayPixels
    """
            
    def getScreenPixels(self, ale):
        grayScreen = ale.getScreenGrayscale()
        resized = cv2.resize(grayScreen, (84, 84))
        return resized
    
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
        
    def getActionFromModel(self, historyBuffer, totalStep):
        if totalStep <= 10**6:
            self.greedyEpsilon = 1.0 - 0.9 / 10**6 * totalStep

        if random.random() < self.greedyEpsilon:
            return random.randrange(0, len(self.legalActions))
        else:
            actionValues = self.modelRunner.predict(historyBuffer)
            actionIndex = np.argmax(actionValues)
            return actionIndex
    
    def clipReward(self, reward):
            if reward > 0:
                return 1
            elif reward < 0:
                return -1
            else:
                return 0
        
    def gogo(self):
        ale = ALEInterface()
        
        max_frames_per_episode = ale.getInt("max_num_frames_per_episode");
        ale.setInt("random_seed",123)
        
        random_seed = ale.getInt("random_seed")
        print("random_seed: " + str(random_seed))

        if self.settings['SHOW_SCREEN']:
            ale.setBool('display_screen', True)
            
        ale.loadROM(sys.argv[1])
        self.legalActions = ale.getMinimalActionSet()
        print self.legalActions
        
        (self.screen_width,self.screen_height) = ale.getScreenDims()
        print("width/height: " +str(self.screen_width) + "/" + str(self.screen_height))
        
        (display_width,display_height) = (1024,420)
        

        self.modelRunner = ModelRunner(
                                    self.settings, 
                                    maxActionNo = len(self.legalActions),
                                    replayMemory = self.replayMemory
                                    )
        
        ram_size = ale.getRAMSize()
        ram = np.zeros((ram_size),dtype=np.uint8)
        action = 0
        
        for epoch in range(1, self.settings['MAX_EPOCH'] + 1):
            epochTotalReward = 0
            episodeTotalReward = 0
            epochStartTime = time.time()
            episodeStartTime = time.time()
            ale.reset_game()
            episode = 1
            rewardSum = 0
            historyBuffer = np.zeros((settings['TRAIN_BATCH_SIZE'], 
                                      settings['SCREEN_HISTORY'],
                                      settings['SCREEN_HEIGHT'], 
                                      settings['SCREEN_WIDTH']))
             
            for stepNo in range(self.settings['EPOCH_STEP']):
                episodeStep = ale.getEpisodeFrameNumber()
                totalStep = ale.getFrameNumber()

                actionIndex = self.getActionFromModel(historyBuffer, totalStep)
                action = self.legalActions[actionIndex]
                    
                reward = ale.act(action);
                clippedReward = self.clipReward(reward)
                rewardSum += clippedReward
                episodeTotalReward += clippedReward
                epochTotalReward += clippedReward
                
                state = self.getScreenPixels(ale)

                self.replayMemory.add(actionIndex, rewardSum, state, ale.game_over())
                self.modelRunner.train()
                
                historyBuffer[0, :-1] = historyBuffer[0, 1:]    
                historyBuffer[0, -1] = state
                
                rewardSum = 0    
            
                if(ale.game_over()):
                    print "Step Number: %s, Elapsed: %.1fs" % (stepNo, (time.time() - episodeStartTime))
                    print("Episode " + str(episode) + " ended with score: " + str(episodeTotalReward))
                    
                    ale.reset_game()
                    episodeStartTime = time.time()
                    
                    episode = episode + 1
                    episodeTotalReward = 0
                    historyBuffer.fill(0)
                    
                for skipFrame in range(self.settings['SKIP_SCREEN']):
                    reward = ale.act(action);
                    clippedReward = self.clipReward(reward)
                    rewardSum += clippedReward
                    episodeTotalReward += clippedReward
                    epochTotalReward += clippedReward
                    
                 
            print "[ Epoch %s ] ended with avg reward: %.1f. elapsed: %.0fs" % \
                  (epoch, float(epochTotalReward) / episode, 
                   time.time() - epochStartTime)
                
                
        self.modelRunner.finishTrain()
        
class DebugInput(threading.Thread):
    def __init__(self, player):
        threading.Thread.__init__(self)
        self.player = player
        self.running = True
    
    def run(self):
        while (self.running):
            input = raw_input('')
            if input == 'd':
                if player.settings['SHOW_SCREEN']:
                    player.settings['SHOW_SCREEN'] = False
                else:
                    player.settings['SHOW_SCREEN'] = True
                print 'settings[\'SHOW_SCREEN\'] : %s' % player.settings['SHOW_SCREEN']
                
    def finish(self):
        self.running = False
        
                
    
if __name__ == '__main__':    
    if(len(sys.argv) < 2):
        print("Usage ./ale_python_test_pygame_player.py <ROM_FILE_NAME>")
        sys.exit()
    
    settings = {}

    settings['SHOW_SCREEN'] = False
    settings['USE_KEYBOARD'] = False
    settings['SOLVER_PROTOTXT'] = 'models/solver.prototxt'
    settings['TARGET_PROTOTXT'] = 'models/target.prototxt'
    settings['TRAIN_BATCH_SIZE'] = 32
    settings['MAX_REPLAY_MEMORY'] = 1000000
    settings['MAX_EPOCH'] = 200
    settings['EPOCH_STEP'] = 250000
    settings['DISCOUNT_FACTOR'] = 0.99
    settings['UPDATE_STEP'] = 2000
    settings['SKIP_SCREEN'] = 3
    settings['SCREEN_WIDTH'] = 84
    settings['SCREEN_HEIGHT'] = 84
    settings['SCREEN_HISTORY'] = 4
    
    player = DeepRLPlayer(settings)
    player.gogo()
