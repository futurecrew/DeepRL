#!/usr/bin/env python

# ale_python_test_pygame_player.py
# Author: Ben Goodrich
#
# This modified ale_python_test_pygame.py to provide a fully interactive experience allowing the game_player
# to play. RAM Contents, current action, and reward are also displayed.
# keys are:
# arrow keys -> up/down/left/right
# z -> fire button
import sys
import os
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
        
        util.Logger('output')
        
        if os.path.exists('output') == False:
            os.makedirs('output')
        if os.path.exists('snapshot') == False:
            os.makedirs('snapshot')
        
        # DJDJ
        self.debug = False        
        DebugInput(self).start()
        

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
            return random.randrange(0, len(self.legalActions)), 'random'
        else:
            actionValues = self.modelRunner.predict(historyBuffer)
            actionIndex = np.argmax(actionValues)
            return actionIndex, 'trained'
        
    def printEnv(self):
        print '[ Running Environment ]'
        for key in self.settings.keys():
            print '%s: \t%s' % (key, self.settings[key])
        
    def gogo(self):
        self.printEnv()
        
        ale = ALEInterface()
        
        max_frames_per_episode = ale.getInt("max_num_frames_per_episode");
        ale.setInt("random_seed",123)
        
        random_seed = ale.getInt("random_seed")
        print("random_seed: " + str(random_seed))

        if self.settings['SHOW_SCREEN'] or 'PLAY' in self.settings:
            ale.setBool('display_screen', True)
            
        # DJDJ
        ale.setInt('frame_skip', 4)
        
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
        totalStep = 0
        
        # DJDJ
        if 'RESORE' in settings:
            totalStep = 10**6 - 1
        elif 'PLAY' in settings:
            totalStep = 10**6 - 1
            #self.greedyEpsilon = 0.0
        
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
                                      settings['SCREEN_WIDTH']), dtype=np.float32)

            for stepNo in range(self.settings['EPOCH_STEP']):
                totalStep += 1

                actionIndex, type = self.getActionFromModel(historyBuffer, totalStep)
                action = self.legalActions[actionIndex]
                
                if (self.debug):
                    print 'epsilon : %.2f, action : %s, %s' % (self.greedyEpsilon, action, type)
                    
                reward = ale.act(action);
                rewardSum += reward
                episodeTotalReward += reward
                epochTotalReward += reward
                
                state = self.getScreenPixels(ale)

                if 'PLAY' not in settings:
                    self.replayMemory.add(actionIndex, rewardSum, state, ale.game_over())
                    
                    # DJDJ
                    if totalStep % 4 == 0:
                        self.modelRunner.train()
                
                historyBuffer[0, :-1] = historyBuffer[0, 1:]
                # DJDJ
                historyBuffer[0, -1] = state
                #historyBuffer[0, -1] = state / 255.0
                
                rewardSum = 0    
            
                if(ale.game_over()):
                    print "Episode %s : score: %s, step: %s, elapsed: %.1fs, avg: %.2f, total step=%s" % (
                                                                                episode, episodeTotalReward,
                                                                                stepNo, (time.time() - episodeStartTime),
                                                                                float(epochTotalReward) / episode,
                                                                                totalStep)
                    ale.reset_game()
                    episodeStartTime = time.time()
                    
                    episode += 1
                    episodeTotalReward = 0
                    historyBuffer.fill(0)
                    
                # DJDJ
                """
                for skipFrame in range(self.settings['SKIP_SCREEN']):
                    reward = ale.act(action);
                    rewardSum += reward
                    episodeTotalReward += reward
                    epochTotalReward += reward
                """
                 
            print "[ Epoch %s ] ended with avg score: %.1f. elapsed: %.0fs. last e: %.2f" % \
                  (epoch, float(epochTotalReward) / episode, 
                   time.time() - epochStartTime,
                   self.greedyEpsilon)
                
                
        self.modelRunner.finishTrain()
        
class DebugInput(threading.Thread):
    def __init__(self, player):
        threading.Thread.__init__(self)
        self.player = player
        self.running = True
    
    def run(self):
        time.sleep(2)
        while (self.running):
            input = raw_input('')
            if input == 'd':
                self.player.debug = not self.player.debug
                print 'Debug mode : %s' % self.player.debug
                
    def finish(self):
        self.running = False
        
                
    
if __name__ == '__main__':    
    if(len(sys.argv) < 2):
        print("Usage ./ale_python_test_pygame_player.py <ROM_FILE_NAME>")
        sys.exit()
    
    settings = {}

    #settings['SHOW_SCREEN'] = True
    settings['SHOW_SCREEN'] = False
    settings['USE_KEYBOARD'] = False
    settings['SOLVER_PROTOTXT'] = 'models/solver2.prototxt'
    settings['TARGET_PROTOTXT'] = 'models/target2.prototxt'
    
    #settings['RESTORE'] = 'snapshot/dqn_iter_1100000.solverstate'
    settings['PLAY'] = 'snapshot/dqn_iter_700000.caffemodel'    
    
    settings['TRAIN_BATCH_SIZE'] = 32
    # DJDJ
    #settings['MAX_REPLAY_MEMORY'] = 1000000
    settings['MAX_REPLAY_MEMORY'] = 900000
    settings['MAX_EPOCH'] = 200
    settings['EPOCH_STEP'] = 250000
    settings['DISCOUNT_FACTOR'] = 0.99
    settings['UPDATE_STEP'] = 10000
    settings['SKIP_SCREEN'] = 3
    settings['SCREEN_WIDTH'] = 84
    settings['SCREEN_HEIGHT'] = 84
    settings['SCREEN_HISTORY'] = 4
    
    player = DeepRLPlayer(settings)
    player.gogo()
