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
import matplotlib.pyplot as plt
from scipy.misc import imresize
import pickle
import threading
import time
import util
from model_runner import ModelRunner

class DeepRLPlayer:
    def __init__(self, settings):
        self.settings = settings
        self.grayPixels = np.zeros((84, 84), np.float)
        self.zeroHistory = []
        for i in range(4):
            self.zeroHistory.append(np.zeros((84, 84), dtype=np.uint8))

        self.greedyEpsilon = 1.0
        self.sendQueue = multiprocessing.Queue()
        
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
        
    def getActionFromModel(self, stateHistoryStack, totalStep):
        
        # DJDJ
        if totalStep <= 10**6:
            self.greedyEpsilon = 1.0 - 0.9 / 10**6 * totalStep

        #if totalStep <= 4 * 10**5:
        #    self.greedyEpsilon = 1.0 - 0.9 / (4 * 10**5) * totalStep

        if random.random() < self.greedyEpsilon:
            return random.randrange(0, len(self.legalActions))
        else:
            actionValues = self.modelRunner.test(stateHistoryStack)
            actionIndex = np.argmax(actionValues)
            return actionIndex
    
        
    def gogo(self):
        ale = ALEInterface()
        
        max_frames_per_episode = ale.getInt("max_num_frames_per_episode");
        ale.set("random_seed",123)
        
        random_seed = ale.getInt("random_seed")
        print("random_seed: " + str(random_seed))
        
        ale.loadROM(sys.argv[1])
        self.legalActions = ale.getMinimalActionSet()
        print self.legalActions
        
        (self.screen_width,self.screen_height) = ale.getScreenDims()
        print("width/height: " +str(self.screen_width) + "/" + str(self.screen_height))
        
        (display_width,display_height) = (1024,420)
        

        self.modelRunner = ModelRunner(self.sendQueue, 
                                      trainBatchSize = settings['TRAIN_BATCH_SIZE'],
                                      solverPrototxt = settings['SOLVER_PROTOTXT'], 
                                      testPrototxt = settings['TEST_PROTOTXT'],
                                      maxReplayMemory = settings['MAX_REPLAY_MEMORY'],
                                      discountFactor = settings['DISCOUNT_FACTOR'],
                                      updateStep = settings['UPDATE_STEP'],
                                      maxActionNo = len(self.legalActions),
                                      )
        
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
        
        while (episode < 1000000):
            gameOver = False
            total_reward = 0.0 
            action = 0
            actionIndex = 0
            rewardSum = 0
            newState = None
            stateHistory = []
            newStateHistory = []
            pickleNo = 0
            
            episodeStartTime = time.time()
            
            state = self.getScreenPixels(ale, screen, game_surface)

            while gameOver == False:            
                episodeStep = ale.getEpisodeFrameNumber()
                totalStep = ale.getFrameNumber()

                if episodeStep % self.settings['SKIP_SCREEN'] != 0:
                    reward = ale.act(action);
                else:
                    if len(stateHistory) == 4:
                        stateHistoryStack = np.reshape(stateHistory, (4, 84, 84))
                        actionIndex = self.getActionFromModel(stateHistoryStack, totalStep)
                        action = self.legalActions[actionIndex]
                    else:
                        action = 0
                        
                    reward = ale.act(action);
    
                    newState = self.getScreenPixels(ale, screen, game_surface)
                        
                    if len(stateHistory) == 4:
                        del stateHistory[0]
                    stateHistory.append(state)
                    
                    if len(newStateHistory) == 4:
                        del newStateHistory[0]
                    newStateHistory.append(newState)
    
    
                    """
                    if len(stateHistory) == 4 and len(newStateHistory) == 4:
                        stateHistoryStack = np.reshape(stateHistory, (4, 84, 84))
                        newStateHistoryStack = np.reshape(newStateHistory, (4, 84, 84))
    
                        # DJDJ
                        pickleNo += 1
                        if pickleNo == 1:
                            actionIndex = 3
                            rewardSum = 0
                            gameOver = False
                        elif pickleNo ==2:
                            actionIndex = 0
                            rewardSum = 1
                            gameOver = False
                        elif pickleNo == 3:
                            actionIndex = 3
                            rewardSum = 0
                            gameOver = True
                        
                        self.modelRunner.addData(stateHistoryStack, actionIndex, rewardSum, newStateHistoryStack, gameOver, episodeStep)
                        rewardSum = 0
                        
                        if pickleNo == 3:
                            self.modelRunner.train()
                            self.modelRunner.train()
                            self.modelRunner.train()
                            self.modelRunner.train()
                            self.modelRunner.train()
                            self.modelRunner.train()
                            self.modelRunner.train()
                            self.modelRunner.train()
                    """                
                    if len(stateHistory) == 4 and len(newStateHistory) == 4:
                        dataList = []
                        newDataList = []
                        for j in range(4):
                            dataList.append(stateHistory[j])
                            newDataList.append(newStateHistory[j])                        
    
                        self.modelRunner.addData(dataList, actionIndex, rewardSum, newDataList, False, episodeStep)
                        self.modelRunner.train()
                        rewardSum = 0
    
                    state = newState                   
                    
                            
                if reward > 0:
                    rewardSum += 1
                elif reward < 0:
                    rewardSum -= 1
                
                ale.getRAM(ram)
                
                if self.settings['SHOW_SCREEN']:
                    self.displayInfo(screen, ram, action, total_reward)
                    pygame.display.flip()
            
                total_reward += reward
                
                #delay to 60fps
                #clock.tick(60.)
            
                if(ale.game_over()):
                    if len(stateHistory) == 4 and len(newStateHistory) == 4:
                        dataList = []
                        for j in range(4):
                            dataList.append(stateHistory[j])
                        self.modelRunner.addData(dataList, actionIndex, rewardSum, self.zeroHistory, True, episodeStep)

                    episode_frame_number = ale.getEpisodeFrameNumber()
                    frame_number = ale.getFrameNumber()
                    print "Frame Number: %s, Episode Frame Number: %s, Elapsed: %.0fs" % (frame_number, episode_frame_number, (time.time() - episodeStartTime))
                    print("Episode " + str(episode) + " ended with score: " + str(total_reward))
                    
                    ale.reset_game()
                    
                    episode = episode + 1
                    gameOver = True

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

    settings['SHOW_SCREEN'] = True
    settings['USE_KEYBOARD'] = False
    settings['SOLVER_PROTOTXT'] = 'models/solver.prototxt'
    settings['TEST_PROTOTXT'] = 'models/test.prototxt'
    settings['TRAIN_BATCH_SIZE'] = 32
    settings['MAX_REPLAY_MEMORY'] = 250000
    settings['DISCOUNT_FACTOR'] = 0.99
    settings['UPDATE_STEP'] = 2000
    settings['SKIP_SCREEN'] = 4
    
    player = DeepRLPlayer(settings)
    player.gogo()
