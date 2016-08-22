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
import multiprocessing
import cv2
import pickle
import threading
import time
import util
#from model_runner import ModelRunner
from model_runner_neon import ModelRunnerNeon
from replay_memory import ReplayMemory

class DeepRLPlayer:
    def __init__(self, settings):
        self.settings = settings
        self.grayPixels = np.zeros((84, 84), np.float)
        self.zeroHistory = []
        for i in range(4):
            self.zeroHistory.append(np.zeros((84, 84), dtype=np.uint8))

        #self.sendQueue = multiprocessing.Queue()
        
        self.batchDimension = (settings['TRAIN_BATCH_SIZE'], 
                                      settings['SCREEN_HISTORY'],
                                      settings['SCREEN_HEIGHT'], 
                                      settings['SCREEN_WIDTH'])

        self.historyBuffer = np.zeros(self.batchDimension, dtype=np.float32)
        self.trainStep = 0
        self.epochDone = 0
        
        if 'PLAY' not in self.settings:
            util.Logger('output')
        
        if os.path.exists('output') == False:
            os.makedirs('output')
        if os.path.exists('snapshot') == False:
            os.makedirs('snapshot')
            
        gameFolder = settings['ROM'].split('/')[-1]
        if '.' in gameFolder:
            gameFolder = gameFolder.split('.')[0]
        self.snapshotFolder = 'snapshot/' + gameFolder
        if os.path.exists(self.snapshotFolder) == False:
            os.makedirs(self.snapshotFolder)
        
        self.printEnv()
        
        self.initializeAle()
        self.initializeReplayMemory()
        self.initializeModel()

        self.debug = False
        DebugInput(self).start()
    
    def initializeAle(self):
        self.ale = ALEInterface()
        
        max_frames_per_episode = self.ale.getInt("max_num_frames_per_episode");
        self.ale.setInt("random_seed",123)
        
        random_seed = self.ale.getInt("random_seed")
        print("random_seed: " + str(random_seed))

        if self.settings['SHOW_SCREEN'] or 'PLAY' in self.settings:
            self.ale.setBool('display_screen', True)
            
        self.ale.setInt('frame_skip', settings['FRAME_SKIP'])
        self.ale.setFloat('repeat_action_probability', 0)
        self.ale.setBool('color_averaging', True)
        
        self.ale.loadROM(settings['ROM'])
        self.legalActions = self.ale.getMinimalActionSet()
        print 'legalActions: %s' % self.legalActions
        
        (self.screen_width,self.screen_height) = self.ale.getScreenDims()
        print("width/height: " +str(self.screen_width) + "/" + str(self.screen_height))
        
        (display_width,display_height) = (1024,420)
        
        ram_size = self.ale.getRAMSize()
        ram = np.zeros((ram_size),dtype=np.uint8)    

    def initializeReplayMemory(self):
        self.replayMemory = ReplayMemory(self.settings['MAX_REPLAY_MEMORY'], self.settings)
        
    def initializeModel(self):
        #self.modelRunner = ModelRunner(
        self.modelRunner = ModelRunnerNeon(
                                    self.settings, 
                                    maxActionNo = len(self.legalActions),
                                    batchDimension = self.batchDimension,
                                    snapshotFolder = self.snapshotFolder
                                    )
        
    def getScreenPixels(self):
        grayScreen = self.ale.getScreenGrayscale()
        resized = cv2.resize(grayScreen, (84, 84))
        return resized
    
    def getActionFromModel(self, mode):
        if mode == 'TEST':
            greedyEpsilon = self.settings['TEST_EPSILON']
        else:
            if self.trainStep * 4 <= 10**6:
                greedyEpsilon = 1.0 - 0.9 / 10**6 * self.trainStep * 4
            else:
                greedyEpsilon = 0.1
             
        if random.random() < greedyEpsilon:
            return random.randrange(0, len(self.legalActions)), greedyEpsilon, 'random'
        else:
            actionValues = self.modelRunner.predict(self.historyBuffer)
            actionIndex = np.argmax(actionValues)
            return actionIndex, greedyEpsilon, 'trained'
        
    def printEnv(self):
        print 'Start time: %s' % time.strftime('%Y.%m.%d-%H:%M:%S')
        print '[ Running Environment ]'
        for key in self.settings.keys():
            print '%s: \t%s' % (key, self.settings[key])
        
    def resetGame(self):
        self.historyBuffer.fill(0)
        self.ale.reset_game()
        for r in range(random.randint(4, 30)):
            self.ale.act(0)
            state = self.getScreenPixels()
            self.addToHistoryBuffer(state)

    def addToHistoryBuffer(self, state):
        self.historyBuffer[0, :-1] = self.historyBuffer[0, 1:]
        self.historyBuffer[0, -1] = state

    def generateReplayMemory(self, count):
        print 'Generating %s replay memory' % count
        self.resetGame()
        for i in range(count):
            actionIndex, greedyEpsilon, type = self.getActionFromModel('TRAIN')
            action = self.legalActions[actionIndex]
            
            reward = self.ale.act(action)
            state = self.getScreenPixels()

            self.replayMemory.add(actionIndex, reward, state, self.ale.game_over())
                
            self.addToHistoryBuffer(state)
                
            if(self.ale.game_over()):
                self.resetGame()
        
    def test(self, epoch):
        episode = 0
        totalReward = 0
        testStartTime = time.time()
        self.resetGame()
        
        for stepNo in range(self.settings['TEST_STEP']):
            actionIndex, greedyEpsilon, actionType = self.getActionFromModel('TEST')
            action = self.legalActions[actionIndex]
            
            if (self.debug):
                print 'epsilon : %.2f, action : %s, %s' % (greedyEpsilon, action, actionType)
                
            reward = self.ale.act(action)
            totalReward += reward
            
            state = self.getScreenPixels()

            self.addToHistoryBuffer(state)
            
            if(self.ale.game_over()):
                episode += 1
                self.resetGame()
                 
        print "[ Test  %s ] avg score: %.1f. elapsed: %.0fs. last e: %.2f" % \
              (epoch, float(totalReward) / episode, 
               time.time() - testStartTime,
               greedyEpsilon)
                  
    def train(self, replayMemoryNo=None):
        if replayMemoryNo == None:
            replayMemoryNo = self.settings['TRAIN_START']
        self.generateReplayMemory(replayMemoryNo)
        
        print 'Start training'
        
        for epoch in range(self.epochDone + 1, self.settings['MAX_EPOCH'] + 1):
            epochTotalReward = 0
            episodeTotalReward = 0
            epochStartTime = time.time()
            episodeStartTime = time.time()
            self.resetGame()
            episode = 1

            for stepNo in range(self.settings['EPOCH_STEP']):
                actionIndex, greedyEpsilon, type = self.getActionFromModel('TRAIN')
                action = self.legalActions[actionIndex]
                
                if (self.debug):
                    print 'epsilon : %.2f, action : %s, %s' % (greedyEpsilon, action, type)
                    
                reward = self.ale.act(action)
                episodeTotalReward += reward
                epochTotalReward += reward
                
                state = self.getScreenPixels()

                self.replayMemory.add(actionIndex, reward, state, self.ale.game_over())
                    
                if stepNo % self.settings['TRAIN_STEP'] == 0:
                    minibatch = self.replayMemory.getMinibatch()
                    self.modelRunner.train(minibatch)
                    self.trainStep += 1
                
                    if self.trainStep % self.settings['SAVE_STEP'] == 0:
                        self.save()
                     
                self.addToHistoryBuffer(state)
                
                if self.ale.game_over():
                    if episode % 50 == 0:
                        print "Ep %s, score: %s, step: %s, elapsed: %.1fs, avg: %.1f, train=%s" % (
                                                                                episode, episodeTotalReward,
                                                                                stepNo, (time.time() - episodeStartTime),
                                                                                float(epochTotalReward) / episode,
                                                                                self.trainStep)
                    episodeStartTime = time.time()
                    
                    episode += 1
                    episodeTotalReward = 0
                    
                    self.resetGame()
                
            print "[ Train %s ] avg score: %.1f. elapsed: %.0fs. last e: %.2f" % \
                  (epoch, float(epochTotalReward) / episode, 
                   time.time() - epochStartTime,
                   greedyEpsilon)
                    
            self.epochDone = epoch
             
            # Test once every epoch
            self.test(epoch)
                
                
        self.modelRunner.finishTrain()
    
    def save(self):
        fileName = '%s/dqn_%s' % (self.snapshotFolder, self.trainStep)
        with open(fileName + '.pickle', 'wb') as f:
            pickle.dump(self, f)
            self.modelRunner.save(fileName + '.weight')
            #print '%s dumped' % fileName
        
    def __getstate__(self):
        self.replayMemoryNo = self.replayMemory.count
        d = dict(self.__dict__)
        del d['ale']
        del d['replayMemory']
        del d['modelRunner']
        return d
        
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
        

def trainOrPlay(settings):
    player = DeepRLPlayer(settings)
    if 'PLAY' in settings:
        player.test(0)
    else:
        player.trainStep = 0
        player.train()
    
def retrain(saveFile):
    print 'Resume trainig: %s' % saveFile

    with open(saveFile + '.pickle') as f:
        player = pickle.load(f)
        player.initializeAle()
        player.initializeReplayMemory()
        player.initializeModel()
        player.modelRunner.load(saveFile + '.weight')
        player.train(player.replayMemoryNo)
        
if __name__ == '__main__':    
    settings = {}

    #game = 'breakout'
    #game = 'space_invaders'
    #game = 'enduro'
    #game = 'kung_fu_master'
    game = 'krull'

    settings['ROM'] = '/media/big/download/roms/%s.bin' % game    
    settings['FRAME_SKIP'] = 4

    #settings['SHOW_SCREEN'] = True
    settings['SHOW_SCREEN'] = False
    settings['USE_KEYBOARD'] = False
    settings['SOLVER_PROTOTXT'] = 'models/solver2.prototxt'
    settings['TARGET_PROTOTXT'] = 'models/target2.prototxt'
    settings['TRAIN_BATCH_SIZE'] = 32
    settings['MAX_REPLAY_MEMORY'] = 1000000
    settings['MAX_EPOCH'] = 200
    settings['EPOCH_STEP'] = 250000
    settings['DISCOUNT_FACTOR'] = 0.99
    settings['UPDATE_STEP'] = 10000               # Copy train network into target network every this train step
    settings['TRAIN_START'] = 50000                 # Start training after filling this replay memory size
    settings['TRAIN_STEP'] = 4                            # Train every this screen step
    settings['TEST_STEP'] = 50000                      # Test for this number of steps
    settings['TEST_EPSILON'] = 0.05                   # Greed epsilon for test
    settings['SAVE_STEP'] = 50000                     # Save result every this training step
    settings['SCREEN_WIDTH'] = 84
    settings['SCREEN_HEIGHT'] = 84
    settings['SCREEN_HISTORY'] = 4

    settings['LEARNING_RATE'] = 0.00025
    settings['RMS_DECAY'] = 0.95
    
    #settings['PLAY'] = 'snapshot/dqn_neon_4350000.prm'    
    #settings['PLAY'] = 'snapshot/breakout/dqn_neon_1050000.prm'
    #settings['PLAY'] = 'snapshot/breakout/dqn_neon_3100000.prm'
    #settings['PLAY'] = 'snapshot/dqn_neon_3600000.prm'
    #settings['PLAY'] = 'snapshot/kung_fu_master/dqn_neon_2100000.prm'
    
    #trainOrPlay(settings)
    retrain('snapshot/%s/%s' % (game, 'dqn_850000'))
