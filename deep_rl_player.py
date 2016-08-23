#!/usr/bin/env python

# deep_rl_player.py
# Author: Daejoong Kim
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
    def __init__(self, settings, playFile=None):
        self.settings = settings
        self.grayPixels = np.zeros((84, 84), np.float)
        self.zeroHistory = []
        for i in range(4):
            self.zeroHistory.append(np.zeros((84, 84), dtype=np.uint8))

        #self.sendQueue = multiprocessing.Queue()
        
        self.batchDimension = (self.settings['train_batch_size'], 
                                      self.settings['screen_history'],
                                      self.settings['screen_height'], 
                                      self.settings['screen_width'])

        self.historyBuffer = np.zeros(self.batchDimension, dtype=np.float32)
        self.trainStep = 0
        self.epochDone = 0
        
        if playFile is None:
            util.Logger('output', settings['game'])
        
        if os.path.exists('output') == False:
            os.makedirs('output')
        if os.path.exists('snapshot') == False:
            os.makedirs('snapshot')
            
        gameFolder = self.settings['rom'].split('/')[-1]
        if '.' in gameFolder:
            gameFolder = gameFolder.split('.')[0]
        self.snapshotFolder = 'snapshot/' + gameFolder
        if os.path.exists(self.snapshotFolder) == False:
            os.makedirs(self.snapshotFolder)
        
        self.printEnv()
        
        if self.settings['show_screen'] or playFile is not None:
            displayScreen = True
        else:
            displayScreen = False
        
        self.initializeAle(displayScreen)
        self.initializeReplayMemory()
        self.initializeModel()

        self.debug = False
        DebugInput(self).start()
    
    def initializeAle(self, displayScreen=False):
        self.ale = ALEInterface()
        
        max_frames_per_episode = self.ale.getInt("max_num_frames_per_episode");
        self.ale.setInt("random_seed",123)
        
        random_seed = self.ale.getInt("random_seed")
        print("random_seed: " + str(random_seed))

        if displayScreen:
            self.ale.setBool('display_screen', True)
            
        # DJDJ
        #self.ale.setInt('frame_skip', self.settings['frame_repeat'])
        self.ale.setFloat('repeat_action_probability', 0)
        self.ale.setBool('color_averaging', True)
        
        self.ale.loadROM(self.settings['rom'])
        self.legalActions = self.ale.getMinimalActionSet()
        print 'legalActions: %s' % self.legalActions
        
        (self.screen_width,self.screen_height) = self.ale.getScreenDims()
        print("width/height: " +str(self.screen_width) + "/" + str(self.screen_height))
        
        (display_width,display_height) = (1024,420)
        
        ram_size = self.ale.getRAMSize()
        ram = np.zeros((ram_size),dtype=np.uint8)    

    def initializeReplayMemory(self):
        self.replayMemory = ReplayMemory(self.settings['max_replay_memory'], self.settings)
        
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
            greedyEpsilon = self.settings['test_epsilon']
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

    def doActions(self, actionIndex):
        action = self.legalActions[actionIndex]
        
        #if (self.debug):
        #    print 'epsilon : %.2f, action : %s, %s' % (greedyEpsilon, action, type)
            
        reward = 0
        lostLife = False 
        lives = self.ale.lives()
        for f in range(self.settings['frame_repeat']):
            reward += self.ale.act(action)
            gameOver = self.ale.game_over()
            if self.ale.lives() < lives or gameOver:
                lostLife = True
                break
        state = self.getScreenPixels()
        
        return reward, state, lostLife, gameOver

    def generateReplayMemory(self, count):
        print 'Generating %s replay memory' % count
        self.resetGame()
        for i in range(count):
            actionIndex, greedyEpsilon, type = self.getActionFromModel('TRAIN')
            
            reward, state, lostLife, gameOver = self.doActions(actionIndex)

            self.replayMemory.add(actionIndex, reward, state, lostLife)
                
            self.addToHistoryBuffer(state)
                
            if(gameOver):
                self.resetGame()
        
    def test(self, epoch):
        episode = 1
        totalReward = 0
        testStartTime = time.time()
        self.resetGame()
        
        for stepNo in range(self.settings['test_step']):
            actionIndex, greedyEpsilon, actionType = self.getActionFromModel('TEST')
                
            reward, state, lostLife, gameOver = self.doActions(actionIndex)
                
            totalReward += reward

            self.addToHistoryBuffer(state)
            
            if(gameOver):
                episode += 1
                self.resetGame()
                 
        print "[ Test  %s ] avg score: %.1f. elapsed: %.0fs. last e: %.2f" % \
              (epoch, float(totalReward) / episode, 
               time.time() - testStartTime,
               greedyEpsilon)
                  
    def train(self, replayMemoryNo=None):
        if replayMemoryNo == None:
            replayMemoryNo = self.settings['train_start']
        self.generateReplayMemory(replayMemoryNo)
        
        print 'Start training'
        
        for epoch in range(self.epochDone + 1, self.settings['max_epoch'] + 1):
            epochTotalReward = 0
            episodeTotalReward = 0
            epochStartTime = time.time()
            episodeStartTime = time.time()
            self.resetGame()
            episode = 1

            for stepNo in range(self.settings['epoch_step']):
                actionIndex, greedyEpsilon, type = self.getActionFromModel('TRAIN')
                
                reward, state, lostLife, gameOver = self.doActions(actionIndex)

                episodeTotalReward += reward
                epochTotalReward += reward

                self.replayMemory.add(actionIndex, reward, state, lostLife)
                    
                if stepNo % self.settings['train_step'] == 0:
                    minibatch = self.replayMemory.getMinibatch()
                    self.modelRunner.train(minibatch)
                    self.trainStep += 1
                
                    if self.trainStep % self.settings['save_step'] == 0:
                        self.save()
                     
                self.addToHistoryBuffer(state)
                
                if gameOver:
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
        

def trainOrPlay(settings, playFile=None):
    player = DeepRLPlayer(settings, playFile)
    if playFile is not None:
        player.modelRunner.load(playFile + '.weight')
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

    #settings['game'] = 'breakout'
    #settings['game'] = 'space_invaders'
    #settings['game'] = 'enduro'
    #settings['game'] = 'kung_fu_master'
    #settings['game'] = 'krull'
    settings['game'] = 'seaquest'

    settings['rom'] = '/media/big/download/roms/%s.bin' % settings['game']    
    settings['frame_repeat'] = 4

    #settings['show_screen'] = True
    settings['show_screen'] = False
    settings['use_keyboard'] = False
    settings['solver_prototxt'] = 'models/solver2.prototxt'
    settings['target_prototxt'] = 'models/target2.prototxt'
    settings['train_batch_size'] = 32
    settings['max_replay_memory'] = 1000000
    settings['max_epoch'] = 200
    settings['epoch_step'] = 250000
    settings['discount_factor'] = 0.99
    settings['update_step'] = 10000               # Copy train network into target network every this train step
    settings['train_start'] = 50000                   # Start training after filling this replay memory size
    settings['train_step'] = 4                            # Train every this screen step
    settings['test_step'] = 50000                     # Test for this number of steps
    settings['test_epsilon'] = 0.05                   # Greed epsilon for test
    settings['save_step'] = 50000                    # Save result every this training step
    settings['screen_width'] = 84
    settings['screen_height'] = 84
    settings['screen_history'] = 4
    settings['learning_rate'] = 0.00025
    settings['rms_decay'] = 0.95
    
    dataFile = None
    
    #dataFile = 'snapshot/dqn_neon_4350000.prm'    
    #dataFile = 'snapshot/breakout/dqn_neon_1050000.prm'
    #dataFile = 'snapshot/breakout/dqn_neon_3100000.prm'
    #dataFile = 'snapshot/dqn_neon_3600000.prm'
    #dataFile = 'snapshot/kung_fu_master/dqn_neon_2100000.prm'
    #dataFile = 'snapshot/%s/%s' % (settings['game'], 'dqn_850000')
    
    trainOrPlay(settings, dataFile)
    #retrain(dataFile)
