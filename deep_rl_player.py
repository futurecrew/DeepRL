#!/usr/bin/env python

# deep_rl_player.py
# Author: Daejoong Kim
import os
from ale_python_interface import ALEInterface
import numpy as np
import random
import multiprocessing
import scipy.ndimage as ndimage
import cv2
import pickle
import threading
import time
import util
import png
from multiprocessing import Process, Queue
#from model_runner import ModelRunner
from replay_memory import ReplayMemory
from sampling_manager import SamplingManager

class DeepRLPlayer:
    def __init__(self, settings, playFile=None, processNo=None, queueList=None, sharedData=None):
        self.settings = settings
        self.playFile = playFile
        self.currentState = None
        self.processNo = processNo
        self.queueList = queueList
        self.sharedData = sharedData
        self.grayPixels = np.zeros((84, 84), np.float)
        self.zeroHistory = []
        for i in range(4):
            self.zeroHistory.append(np.zeros((84, 84), dtype=np.uint8))

        #self.sendQueue = multiprocessing.Queue()
        
        self.batchDimension = (self.settings['train_batch_size'], 
                                      self.settings['screen_history'],
                                      self.settings['screen_height'], 
                                      self.settings['screen_width'])

        self.frameNumber = 0
        self.trainStep = 0
        self.epochDone = 0
        self.trainStart = time.strftime('%Y%m%d_%H%M%S')

        if self.playFile is None:
            logFile="output/%s_%s.log" % (settings['game'], self.trainStart)            
            util.Logger(logFile)
        
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
        self.initializePost()
        
    def initializePost(self):
        """ initialization that should be run on __init__() or after deserialization """
        if self.settings['show_screen'] or self.playFile is not None:
            displayScreen = True
        else:
            displayScreen = False

        self.initializeAle(displayScreen)
        self.initializeModel()
        self.initializeReplayMemory()
        
        """
        if self.processNo != None:
            self.initializeQueue()
        else:
            DebugInput(self).start()
        """
        DebugInput(self).start()
        self.debug = False
        
    def initializeAle(self, displayScreen=False):
        self.ale = ALEInterface()
        
        #max_frames_per_episode = self.ale.getInt("max_num_frames_per_episode");
        self.ale.setInt("random_seed", random.randint(1, 1000))
        
        #random_seed = self.ale.getInt("random_seed")
        #print("random_seed: " + str(random_seed))

        if displayScreen:
            self.ale.setBool('display_screen', True)
            
        self.ale.setFloat('repeat_action_probability', 0)
        
        self.ale.loadROM(self.settings['rom'])
        self.legalActions = self.ale.getMinimalActionSet()
        print 'legalActions: %s' % self.legalActions
        
        (self.screen_width,self.screen_height) = self.ale.getScreenDims()
        print("width/height: " +str(self.screen_width) + "/" + str(self.screen_height))
        
        (display_width,display_height) = (1024,420)
        
        ram_size = self.ale.getRAMSize()
        ram = np.zeros((ram_size),dtype=np.uint8)    
        
    def initializeModel(self):
        #self.modelRunner = ModelRunner(
        if self.settings['backend'] == 'NEON':
            from model_runner_neon import ModelRunnerNeon
            self.modelRunner = ModelRunnerNeon(
                                    self.settings, 
                                    maxActionNo = len(self.legalActions),
                                    batchDimension = self.batchDimension
                                    )
        elif self.settings['backend'] == 'TF':
            from model_runner_tf import ModelRunnerTF
            self.modelRunner = ModelRunnerTF(
                                    self.settings, 
                                    maxActionNo = len(self.legalActions),
                                    batchDimension = self.batchDimension
                                    )
        else:
            print "settings['backend'] should be NEON or TF."

    def initializeReplayMemory(self):
        uniformReplayMemory = ReplayMemory(self.modelRunner.be,
                                     self.settings['use_gpu_replay_mem'],
                                     self.settings['max_replay_memory'], 
                                     self.settings['train_batch_size'],
                                     self.settings['screen_history'],
                                     self.settings['screen_width'],
                                     self.settings['screen_height'],
                                     self.settings['minibatch_random'])
        if self.settings['prioritized_replay'] == True:
            self.replayMemory = SamplingManager(uniformReplayMemory,
                                         self.settings['use_gpu_replay_mem'],
                                         self.settings['max_replay_memory'], 
                                         self.settings['train_batch_size'],
                                         self.settings['screen_history'],
                                         self.settings['prioritized_mode'],
                                         self.settings['sampling_alpha'],
                                         self.settings['sampling_beta'],
                                         self.settings['heap_sort_term'])
        else:
            self.replayMemory = uniformReplayMemory
                                                 
    def initializeQueue(self):
        self.queueManager = QueueManager(self, self.processNo, self.queueList)
        self.queueManager.start()
                                                         
    def getActionFromModel(self, mode):
        if mode == 'TEST':
            greedyEpsilon = self.settings['test_epsilon']
        else:
            minEpsilon = settings['train_min_epsilon']
            trainFrequency = self.settings['train_step']
            if self.trainStep * trainFrequency <= 10**6:
                greedyEpsilon = 1.0 - (1.0 - minEpsilon) / 10**6 * self.trainStep * trainFrequency
            else:
                greedyEpsilon = minEpsilon
             
        if random.random() < greedyEpsilon:
            return random.randrange(0, len(self.legalActions)), greedyEpsilon, 'random'
        else:
            actionValues = self.modelRunner.predict(self.modelRunner.historyBuffer)
            actionIndex = np.argmax(actionValues)
            return actionIndex, greedyEpsilon, 'trained'
        
    def printEnv(self):
        print 'Start time: %s' % time.strftime('%Y.%m.%d %H:%M:%S')
        print '[ Running Environment ]'
        for key in self.settings.keys():
            print '{} : '.format(key).ljust(30) + '{}'.format(self.settings[key])
        
    def resetGame(self):
        self.modelRunner.clearHistoryBuffer()
        self.ale.reset_game()
        self.currentState = None
        for _ in range(random.randint(4, 30)):
            reward, state, terminal, gameOver = self.doActions(0, 'TRAIN')
            self.modelRunner.addToHistoryBuffer(state)
    
    def saveScreenAsPNG(self, basefilename, screen, frameNumber):
        pngfile = open(basefilename + ('-%08d.png' % frameNumber), 'wb')
        pngWriter = png.Writer(screen.shape[1], screen.shape[0], greyscale=True)
        pngWriter.write(pngfile, screen)
        pngfile.close()

    def doActions(self, actionIndex, mode):
        action = self.legalActions[actionIndex]
        reward = 0
        terminal = False 
        lives = self.ale.lives()
        frameRepeat = self.settings['frame_repeat']

        if 'ndimage.zoom' in self.settings and self.settings['ndimage.zoom']:        
            state = self.ale.getScreenRGB()
            for f in range(frameRepeat):
                prevState = state
                reward += self.ale.act(action)
                state = self.ale.getScreenRGB()
                gameOver = self.ale.game_over()
                if self.ale.lives() < lives or gameOver:
                    terminal = True
                    if mode == 'TRAIN' and self.settings['lost_life_game_over'] == True:
                        gameOver = True
                    break
            maxState = np.maximum(prevState, state)
            
            screen = np.dot(maxState, np.array([.299, .587, .114])).astype(np.uint8)
            screen = ndimage.zoom(screen, (0.4, 0.525))
            screen.resize((84, 84))
            return reward, screen, terminal, gameOver
        else:
            if self.currentState is None:
                self.currentState = self.ale.getScreenGrayscale()
            for _ in range(frameRepeat):
                prevState = self.currentState
                reward += self.ale.act(action)
                self.currentState = self.ale.getScreenGrayscale()
                gameOver = self.ale.game_over()
                if self.ale.lives() < lives or gameOver:
                    terminal = True
                    if mode == 'TRAIN' and self.settings['lost_life_game_over'] == True:
                        gameOver = True
                    break
            maxState = np.maximum(prevState, self.currentState)
            resized = cv2.resize(maxState, (84, 84))
            return reward, resized, terminal, gameOver
    
    def generateReplayMemory(self, count):
        print 'Generating %s replay memory' % count
        startTime = time.time()
        self.resetGame()
        for _ in range(count):
            actionIndex, greedyEpsilon, type = self.getActionFromModel('TRAIN')
            reward, state, terminal, gameOver = self.doActions(actionIndex, 'TRAIN')
            self.replayMemory.add(actionIndex, reward, state, terminal)
            self.modelRunner.addToHistoryBuffer(state)
                
            if(gameOver):
                self.resetGame()
        
        print 'Generating replay memory took %.0f sec' % (time.time() - startTime)
        
    def test(self, epoch):
        episode = 0
        totalReward = 0
        testStartTime = time.time()
        self.resetGame()
        
        episodeReward = 0
        for stepNo in range(self.settings['test_step']):
            actionIndex, greedyEpsilon, actionType = self.getActionFromModel('TEST')
                
            reward, state, terminal, gameOver = self.doActions(actionIndex, 'TEST')
                
            episodeReward += reward

            self.modelRunner.addToHistoryBuffer(state)
            
            if(gameOver):
                episode += 1
                totalReward += episodeReward
                self.resetGame()
                episodeReward = 0
            
                if self.debug:
                    print "[ Test  %s ] %s steps, avg score: %.1f. ep: %d, elapsed: %.0fs. last e: %.3f" % \
                          (epoch, stepNo, float(totalReward) / episode, episode, 
                           time.time() - testStartTime,
                           greedyEpsilon)
        
        episode = max(episode, 1)          
        print "[ Test  %s ] avg score: %.1f. elapsed: %.0fs. last e: %.3f" % \
              (epoch, float(totalReward) / episode, 
               time.time() - testStartTime,
               greedyEpsilon)
                  
    def train(self, replayMemoryNo=None):
        if replayMemoryNo == None:
            replayMemoryNo = self.settings['train_start']
        if replayMemoryNo > 0:
            self.generateReplayMemory(replayMemoryNo)
        
        print 'Start training'
        startTime = time.time()
        
        for epoch in range(self.epochDone + 1, self.settings['max_epoch'] + 1):
            epochTotalReward = 0
            episodeTotalReward = 0
            epochStartTime = time.time()
            episodeStartTime = time.time()
            self.resetGame()
            episode = 1

            for stepNo in range(self.settings['epoch_step']):
                actionIndex, greedyEpsilon, type = self.getActionFromModel('TRAIN')
                
                reward, state, terminal, gameOver = self.doActions(actionIndex, 'TRAIN')

                episodeTotalReward += reward
                epochTotalReward += reward

                self.replayMemory.add(actionIndex, reward, state, terminal)
                    
                if stepNo % self.settings['train_step'] == 0:
                    minibatch = self.replayMemory.getMinibatch()
                    self.modelRunner.train(minibatch, self.replayMemory, self.debug)
                    self.trainStep += 1
                
                    if (self.trainStep % self.settings['save_step'] == 0) and (self.processNo == None or self.processNo == 0):
                        self.save()
                     
                self.modelRunner.addToHistoryBuffer(state)
                
                if gameOver:
                    if episode % 500 == 0:
                        print "Ep %s, score: %s, step: %s, elapsed: %.1fs, avg: %.1f, train=%s, t_elapsed: %.1fs" % (
                                                                                episode, episodeTotalReward,
                                                                                stepNo, (time.time() - episodeStartTime),
                                                                                float(epochTotalReward) / episode,
                                                                                self.trainStep,
                                                                                (time.time() - startTime))
                    episodeStartTime = time.time()
                    
                    episode += 1
                    episodeTotalReward = 0
                    
                    self.resetGame()
                    
                    if self.settings['multi_process_no'] > 1 and episode % self.settings['multi_process_copy_step']  == 0:
                        self.queueManager.sendParams()

                if stepNo > 0 and stepNo % self.settings['update_step'] == 0:
                    self.modelRunner.updateModel()
                
            print "[ Train %s ] avg score: %.1f. elapsed: %.0fs. last e: %.3f, train=%s" % \
                  (epoch, float(epochTotalReward) / episode, 
                   time.time() - epochStartTime,
                   greedyEpsilon, self.trainStep)
             
            # Test once every epoch
            if self.processNo == None or self.processNo == 0:
                self.test(epoch)
                    
            self.epochDone = epoch
                
                
        self.modelRunner.finishTrain()
    
    def save(self):
        timeSnapshotFolder = self.snapshotFolder + '/' + self.trainStart
        if os.path.exists(timeSnapshotFolder) == False:
            os.makedirs(timeSnapshotFolder)
        
        fileName = '%s/dqn_%s' % (timeSnapshotFolder, self.trainStep)
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
        del d['queueList']
        del d['sharedData']
        #del d['queueManager']        
        return d
        
class QueueManager(threading.Thread):
    def __init__(self, player, processNo, queueList):
        threading.Thread.__init__(self)
        self.player = player
        self.processNo = processNo
        self.queueList = queueList
        self.running = True
        self.sendNo = 0
    
    def run(self):
        while (self.running):
            params = self.queueList[self.processNo].get()
            #params = pickle.loads(params)
            #start = time.time()
            #params = self.player.sharedData['params']
            self.player.modelRunner.setParams(params)
            #print 'run() processNo=%s took %.2fs' % (self.processNo, time.time() - start)

    def sendParams(self):
        #start = time.time()
        params = self.player.modelRunner.getParams()
        #params = pickle.dumps(params, protocol=-1)
        #print 'sendParams() processNo=%s took %.2fs' % (self.processNo, time.time() - start)
        for i, queue in enumerate(self.queueList):
            if i != self.processNo:
                queue.put(params)
    """
    def run(self):
        while (self.running):
            self.queueList[self.processNo].get()
            #start = time.time()
            params = self.player.sharedData['params']
            #params = pickle.loads(params)
            self.player.modelRunner.setParams(params)
            #print 'run() processNo=%s took %.2fs' % (self.processNo, time.time() - start)

    def sendParams(self):
        #step1 = time.time()
        params = self.player.modelRunner.getParams()
        #step2 = time.time()
        #params = pickle.dumps(params, protocol=-1)
        #params = pickle.dumps(params)
        #step3 = time.time()
        self.player.sharedData['params'] = params
        #step4 = time.time()
        #print 'sendParams() processNo=%s time1=%.2fs, time2=%.2fs, time3=%.2fs, total=%.2fs. pickledSize=%s' \
        #    % (self.processNo, step2-step1, step3-step2, step4-step3, time.time() - step1, len(params))
        for i, queue in enumerate(self.queueList):
            if i != self.processNo:
                queue.put(1)     # send 1 to notify that sharedData is written
    """
    def finish(self):
        self.running = False
        
class DebugInput(threading.Thread):
    def __init__(self, player):
        threading.Thread.__init__(self)
        self.player = player
        self.running = True
    
    def run(self):
        time.sleep(5)
        while (self.running):
            input = raw_input('')
            if input == 'd':
                self.player.debug = not self.player.debug
                print 'Debug mode : %s' % self.player.debug
                
    def finish(self):
        self.running = False
        
global_data = []

def forkProcess(settings, processNo, queueList, sharedData):
    print 'forkProcess %s' % processNo
    player = DeepRLPlayer(settings, processNo= processNo, queueList=queueList, sharedData=sharedData)
    player.train()
    
def train(settings, saveFile=None):
    if saveFile is not None:        # retrain
        with open(saveFile + '.pickle') as f:
            player = pickle.load(f)
            player.trainStart = time.strftime('%Y%m%d_%H%M%S')
            logFile="output/%s_%s.log" % (settings['game'], player.trainStart)            
            util.Logger(logFile)
            print 'Resume trainig: %s' % saveFile
            player.printEnv()
            player.initializePost()
            player.modelRunner.load(saveFile + '.weight')
            player.train(replayMemoryNo = player.replayMemoryNo)
    else:
        multiProcessNo = settings['multi_process_no']
        if multiProcessNo > 1:
            sharedData = multiprocessing.Manager().dict()
            queueList = []
            processList = []
            for i in range(multiProcessNo):        
                queue = Queue()
                queueList.append(queue)
                
            for i in range(multiProcessNo):        
                print 'creating a child process[%s]' % i
                p = Process(target=forkProcess, args=(settings, i, queueList, sharedData))
                p.start()
                processList.append(p)
                
            while True:
                time.sleep(1000)
            
        else:
            player = DeepRLPlayer(settings)
            player.trainStep = 0
            player.train()

def play(settings, playFile=None):
    print 'Play using dataFile: %s' % playFile
    player = DeepRLPlayer(settings, playFile)
    player.modelRunner.load(playFile + '.weight')
    player.test(0)
    
if __name__ == '__main__':    
    settings = {}

    #settings['game'] = 'breakout'
    settings['game'] = 'space_invaders'
    #settings['game'] = 'enduro'
    #settings['game'] = 'kung_fu_master'
    #settings['game'] = 'krull'
    #settings['game'] = 'hero'
    #settings['game'] = 'qbert'
    #settings['game'] = 'time_pilot'

    settings['rom'] = '/media/big/download/roms/%s.bin' % settings['game']    
    settings['frame_repeat'] = 4    
    settings['show_screen'] = False
    settings['use_keyboard'] = False
    settings['train_batch_size'] = 32
    settings['max_replay_memory'] = 1000000
    settings['max_epoch'] = 200
    settings['epoch_step'] = 250000
    settings['discount_factor'] = 0.99
    settings['train_min_epsilon'] = 0.1           # Minimum greey epsilon value for exloration
    settings['update_step'] = 10000               # Copy train network into target network every this train step
    settings['train_start'] = 50000                   # Start training after filling this replay memory size
    settings['train_step'] = 4                            # Train every this screen step
    settings['test_step'] = 125000                   # Test for this number of steps
    settings['test_epsilon'] = 0.05                   # Greed epsilon for test
    settings['save_step'] = 50000                    # Save result every this training step
    settings['screen_width'] = 84
    settings['screen_height'] = 84
    settings['screen_history'] = 4
    settings['learning_rate'] = 0.00025
    settings['rms_decay'] = 0.95
    settings['lost_life_game_over'] = True
    #settings['lost_life_game_over'] = False
    settings['update_step_in_stepNo'] = True
    settings['double_dqn'] = False
    settings['prioritized_replay'] = False
    settings['use_priority_weight'] = True
    settings['minibatch_random'] = True        # Whether to use random indexing for minibatch or not 
    settings['multi_process_no'] = 0                # Number of multi processor for Asynchronous RL

    #settings['backend'] = 'NEON'
    settings['backend'] = 'TF'
    
    settings['use_self.currentState'] = True
    settings['use_successive_two_frames'] = True
    settings['dnn_initializer'] = 'fan_in'
    #settings['dnn_initializer'] = 'xavier'
    settings['optimizer'] = 'RMSProp'
    #settings['ndimage.zoom'] = True

    #settings['use_gpu_replay_mem'] = True           # Whether to store replay memory in gpu or not to speed up leraning
    settings['use_gpu_replay_mem'] = False

    """
    # Double DQN hyper params
    settings['double_dqn'] = True
    settings['train_min_epsilon'] = 0.01
    settings['test_epsilon'] = 0.001
    settings['update_step'] = 30000
    """

    """
    # Prioritized experience replay params for RANK
    settings['prioritized_replay'] = True
    settings['learning_rate'] = 0.00025 / 4
    settings['prioritized_mode'] = 'RANK'
    settings['sampling_alpha'] = 0.7
    settings['sampling_beta'] = 0.5
    settings['heap_sort_term'] = 250000
    """

    """
    # Prioritized experience replay params for PROPORTION
    settings['prioritized_replay'] = True
    settings['learning_rate'] = 0.00025 / 4
    #settings['learning_rate'] = 0.00025 * 5
    settings['prioritized_mode'] = 'PROPORTION'
    settings['sampling_alpha'] = 0.6
    settings['sampling_beta'] = 0.4
    settings['heap_sort_term'] = 250000
    """

    """
    # Asynchronous RL
    settings['train_start'] = settings['train_batch_size'] + settings['screen_history'] - 1 
    settings['max_replay_memory'] = settings['train_start'] + 100
    settings['minibatch_random'] = False
    settings['multi_process_no'] = 8
    settings['multi_process_copy_step'] = 1
    """
    
    dataFile = None    
    #dataFile = 'snapshot/breakout/dqn_neon_3100000.prm'
    #dataFile = 'snapshot/%s/%s' % (settings['game'], '20160831_200101/dqn_200000')
    
    train(settings, dataFile)
    #play(settings, dataFile)
