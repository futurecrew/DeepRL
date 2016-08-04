import caffe
import numpy as np
import random
import thread
import time
import threading
import multiprocessing

class ModelRunner():
    def __init__(self, queueFromGame, 
                 trainBatchSize, solverPrototxt, testPrototxt, 
                 maxReplayMemory, discountFactor,
                 updateStep, maxActionNo,
                 caffemodel = None):
        
        self.trainBatchSize = trainBatchSize
        self.discountFactor = discountFactor
        self.updateStep = updateStep
        self.maxReplayMemory = maxReplayMemory
        
        caffe.set_mode_gpu()
        
        # Train
        self.solver = caffe.SGDSolver(solverPrototxt)
        self.trainNet = self.solver.net
        self.maxActionNo = maxActionNo
        
        # Test
        #self.testNet = caffe.Net(testPrototxt, caffemodel, caffe.TEST)
        self.testNet = caffe.Net(testPrototxt, caffe.TEST)

        self.replayMemory = []            
        self.running = True
        self.step = 0

        # DJDJ
        #thread.start_new_thread(self.train, ())

        self.dataIndex = 2 

    def addData(self, stateHistory, action, reward, newStateHistory, gameOver, episodeStep):
        if len(self.replayMemory) > self.maxReplayMemory:
            del self.replayMemory[0]
        
        self.replayMemory.append((stateHistory, action, reward, newStateHistory, gameOver, episodeStep))
            
    def test(self, stateHistoryStack):
        reshapedData = stateHistoryStack.reshape((1, 4, 84, 84))
        blobs_out = self.testNet.forward(data=reshapedData.astype(np.float32, copy=False))            
        return blobs_out['cls_score']

    def train(self):
        #if len(self.replayMemory) <= 50000:
        if len(self.replayMemory) % 1000 == 0:
            print '-- len(self.replayMemory) : %s' % len(self.replayMemory)

        if len(self.replayMemory) <= 1000000:
            return
                        
        for i in range(0, self.trainBatchSize):
            stateHistory, actionIndex, reward, newStateHistory, gameOver, episodeStep \
                = self.replayMemory[random.randint(0, len(self.replayMemory)-1)]
            
            stateHistoryStack = np.reshape(stateHistory, (4, 84, 84))    
            newStateHistoryStack = np.reshape(newStateHistory, (4, 84, 84))    

            if i == 0:
                trainState = np.zeros((self.trainBatchSize, 4, 84, 84), dtype=np.float32)
                trainNewState = np.zeros((self.trainBatchSize, 4, 84, 84), dtype=np.float32)
                trainAction = []
                trainReward = []
                trainGameOver = []
                steps = []
             
            trainState[i, :, :, :] = stateHistoryStack
            trainNewState[i, :, :, :] = newStateHistoryStack
            trainAction.append(actionIndex)
            trainReward.append(reward)
            trainGameOver.append(gameOver)
            steps.append(episodeStep)

        label = np.zeros((self.trainBatchSize, self.maxActionNo), dtype=np.float32)
            
        self.solver.net.forward(data=trainNewState.astype(np.float32, copy=False),
                                                  labels=label)
        newActionValues = self.solver.net.blobs['cls_score'].data.copy()
            
        self.solver.net.forward(data=trainState.astype(np.float32, copy=False),
                                                  labels=label)
        label = self.solver.net.blobs['cls_score'].data.copy()
        
        for i in range(0, self.trainBatchSize):
            if trainGameOver[i]:
                label[i, trainAction[i]] = trainReward[i]
            else:
                label[i, trainAction[i]] = trainReward[i] + self.discountFactor* np.max(newActionValues[i])

        self.solver.net.blobs['data'].data[...] = trainState.astype(np.float32, copy=False)
        self.solver.net.blobs['labels'].data[...] = label

        self.solver.step(1)

        """
        classScore = self.solver.net.blobs['cls_score'].data.copy()

        if trainGameOver[0]:
            print 'game over'
            print 'label  : %s' % label[0]
            print 'score : %s' % classScore[0]
        """
        
        """
        self.solver.net.forward(data=trainState.astype(np.float32, copy=False),
                                                  labels=label)

        diff = classScore - label
        loss = np.sum(diff * diff) / self.trainBatchSize / 2
        if loss > 0.2:
            print 'loss : %s' % loss
        """

        self.step += 1
        
        if  self.step % self.updateStep == 0:
            self.updateModel()

        #print 'ModelRunner thread finished.'

    def updateModel(self):
        for param in self.testNet.params:
            for i in range(len(self.testNet.params[param])):
                self.testNet.params[param][i].data[...] = self.trainNet.params[param][i].data.copy()
        print ('Updated test model')

    def finishTrain(self):
        self.running = False
        
        