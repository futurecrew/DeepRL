import caffe
import numpy as np
import random
import thread
import time
import threading
import multiprocessing

class ModelRunner():
    def __init__(self, queueFromGame, 
                 trainBatchSize, solverPrototxt, testPrototxt, maxReplayMemory=0, discountFactor=0,
                 updateStep = 10000, caffemodel=None, maxActionNo=0):
        
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

        thread.start_new_thread(self.train, ())

    def addData(self, state, action, reward, newState):
        if len(self.replayMemory) > self.maxReplayMemory:
            del self.replayMemory[0]
        self.replayMemory.append((state, action, reward, newState))
            
    def test(self, testData):
        reshapedData = testData.reshape((1, 1, testData.shape[0], testData.shape[1]))
        blobs_out = self.testNet.forward(data=reshapedData.astype(np.float32, copy=False))            
        return blobs_out['cls_score']

    def train(self):
        print 'ModelRunner thread started.'

        self.step = 0
        while self.running:
            if len(self.replayMemory) <= 2000:
                time.sleep(1)
                continue
                            
            for i in range(0, self.trainBatchSize):
                state, action, reward, newState = self.replayMemory[random.randint(0, len(self.replayMemory)-1)]
                
                if i == 0:
                    trainState = np.zeros((self.trainBatchSize, 1, state.shape[0], state.shape[1]), dtype=np.float32)
                    trainNewState = np.zeros((self.trainBatchSize, 1, newState.shape[0], newState.shape[1]), dtype=np.float32)
                    trainAction = []
                    trainReward = []
                    trainGameOver = []
                    
                trainState[i, :, :, :] = state
                trainNewState[i, :, :, :] = newState
                trainAction.append(action)
                trainReward.append(reward)
                #trainGameOver.append(gameOver)
                trainGameOver.append(False)

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
                
            #if np.max(trainReward) != 0:
                #print 'reward : %s' % reward
                        
            self.solver.net.blobs['data'].data[...] = trainState.astype(np.float32, copy=False)
            self.solver.net.blobs['labels'].data[...] = label
    
            self.solver.step(1)

            self.step += 1
            
            if  self.step % self.updateStep == 0:
                self.updateModel()
                


        print 'ModelRunner thread finished.'

    def updateModel(self):
        for param in self.testNet.params:
            for i in range(len(self.testNet.params[param])):
                self.testNet.params[param][i].data[...] = self.trainNet.params[param][i].data.copy()
        print ('Updated test model')

    def finishTrain(self):
        self.running = False
        
        