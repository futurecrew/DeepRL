import caffe
import numpy as np
import random
import time
import threading

class ModelRunner():
    def __init__(self, solver_prototxt, testPrototxt, trainBatchSize=1, maxReplayMemory=0, discountFactor=0,
                 caffemodel=None, maxActionNo=0, updateStep = 10000):
        
        self.trainBatchSize = trainBatchSize
        self.testPrototxt = testPrototxt
        self.maxReplayMemory = maxReplayMemory
        self.discountFactor = discountFactor
        
        # Train net
        self.solver = caffe.SGDSolver(solver_prototxt)
        self.trainNet = self.solver.net
        self.maxActionNo = maxActionNo
        
        # Test net
        #self.testNet = caffe.Net(testPrototxt, caffemodel, caffe.TEST)
        self.testNet = caffe.Net(testPrototxt, caffe.TEST)
            
        self.updateStep = updateStep
        self.replayMemory = []
        self.running = True
        
    def addData(self, (state, action, reward, newState, gameOver)):
        if len(self.replayMemory) > self.maxReplayMemory:
            del self.replayMemory[0]
        self.replayMemory.append((state, action, reward, newState, gameOver))
            
    def test(self, testData):
        reshapedData = testData.reshape((1, 1, testData.shape[0], testData.shape[1]))
        blobs_out = self.testNet.forward(data=reshapedData.astype(np.float32, copy=False))            
        return blobs_out['cls_score']

    def train(self):
        print 'ModelRunner thread started.'

        trainStep = 0
        
        while self.running:
            if len(self.replayMemory) <= self.trainBatchSize * 10:
                time.sleep(1)
                continue
            
            for i in range(0, self.trainBatchSize):
                state, action, reward, newState, gameOver = self.replayMemory[random.randint(0, len(self.replayMemory)-1)]
                
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
                trainGameOver.append(gameOver)
                    
            #reshapedState = trainState.reshape((self.trainBatchSize, 1, state.shape[0], state.shape[1]))
            #reshapedNewState = trainNewState.reshape((self.trainBatchSize, 1, newState.shape[0], newState.shape[1]))

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
            trainStep += 1
            
            if trainStep % self.updateStep == 0:
                self.updateNet()
            

        print 'ModelRunner thread finished.'

    def updateNet(self):
        print ('Updated model weights from train net')
        for param in self.trainNet.params:
            for i in range(len(self.trainNet.params[param])):
                self.testNet.params[param][i].data[...] = self.trainNet.params[param][i].data.copy()

    def runTrain(self, receiveQueue, sendQueue):
        self.receiveQueue = receiveQueue 
        self.sendQueue = sendQueue
        self.queueReader = QueueReader(self, self.receiveQueue, self.sendQueue)
        self.queueReader.start()
        self.train()        
        
    def finish(self):
        self.running = False
        
        if self.mode == 'TRAIN':
            self.queueReader.finish()
        
        
class QueueReader(threading.Thread):
    def __init__(self, modelRunner, receiveQueue, sendQueue):
        threading.Thread.__init__(self)        
        self.modelRunner = modelRunner
        self.receiveQueue = receiveQueue
        self.sendQueue = sendQueue
        self.running = True
        
    def run(self):
        while self.running:
            obj = self.receiveQueue.get()
            key = obj[0]
            value = obj[1]
            
            #print 'queue.get : %s' % key
            
            if key == 'addData':
                self.modelRunner.addData(value)
            elif key == 'test':
                result = self.modelRunner.test(value)
                self.sendQueue.put(result)
            elif key == 'finish':
                self.modelRunner.finish()
                
            
        
    def finish(self):
        self.running = False
            