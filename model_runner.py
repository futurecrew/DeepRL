import caffe
import numpy as np
import random
import thread
import time
import threading
import multiprocessing

class ModelRunner():
    def __init__(self, settings,  maxActionNo, replayMemory, caffemodel = None):
        self.trainBatchSize = settings['TRAIN_BATCH_SIZE']
        self.discountFactor = settings['DISCOUNT_FACTOR']
        self.updateStep = settings['UPDATE_STEP']
        self.maxReplayMemory = settings['MAX_REPLAY_MEMORY']
        
        caffe.set_mode_gpu()
        
        # Train
        self.solver = caffe.SGDSolver(settings['SOLVER_PROTOTXT'])
        self.trainNet = self.solver.net
        self.maxActionNo = maxActionNo
        
        # Test
        if caffemodel != None:
            self.targetNet = caffe.Net(settings['TARGET_PROTOTXT'], caffemodel, caffe.TEST)
        else:
            self.targetNet = caffe.Net(settings['TARGET_PROTOTXT'], caffe.TEST)

        if 'RESTORE' in settings:
            self.solver.restore(settings['RESTORE'])
            self.updateModel()
        if 'PLAY' in settings:
            self.trainNet.copy_from(settings['PLAY'])

        self.replayMemory = replayMemory
        self.running = True
        self.step = 0
        self.blankLabel = np.zeros((self.trainBatchSize, self.maxActionNo), dtype=np.float32)

        #thread.start_new_thread(self.train, ())
    
    def clipReward(self, reward):
            if reward > 0:
                return 1
            elif reward < 0:
                return -1
            else:
                return 0

    def predict(self, historyBuffer):
        self.trainNet.forward(
                              data = (historyBuffer).astype(np.float32, copy=False),
                              labels = self.blankLabel)
        return self.trainNet.blobs['cls_score'].data[0]            

    def train(self):
        if self.replayMemory.count <= 50000:
        #if self.replayMemory.count <= 1000:
            return
        
        prestates, actions, rewards, poststates, gameOvers = self.replayMemory.getMinibatch()
        
        # Get Q*(s, a)
        self.targetNet.forward(data=poststates.astype(np.float32, copy=False))
        targetValues = self.targetNet.blobs['cls_score'].data
        
        # Get Q(s, a)
        self.trainNet.forward(data=prestates.astype(np.float32, copy=False),
                                                  labels=self.blankLabel)
        label = self.trainNet.blobs['cls_score'].data.copy()
        
        for i in range(0, self.trainBatchSize):
            if gameOvers[i]:
                label[i, actions[i]] = self.clipReward(rewards[i])
            else:
                label[i, actions[i]] = self.clipReward(rewards[i]) + self.discountFactor* np.max(targetValues[i])

        self.trainNet.blobs['data'].data[...] = prestates.astype(np.float32, copy=False)
        self.trainNet.blobs['labels'].data[...] = label

        self.solver.step(1)

        """
        classScore = self.trainNet.blobs['cls_score'].data.copy()

        if trainGameOver[0]:
            print 'game over'
            print 'label  : %s' % label[0]
            print 'score : %s' % classScore[0]
        """
        
        """
        self.trainNet.forward(data=trainState.astype(np.float32, copy=False),
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
        for param in self.targetNet.params:
            for i in range(len(self.targetNet.params[param])):
                self.targetNet.params[param][i].data[...] = self.trainNet.params[param][i].data.copy()
        print ('Updated target model')

    def finishTrain(self):
        self.running = False
        
        