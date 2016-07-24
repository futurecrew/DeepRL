import caffe
import numpy as np
import random
import time
import threading

class ModelRunner(threading.Thread):
    def __init__(self, name, 
                 solver_prototxt=None, maxReplayMemory=0, discountFactor=0,
                 prototxt=None, caffemodel=None, maxActionNo=0):
        
        threading.Thread.__init__(self)        
        self.name = name
        self.maxReplayMemory = maxReplayMemory
        self.discountFactor = discountFactor
        if solver_prototxt != None:
            self.solver = caffe.SGDSolver(solver_prototxt)
            self.net = self.solver.net
            self.maxActionNo = maxActionNo
        else:
            #self.net = caffe.Net(prototxt, caffemodel, caffe.TEST)
            self.net = caffe.Net(prototxt, caffe.TEST)
            
        self.replayMemory = []
        self.running = True
            
    def addData(self, state, action, reward, newState):
        if len(self.replayMemory) > self.maxReplayMemory:
            del self.replayMemory[0]
        self.replayMemory.append((state, action, reward, newState))
            
    def test(self, testData):
        reshapedData = testData.reshape((1, 1, testData.shape[0], testData.shape[1]))
        blobs_out = self.net.forward(data=reshapedData.astype(np.float32, copy=False))            
        return blobs_out['cls_score']

    def train(self):
        print 'ModelRunner thread started.'

        while self.running:
            if len(self.replayMemory) <= 0:
                time.sleep(1)
                continue
            
            state, action, reward, newState = self.replayMemory[random.randint(0, len(self.replayMemory)-1)]
            
            reshapedState = state.reshape((1, 1, state.shape[0], state.shape[1]))
            reshapedNewState = newState.reshape((1, 1, newState.shape[0], newState.shape[1]))
            label = np.zeros((1, self.maxActionNo), dtype=np.float32)
            
            self.solver.net.forward(data=reshapedNewState.astype(np.float32, copy=False),
                                                      labels=label)
            newActionValues = self.solver.net.blobs['cls_score'].data.copy()
            
            self.solver.net.forward(data=reshapedState.astype(np.float32, copy=False),
                                                      labels=label)
            label = self.solver.net.blobs['cls_score'].data.copy()
            
            label[0][action] = reward + self.discountFactor* np.max(newActionValues)
            #label[0][action] = 100
            
            if reward != 0:
                print 'reward : %s' % reward
                    
                if action != 0:
                    pass
            
            self.solver.net.blobs['data'].data[...] = reshapedState.astype(np.float32, copy=False)
            self.solver.net.blobs['labels'].data[...] = label
    
            self.solver.step(1)

        print 'ModelRunner thread finished.'

    def copyFrom(self, fromNet):
        print ('Loading trained model weights from {:s}').format(fromNet.name)
        
        for param in self.net.params:
            for i in range(len(self.net.params[param])):
                self.net.params[param][i].data[...] = fromNet.net.params[param][i].data.copy()

    def run(self):
        self.train()
            
        
    def finishTrain(self):
        self.running = False