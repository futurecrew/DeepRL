import caffe
import numpy as np
import random

class ModelRunner:
    def __init__(self, name, 
                 solver_prototxt=None, maxReplayMemory=0, discountFactor=0,
                 prototxt=None, caffemodel=None):
        self.name = name
        self.maxReplayMemory = maxReplayMemory
        self.discountFactor = discountFactor
        if solver_prototxt != None:
            self.solver = caffe.SGDSolver(solver_prototxt)
            self.net = self.solver.net
        else:
            #self.net = caffe.Net(prototxt, caffemodel, caffe.TEST)
            self.net = caffe.Net(prototxt, caffe.TEST)
            
        self.replayMemory = []
    
    def addData(self, state, action, reward, newState):
        if len(self.replayMemory) > self.maxReplayMemory:
            del self.replayMemory[0]
        self.replayMemory.append((state, action, reward, newState))        
        
    def test(self, testData):
        reshapedData = testData.reshape((1, 1, testData.shape[0], testData.shape[1]))
        blobs_out = self.net.forward(data=reshapedData.astype(np.float32, copy=False))            
        return blobs_out['cls_prob']

    def train(self):
        state, action, reward, newState = self.replayMemory[random.randint(0, len(self.replayMemory)-1)]
        newActionValues = self.solver.net.forward(data=newState.astype(np.float32, copy=False))
        actionValues = self.solver.net.forward(data=state.astype(np.float32, copy=False))
        
        optimal = reward + self.discountFactor* np.max(newActionValues)
        actionValues[action] = optimal
         
        self.solver.net.backward(diff=actionValues.astype(np.float32, copy=False))

    def copy(self, fromNet):
        print ('Loading trained model '
               'weights from {:s}').format(fromNet.name)
        
        return
    
        for i in range(len(self.net.params)):
            self.net.params[i].data[...] = fromNet.net.params[i].data[...]
