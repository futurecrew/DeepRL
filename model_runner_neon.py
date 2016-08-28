import numpy as np
import random
import thread
import time
import threading
import traceback
from neon.backends import gen_backend
from neon.layers import Conv, Affine, Pooling
from neon.initializers import Gaussian
from neon.transforms.activation import Rectlin, Softmax
from neon.models import Model
from neon.layers import GeneralizedCost
from neon.transforms import SumSquared
from neon.optimizers import GradientDescentMomentum, RMSProp

class ModelRunnerNeon():
    def __init__(self, settings,  maxActionNo, batchDimension):
        self.settings = settings
        self.trainBatchSize = settings['train_batch_size']
        self.discountFactor = settings['discount_factor']
        self.updateStep = settings['update_step']
        self.totalTrainStep = 0
        
        self.be = gen_backend(backend='gpu',             
                         batch_size=self.trainBatchSize)

        self.inputShape = (batchDimension[1], batchDimension[2], batchDimension[3], batchDimension[0])
        self.input = self.be.empty(self.inputShape)
        self.input.lshape = self.inputShape # HACK: needed for convolutional networks
        self.targets = self.be.empty((maxActionNo, self.trainBatchSize))

        self.trainNet = Model(self.createLayers(maxActionNo))
        self.cost = GeneralizedCost(costfunc=SumSquared())
        # Bug fix
        for l in self.trainNet.layers.layers:
            l.parallelism = 'Disabled'
        self.trainNet.initialize(self.inputShape[:-1], self.cost)
        
        self.targetNet = Model(self.createLayers(maxActionNo))
        # Bug fix
        for l in self.targetNet.layers.layers:
            l.parallelism = 'Disabled'
        self.targetNet.initialize(self.inputShape[:-1])
        

        self.optimizer = RMSProp(decay_rate=settings['rms_decay'],
                                            learning_rate=settings['learning_rate'])

        self.maxActionNo = maxActionNo
        self.running = True
        self.blankLabel = np.zeros((self.trainBatchSize, self.maxActionNo), dtype=np.float32)

    def createLayers(self, maxActionNo):
        init_gauss = Gaussian(0, 0.01)
        layers = [Conv(fshape=(8, 8, 32), strides=4, init=init_gauss, activation=Rectlin()),
                        Conv(fshape=(4, 4, 64), strides=2, init=init_gauss, activation=Rectlin()),
                        Conv(fshape=(3, 3, 64), strides=1, init=init_gauss, activation=Rectlin()),
                        Affine(nout=512, init=init_gauss, activation=Rectlin()),
                        Affine(nout=maxActionNo, init=init_gauss)
                        ]
        
        return layers        
        
    def clipReward(self, reward):
            if reward > 0:
                return 1
            elif reward < 0:
                return -1
            else:
                return 0

    def setInput(self, data):
        self.input.set(data.transpose(1, 2, 3, 0).copy())
        # normalize network input between 0 and 1
        self.be.divide(self.input, 255, self.input)
        
    def predict(self, historyBuffer):
        self.setInput(historyBuffer)
        output  = self.trainNet.fprop(self.input, inference=True)
        return output.T.asnumpyarray()[0]            

    def train(self, minibatch, replayMemory, debug):
        if self.settings['prioritized_replay'] == True:
            prestates, actions, rewards, poststates, lostLives, replayIndexes, heapIndexes, weights = minibatch
        else:
            prestates, actions, rewards, poststates, lostLives = minibatch
        
        # Get Q*(s, a) with targetNet
        self.setInput(poststates)
        postQvalue = self.targetNet.fprop(self.input, inference=True).T.asnumpyarray()
        
        if self.settings['double_dqn'] == True:
            # Get Q*(s, a) with trainNet
            postQvalue2 = self.trainNet.fprop(self.input, inference=True).T.asnumpyarray()
        
        # Get Q(s, a) with trainNet
        self.setInput(prestates)
        preQvalue = self.trainNet.fprop(self.input, inference=False)
        
        label = preQvalue.asnumpyarray().copy()
        for i in range(0, self.trainBatchSize):
            if lostLives[i]:
                label[actions[i], i] = self.clipReward(rewards[i])
            else:
                if self.settings['double_dqn'] == True:
                    maxIndex = np.argmax(postQvalue2[i])
                    label[actions[i], i] = self.clipReward(rewards[i]) + self.discountFactor* postQvalue[i][maxIndex]
                else:
                    label[actions[i], i] = self.clipReward(rewards[i]) + self.discountFactor* np.max(postQvalue[i])

        # copy targets to GPU memory
        self.targets.set(label)
    
        delta = self.cost.get_errors(preQvalue, self.targets)
        
        if self.settings['prioritized_replay'] == True:
            deltaValue = delta.asnumpyarray()
            for i in range(self.trainBatchSize):
                if debug:
                    print 'weight[%s]: %.5f, delta: %.5f, newDelta: %.5f' % (i, weights[i], deltaValue[actions[i], i], weights[i] * deltaValue[actions[i], i]) 
                replayMemory.updateTD(heapIndexes[i], abs(deltaValue[actions[i], i]))
                deltaValue[actions[i], i] = weights[i] * deltaValue[actions[i], i]
            if self.settings['use_priority_weight'] == True:
                delta.set(deltaValue.copy())
            #deltaValue2 = delta.asnumpyarray()
            #pass
            
        self.be.clip(delta, -1.0, 1.0, out = delta)        
        self.trainNet.bprop(delta)
        self.optimizer.optimize(self.trainNet.layers_to_optimize, epoch=0)

        self.totalTrainStep += 1
        
        if self.totalTrainStep % self.updateStep == 0:
            self.updateModel()

    def updateModel(self):
        # have to serialize also states for batch normalization to work
        pdict = self.trainNet.get_description(get_weights=True, keep_states=True)
        self.targetNet.deserialize(pdict, load_states=True)
        #print ('Updated target model')

    def finishTrain(self):
        self.running = False
    
    def load(self, fileName):
        self.trainNet.load_params(fileName)
        self.updateModel()
        
    def save(self, fileName):
        self.trainNet.save_params(fileName)
        