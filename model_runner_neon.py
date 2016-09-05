import numpy as np
import random
import math
import time
import threading
import traceback
from neon.backends import gen_backend
from neon.layers import Conv, Affine, Pooling
from neon.initializers import Gaussian, Xavier, Uniform
from neon.transforms.activation import Rectlin, Softmax
from neon.models import Model
from neon.layers import GeneralizedCost
from neon.transforms import SumSquared
from neon.optimizers import RMSProp, Adam, Schedule
from neon.optimizers.optimizer import Optimizer, get_param_list

class ModelRunnerNeon():
    def __init__(self, settings,  maxActionNo, batchDimension):
        self.settings = settings
        self.trainBatchSize = settings['train_batch_size']
        self.discountFactor = settings['discount_factor']
        self.updateStep = settings['update_step']
        self.useGpuReplayMem = settings['use_gpu_replay_mem']
        
        self.be = gen_backend(backend='gpu',             
                         batch_size=self.trainBatchSize)

        self.inputShape = (batchDimension[1], batchDimension[2], batchDimension[3], batchDimension[0])
        self.input = self.be.empty(self.inputShape)
        self.input.lshape = self.inputShape # HACK: needed for convolutional networks
        self.targets = self.be.empty((maxActionNo, self.trainBatchSize))

        if self.useGpuReplayMem:
            self.historyBuffer = self.be.zeros(batchDimension, dtype=np.uint8)
            self.input_uint8 = self.be.empty(self.inputShape, dtype=np.uint8)
        else:
            self.historyBuffer = np.zeros(batchDimension, dtype=np.float32)

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

        if self.settings['optimizer'] == 'RMSPropDeepMind':		# RMSPropDeepMind
            self.optimizer = RMSPropDeepMind(decay_rate=settings['rms_decay'],
                                            learning_rate=settings['learning_rate'],
                                            epsilon=0.01)
        if self.settings['optimizer'] == 'Adam':        # Adam
            self.optimizer = Adam(beta_1=settings['rms_decay'],
                                            beta_2=settings['rms_decay'],
                                            learning_rate=settings['learning_rate'])
        else:		# Neon RMSProp
            self.optimizer = RMSProp(decay_rate=settings['rms_decay'],
                                            learning_rate=settings['learning_rate'])

        self.maxActionNo = maxActionNo
        self.running = True
        self.blankLabel = np.zeros((self.trainBatchSize, self.maxActionNo), dtype=np.float32)

    def addToHistoryBuffer(self, state):
        if self.useGpuReplayMem:
            self.historyBuffer[0, :-1][:] = self.historyBuffer[0, 1:]
            self.historyBuffer[0, -1][:] = state
        else:
            self.historyBuffer[0, :-1] = self.historyBuffer[0, 1:]
            self.historyBuffer[0, -1] = state

    def clearHistoryBuffer(self):
        if self.useGpuReplayMem:
            self.historyBuffer[:] = 0
        else:
            self.historyBuffer.fill(0)

    def getInitializer(self, inputSize):
        dnnInit = self.settings['dnn_initializer']
        if dnnInit == 'xavier':
            initializer = Xavier()
        elif dnnInit == 'fan_in':
            stdDev = 1.0 / math.sqrt(inputSize)
            initializer = Uniform(low=-stdDev, high=stdDev)
        else:
            initializer = Gaussian(0, 0.01)
        return initializer
            
    def createLayers(self, maxActionNo):
        layers = []

        initializer = self.getInitializer(inputSize = 4 * 8 * 8)
        layers.append(Conv(fshape=(8, 8, 32), strides=4, init=initializer, bias=initializer, activation=Rectlin()))

        initializer = self.getInitializer(inputSize = 32 * 4 * 4)
        layers.append(Conv(fshape=(4, 4, 64), strides=2, init=initializer, bias=initializer, activation=Rectlin()))
        
        initializer = self.getInitializer(inputSize = 64 * 3 * 3)
        layers.append(Conv(fshape=(3, 3, 64), strides=1, init=initializer, bias=initializer, activation=Rectlin()))
        
        initializer = self.getInitializer(inputSize = 7 * 7 * 64)
        layers.append(Affine(nout=512, init=initializer, bias=initializer, activation=Rectlin()))
        
        initializer = self.getInitializer(inputSize = 512)
        layers.append(Affine(nout=maxActionNo, init=initializer, bias=initializer))
        
        return layers        
        
    def clipReward(self, reward):
            if reward > 0:
                return 1
            elif reward < 0:
                return -1
            else:
                return 0

    def setInput(self, data):
        if self.useGpuReplayMem:
            self.be.copy_transpose(data, self.input_uint8, axes=(1, 2, 3, 0))
            self.input[:] = self.input_uint8 / 255
        else:
            self.input.set(data.transpose(1, 2, 3, 0).copy())
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
        

class RMSPropDeepMind(Optimizer):

    """
    Root Mean Square propagation DeepMind replicate.

    Root Mean Square (RMS) propagation protects against vanishing and
    exploding gradients. In RMSprop, the gradient is divided by a running
    average of recent gradients. Given the parameters :math:`\\theta`, gradient :math:`\\nabla J`,
    we keep a running average :math:`\\mu` of the last :math:`1/\\lambda` gradients squared.
    The update equations are then given by

    .. math::

        \\mu' &= \\lambda\\mu + (1-\\lambda)(\\nabla J)
        \\mu2' &= \\lambda\\mu2 + (1-\\lambda)(\\nabla J)^2

    .. math::

        # \\theta' &= \\theta - \\frac{\\alpha}{\\sqrt{\\mu + \\epsilon} + \\epsilon}\\nabla J
        \\theta' &= \\theta - \\frac{\\alpha}{\\sqrt{\\mu2 - (\\mu)^2 + \\epsilon}}\\nabla J

    where we use :math:`\\epsilon` as a (small) smoothing factor to prevent from dividing by zero.
    """

    def __init__(self, stochastic_round=False, decay_rate=0.95, learning_rate=2e-3, epsilon=1e-6,
                 gradient_clip_norm=None, gradient_clip_value=None, name=None,
                 schedule=Schedule()):
        """
        Class constructor.

        Arguments:
            stochastic_round (bool): Set this to True for stochastic rounding.
                                     If False rounding will be to nearest.
                                     If True will perform stochastic rounding using default width.
                                     Only affects the gpu backend.
            decay_rate (float): decay rate of states
            learning_rate (float): the multiplication coefficent of updates
            epsilon (float): smoothing epsilon to avoid divide by zeros
            gradient_clip_norm (float, optional): Target gradient norm.
                                                  Defaults to None.
            gradient_clip_value (float, optional): Value to element-wise clip
                                                   gradients.
                                                   Defaults to None.
            schedule (neon.optimizers.optimizer.Schedule, optional): Learning rate schedule.
                                                                     Defaults to a constant.
        Notes:
            Only constant learning rate is supported currently.
        """
        super(RMSPropDeepMind, self).__init__(name=name)
        self.state_list = None

        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        self.schedule = schedule
        self.gradient_clip_norm = gradient_clip_norm
        self.gradient_clip_value = gradient_clip_value
        self.stochastic_round = stochastic_round

    def optimize(self, layer_list, epoch):
        """
        Apply the learning rule to all the layers and update the states.

        Arguments:
            layer_list (list): a list of Layer objects to optimize.
            epoch (int): the current epoch, needed for the Schedule object.
        """
        lrate = self.schedule.get_learning_rate(self.learning_rate, epoch)
        epsilon, decay = (self.epsilon, self.decay_rate)
        param_list = get_param_list(layer_list)

        scale_factor = self.clip_gradient_norm(param_list, self.gradient_clip_norm)

        for (param, grad), states in param_list:

            param.rounding = self.stochastic_round
            if len(states) == 0:
                states.append(self.be.zeros_like(grad))
                states.append(self.be.zeros_like(grad))

            grad = grad / self.be.bsz
            grad = self.clip_gradient_value(grad, self.gradient_clip_value)

            # update state
            g = states[0]
            g[:] = decay * g + grad * (1.0 - decay)
            g2 = states[1]
            g2[:] = decay * g2 + self.be.square(grad) * (1.0 - decay)

            param[:] = param \
                - (scale_factor * grad * lrate) / self.be.sqrt(g2 - self.be.square(g) + epsilon)

