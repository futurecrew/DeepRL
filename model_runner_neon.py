import numpy as np
import os
import random
import math
import time
import threading
import traceback
import pickle
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
    def __init__(self, settings,  max_action_no, batch_dimension):
        self.settings = settings
        self.train_batch_size = settings['train_batch_size']
        self.discount_factor = settings['discount_factor']
        self.use_gpu_replay_mem = settings['use_gpu_replay_mem']
        
        self.be = gen_backend(backend='gpu',             
                         batch_size=self.train_batch_size)

        self.input_shape = (batch_dimension[1], batch_dimension[2], batch_dimension[3], batch_dimension[0])
        self.input = self.be.empty(self.input_shape)
        self.input.lshape = self.input_shape # HACK: needed for convolutional networks
        self.targets = self.be.empty((max_action_no, self.train_batch_size))

        if self.use_gpu_replay_mem:
            self.history_buffer = self.be.zeros(batch_dimension, dtype=np.uint8)
            self.input_uint8 = self.be.empty(self.input_shape, dtype=np.uint8)
        else:
            self.history_buffer = np.zeros(batch_dimension, dtype=np.float32)

        self.train_net = Model(self.create_layers(max_action_no))
        self.cost = GeneralizedCost(costfunc=SumSquared())
        # Bug fix
        for l in self.train_net.layers.layers:
            l.parallelism = 'Disabled'
        self.train_net.initialize(self.input_shape[:-1], self.cost)
        
        self.target_net = Model(self.create_layers(max_action_no))
        # Bug fix
        for l in self.target_net.layers.layers:
            l.parallelism = 'Disabled'
        self.target_net.initialize(self.input_shape[:-1])

        if self.settings['optimizer'] == 'Adam':        # Adam
            self.optimizer = Adam(beta_1=settings['rms_decay'],
                                            beta_2=settings['rms_decay'],
                                            learning_rate=settings['learning_rate'])
        else:		# Neon RMSProp
            self.optimizer = RMSProp(decay_rate=settings['rms_decay'],
                                            learning_rate=settings['learning_rate'])

        self.max_action_no = max_action_no
        self.running = True

    def add_to_history_buffer(self, state):
        if self.use_gpu_replay_mem:
            self.history_buffer[0, :-1][:] = self.history_buffer[0, 1:]
            self.history_buffer[0, -1][:] = state
        else:
            self.history_buffer[0, :-1] = self.history_buffer[0, 1:]
            self.history_buffer[0, -1] = state

    def clear_history_buffer(self):
        if self.use_gpu_replay_mem:
            self.history_buffer[:] = 0
        else:
            self.history_buffer.fill(0)

    def get_initializer(self, input_size):
        dnnInit = self.settings['dnn_initializer']
        if dnnInit == 'xavier':
            initializer = Xavier()
        elif dnnInit == 'fan_in':
            std_dev = 1.0 / math.sqrt(input_size)
            initializer = Uniform(low=-std_dev, high=std_dev)
        else:
            initializer = Gaussian(0, 0.01)
        return initializer
            
    def create_layers(self, max_action_no):
        layers = []

        initializer = self.get_initializer(input_size = 4 * 8 * 8)
        layers.append(Conv(fshape=(8, 8, 32), strides=4, init=initializer, bias=initializer, activation=Rectlin()))

        initializer = self.get_initializer(input_size = 32 * 4 * 4)
        layers.append(Conv(fshape=(4, 4, 64), strides=2, init=initializer, bias=initializer, activation=Rectlin()))
        
        initializer = self.get_initializer(input_size = 64 * 3 * 3)
        layers.append(Conv(fshape=(3, 3, 64), strides=1, init=initializer, bias=initializer, activation=Rectlin()))
        
        initializer = self.get_initializer(input_size = 7 * 7 * 64)
        layers.append(Affine(nout=512, init=initializer, bias=initializer, activation=Rectlin()))
        
        initializer = self.get_initializer(input_size = 512)
        layers.append(Affine(nout=max_action_no, init=initializer, bias=initializer))
        
        return layers        
        
    def clip_reward(self, reward):
            if reward > 0:
                return 1
            elif reward < 0:
                return -1
            else:
                return 0

    def set_input(self, data):
        if self.use_gpu_replay_mem:
            self.be.copy_transpose(data, self.input_uint8, axes=(1, 2, 3, 0))
            self.input[:] = self.input_uint8 / 255
        else:
            self.input.set(data.transpose(1, 2, 3, 0).copy())
            self.be.divide(self.input, 255, self.input)

    def predict(self, history_buffer):
        self.set_input(history_buffer)
        output  = self.train_net.fprop(self.input, inference=True)
        return output.T.asnumpyarray()[0]            

    def train(self, minibatch, replay_memory, debug):
        if self.settings['prioritized_replay'] == True:
            prestates, actions, rewards, poststates, terminals, replay_indexes, heap_indexes, weights = minibatch
        else:
            prestates, actions, rewards, poststates, terminals = minibatch
        
        # Get Q*(s, a) with targetNet
        self.set_input(poststates)
        post_qvalue = self.target_net.fprop(self.input, inference=True).T.asnumpyarray()
        
        if self.settings['double_dqn'] == True:
            # Get Q*(s, a) with trainNet
            post_qvalue2 = self.train_net.fprop(self.input, inference=True).T.asnumpyarray()
        
        # Get Q(s, a) with trainNet
        self.set_input(prestates)
        pre_qvalue = self.train_net.fprop(self.input, inference=False)
        
        label = pre_qvalue.asnumpyarray().copy()
        for i in range(0, self.train_batch_size):
            if terminals[i]:
                label[actions[i], i] = self.clip_reward(rewards[i])
            else:
                if self.settings['double_dqn'] == True:
                    maxIndex = np.argmax(post_qvalue2[i])
                    label[actions[i], i] = self.clip_reward(rewards[i]) + self.discount_factor* post_qvalue[i][maxIndex]
                else:
                    label[actions[i], i] = self.clip_reward(rewards[i]) + self.discount_factor* np.max(post_qvalue[i])

        # copy targets to GPU memory
        self.targets.set(label)
    
        delta = self.cost.get_errors(pre_qvalue, self.targets)
        
        if self.settings['prioritized_replay'] == True:
            delta_value = delta.asnumpyarray()
            for i in range(self.train_batch_size):
                if debug:
                    print 'weight[%s]: %.5f, delta: %.5f, newDelta: %.5f' % (i, weights[i], delta_value[actions[i], i], weights[i] * delta_value[actions[i], i]) 
                replay_memory.update_td(heap_indexes[i], abs(delta_value[actions[i], i]))
                delta_value[actions[i], i] = weights[i] * delta_value[actions[i], i]
            if self.settings['use_priority_weight'] == True:
                delta.set(delta_value.copy())
            #delta_value2 = delta.asnumpyarray()
            #pass
            
        self.be.clip(delta, -1.0, 1.0, out = delta)        
        self.train_net.bprop(delta)
        self.optimizer.optimize(self.train_net.layers_to_optimize, epoch=0)

    def update_model(self):
        # have to serialize also states for batch normalization to work
        pdict = self.train_net.get_description(get_weights=True, keep_states=True)
        self.target_net.deserialize(pdict, load_states=True)
        #print ('Updated target model')

    def finish_train(self):
        self.running = False
    
    def load(self, file_name):
        self.train_net.load_params(file_name)
        self.update_model()
        
    def save(self, file_name):
        self.train_net.save_params(file_name)
        
