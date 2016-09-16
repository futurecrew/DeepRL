import numpy as np
import os
import random
import math
import time
import threading
import traceback
import pickle
import tensorflow as tf

class ModelRunnerTF():
    def __init__(self, settings,  maxActionNo, batchDimension):
        self.settings = settings
        self.trainBatchSize = settings['train_batch_size']
        self.discountFactor = settings['discount_factor']
        self.updateStep = settings['update_step']
        self.maxActionNo = maxActionNo
        self.be = None
        self.historyBuffer = np.zeros((1, batchDimension[1], batchDimension[2], batchDimension[3]), dtype=np.float32)
        self.actionMat = np.zeros((self.trainBatchSize, self.maxActionNo))
        
        self.sess = tf.Session()
        
        self.x, self.y = self.buildNetwork('policy', True, maxActionNo)
        assert (len(tf.trainable_variables()) == 10),"Expected 10 trainable_variables"
        assert (len(tf.all_variables()) == 10),"Expected 10 total variables"
        self.x_target, self.y_target = self.buildNetwork('target', False, maxActionNo)
        assert (len(tf.trainable_variables()) == 10),"Expected 10 trainable_variables"
        assert (len(tf.all_variables()) == 20),"Expected 20 total variables"

        # build the variable copy ops
        self.update_target = []
        trainable_variables = tf.trainable_variables()
        all_variables = tf.all_variables()
        for i in range(0, len(trainable_variables)):
            self.update_target.append(all_variables[len(trainable_variables) + i].assign(trainable_variables[i]))

        self.a = tf.placeholder(tf.float32, shape=[None, maxActionNo])
        print('a %s' % (self.a.get_shape()))
        self.y_ = tf.placeholder(tf.float32, [None])
        print('y_ %s' % (self.y_.get_shape()))

        self.y_a = tf.reduce_sum(tf.mul(self.y, self.a), reduction_indices=1)
        print('y_a %s' % (self.y_a.get_shape()))

        difference = tf.abs(self.y_a - self.y_)
        quadratic_part = tf.clip_by_value(difference, 0.0, 1.0)
        linear_part = difference - quadratic_part
        errors = (0.5 * tf.square(quadratic_part)) + linear_part
        self.loss = tf.reduce_sum(errors)
        #self.loss = tf.reduce_mean(tf.square(self.y_a - self.y_))

        # (??) learning rate
        # Note tried gradient clipping with rmsprop with this particular loss function and it seemed to suck
        # Perhaps I didn't run it long enough
        #optimizer = GradientClippingOptimizer(tf.train.RMSPropOptimizer(args.learning_rate, decay=.95, epsilon=.01))
        optimizer = tf.train.RMSPropOptimizer(settings['learning_rate'], decay=.95, epsilon=.01)
        self.train_step = optimizer.minimize(self.loss)

        self.saver = tf.train.Saver(max_to_keep=25)

        # Initialize variables
        self.sess.run(tf.initialize_all_variables())
        self.sess.run(self.update_target) # is this necessary?

        print("Network Initialized")

    def addToHistoryBuffer(self, state):
        self.historyBuffer[0, :-1] = self.historyBuffer[0, 1:]
        self.historyBuffer[0, -1] = state

    def clearHistoryBuffer(self):
        self.historyBuffer.fill(0)

    def buildNetwork(self, name, trainable, numActions):
        
        print("Building network for %s trainable=%s" % (name, trainable))

        # First layer takes a screen, and shrinks by 2x
        x = tf.placeholder(tf.uint8, shape=[None, 84, 84, 4], name="screens")
        print(x)

        x_normalized = tf.to_float(x) / 255.0
        print(x_normalized)

        # Second layer convolves 32 8x8 filters with stride 4 with relu
        with tf.variable_scope("cnn1_" + name):
            W_conv1, b_conv1 = self.makeLayerVariables([8, 8, 4, 32], trainable, "conv1")

            h_conv1 = tf.nn.relu(tf.nn.conv2d(x_normalized, W_conv1, strides=[1, 4, 4, 1], padding='VALID') + b_conv1, name="h_conv1")
            print(h_conv1)

        # Third layer convolves 64 4x4 filters with stride 2 with relu
        with tf.variable_scope("cnn2_" + name):
            W_conv2, b_conv2 = self.makeLayerVariables([4, 4, 32, 64], trainable, "conv2")

            h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 2, 2, 1], padding='VALID') + b_conv2, name="h_conv2")
            print(h_conv2)

        # Fourth layer convolves 64 3x3 filters with stride 1 with relu
        with tf.variable_scope("cnn3_" + name):
            W_conv3, b_conv3 = self.makeLayerVariables([3, 3, 64, 64], trainable, "conv3")

            h_conv3 = tf.nn.relu(tf.nn.conv2d(h_conv2, W_conv3, strides=[1, 1, 1, 1], padding='VALID') + b_conv3, name="h_conv3")
            print(h_conv3)

        h_conv3_flat = tf.reshape(h_conv3, [-1, 7 * 7 * 64], name="h_conv3_flat")
        print(h_conv3_flat)

        # Fifth layer is fully connected with 512 relu units
        with tf.variable_scope("fc1_" + name):
            W_fc1, b_fc1 = self.makeLayerVariables([7 * 7 * 64, 512], trainable, "fc1")

            h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1, name="h_fc1")
            print(h_fc1)

        # Sixth (Output) layer is fully connected linear layer
        with tf.variable_scope("fc2_" + name):
            W_fc2, b_fc2 = self.makeLayerVariables([512, numActions], trainable, "fc2")

            y = tf.matmul(h_fc1, W_fc2) + b_fc2
            print(y)
            
        return x, y

    def makeLayerVariables(self, shape, trainable, name_suffix):
        stdv = 1.0 / math.sqrt(np.prod(shape[0:-1]))
        weights = tf.Variable(tf.random_uniform(shape, minval=-stdv, maxval=stdv), trainable=trainable, name='W_' + name_suffix)
        biases  = tf.Variable(tf.random_uniform([shape[-1]], minval=-stdv, maxval=stdv), trainable=trainable, name='W_' + name_suffix)
        return weights, biases
    
    def clipReward(self, reward):
            if reward > 0:
                return 1
            elif reward < 0:
                return -1
            else:
                return 0

    def predict(self, historyBuffer):
        ''' Get state-action value predictions for an observation 
        Args:
            observation: the observation
        '''
        return self.sess.run([self.y], {self.x: historyBuffer.transpose(0, 2, 3, 1)})[0]
        #return self.sess.run(self.policy_q_layer, feed_dict={self.observation:historyBuffer.transpose(0, 2, 3, 1)})[0]
        
    def train(self, minibatch, replayMemory, debug):
        if self.settings['prioritized_replay'] == True:
            prestates, actions, rewards, poststates, terminals, replayIndexes, heapIndexes, weights = minibatch
        else:
            prestates, actions, rewards, poststates, terminals = minibatch
        
        self.actionMat.fill(0)
        for i in range(self.trainBatchSize):
            self.actionMat[i, actions[i]] = 1

        y2 = self.y_target.eval(feed_dict={self.x_target: poststates.transpose(0, 2, 3, 1)}, session=self.sess)
        if self.settings['double_dqn'] == True:
            y3 = self.y.eval(feed_dict={self.x: poststates.transpose(0, 2, 3, 1)}, session=self.sess)

        y_ = np.zeros(self.trainBatchSize)
        
        for i in range(0, self.trainBatchSize):
            self.actionMat[i, actions[i]] = 1
            clippedReward = self.clipReward(rewards[i])
            if terminals[i]:
                y_[i] = clippedReward
            else:
                if self.settings['double_dqn'] == True:
                    maxIndex = np.argmax(y3[i])
                    y_[i] = clippedReward + self.discountFactor * y2[i][maxIndex]
                else:
                    y_[i] = clippedReward + self.discountFactor * np.max(y2[i])

        self.train_step.run(feed_dict={
            self.x: prestates.transpose(0, 2, 3, 1),
            self.a: self.actionMat,
            self.y_: y_
        }, session=self.sess)

    def updateModel(self):
        self.sess.run(self.update_target)

    def load(self, fileName):
        self.saver.restore(self.sess, fileName)
        self.updateModel()
        
    def save(self, fileName):
        self.saver.save(self.sess, fileName)
        

