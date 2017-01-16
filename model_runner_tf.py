import numpy as np
import os
import random
import math
import time
import threading
import traceback
import pickle
import tensorflow as tf
from network_model.model import Model, new_session

class ModelRunnerTF(object):
    def __init__(self, args,  max_action_no, batch_dimension, thread_no):
        self.args = args
        learning_rate = args.learning_rate
        rms_decay = args.rms_decay
        rms_epsilon =  args.rms_epsilon
        self.network = args.network
        self.thread_no = thread_no
        
        self.train_batch_size = args.train_batch_size
        self.discount_factor = args.discount_factor
        self.max_action_no = max_action_no
        self.be = None
        self.action_mat = np.zeros((self.train_batch_size, self.max_action_no))
        tf.logging.set_verbosity(tf.logging.WARN)
        
        self.sess = new_session()
        self.init_models(self.network, max_action_no, learning_rate, rms_decay, rms_epsilon)

    def init_models(self, network, max_action_no, learning_rate, rms_decay, rms_epsilon):        
        with tf.device(self.args.device):
            model_policy = Model(self.args, "policy", True, max_action_no, self.thread_no)
            model_target = Model(self.args, "target", False, max_action_no, self.thread_no)
    
            self.x_in, self.y, self.var_train = model_policy.x, model_policy.y, model_policy.variables
            self.x_target, self.y_target, self.var_target = model_target.x, model_target.y, model_target.variables

            # build the variable copy ops
            self.update_target = []
            for i in range(0, len(self.var_target)):
                self.update_target.append(self.var_target[i].assign(self.var_train[i]))
    
            self.a_in = tf.placeholder(tf.float32, shape=[None, max_action_no])
            self.y_ = tf.placeholder(tf.float32, [None])
            self.y_a = tf.reduce_sum(tf.mul(self.y, self.a_in), reduction_indices=1)
    
            optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=rms_decay, epsilon=rms_epsilon)
            self.difference = tf.abs(self.y_a - self.y_)

            if self.args.prioritized_replay == True:
                self.priority_weight = tf.placeholder(tf.float32, shape=[None], name="priority_weight")
            else:
                self.priority_weight = None
                
            self.loss = 0.5 * tf.square(self.difference)
                
            gvs = optimizer.compute_gradients(self.loss, var_list=self.var_train, grad_loss=self.priority_weight)
            new_gvs = [(tf.clip_by_value(grad, -1, 1), var) for grad, var in gvs]
            self.train_step = optimizer.apply_gradients(new_gvs)
            
            self.saver = tf.train.Saver(max_to_keep=100)
    
            # Initialize variables
            self.sess.run(tf.initialize_all_variables())
            self.sess.run(self.update_target) # is this necessary?

    def clip_reward(self, reward):
        if reward > self.args.clip_reward_high:
            return self.args.clip_reward_high
        elif reward < self.args.clip_reward_low:
            return self.args.clip_reward_low
        else:
            return reward

    def predict(self, history_buffer):
        return self.sess.run([self.y], {self.x_in: history_buffer})[0]
    
    def print_weights(self):
        for var in self.var_train:
            print ''
            print '[ ' + var.name + ']'
            print self.sess.run(var)
        
    def train(self, minibatch, replay_memory, learning_rate, debug):
        global global_step_no

        if self.args.prioritized_replay == True:
            prestates, actions, rewards, poststates, terminals, replay_indexes, heap_indexes, weights = minibatch
        else:
            prestates, actions, rewards, poststates, terminals = minibatch
        
        y2 = self.y_target.eval(feed_dict={self.x_target: poststates}, session=self.sess)
        
        if self.args.double_dqn == True:
            y3 = self.y.eval(feed_dict={self.x_in: poststates}, session=self.sess)

        self.action_mat.fill(0)
        y_ = np.zeros(self.train_batch_size)
        
        for i in range(self.train_batch_size):
            self.action_mat[i, actions[i]] = 1
            if self.args.clip_reward:
                reward = self.clip_reward(rewards[i])
            else:
                reward = rewards[i]
            if terminals[i]:
                y_[i] = reward
            else:
                if self.args.double_dqn == True:
                    max_index = np.argmax(y3[i])
                    y_[i] = reward + self.discount_factor * y2[i][max_index]
                else:
                    y_[i] = reward + self.discount_factor * np.max(y2[i])

        if self.args.prioritized_replay == True:
            delta_value, _, y_a = self.sess.run([self.difference, self.train_step, self.y_a], feed_dict={
                self.x_in: prestates,
                self.a_in: self.action_mat,
                self.y_: y_,
                self.priority_weight: weights
            })
            for i in range(self.train_batch_size):
                replay_memory.update_td(heap_indexes[i], abs(delta_value[i]))
                if debug:
                    print 'y_- y_a[%s]: %.5f, y_: %.5f, y_a: %.5f' % (i, (y_[i] - y_a[i]), y_[i], y_a[i]) 
                    print 'weight[%s]: %.5f, delta: %.5f, newDelta: %.5f' % (i, weights[i], delta_value[i], weights[i] * delta_value[i]) 
        else:
            self.sess.run(self.train_step, feed_dict={
                self.x_in: prestates,
                self.a_in: self.action_mat,
                self.y_: y_
            })

    def update_model(self):
        self.sess.run(self.update_target)

    def load(self, fileName):
        self.saver.restore(self.sess, fileName)
        self.update_model()
        
    def save(self, fileName):
        self.saver.save(self.sess, fileName)
        

