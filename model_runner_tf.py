import numpy as np
import os
import random
import math
import time
import threading
import traceback
import pickle
import tensorflow as tf

class ModelRunnerTF(object):
    def __init__(self, args,  max_action_no, batch_dimension):
        self.args = args
        learning_rate = args.learning_rate
        rms_decay = args.rms_decay
        rms_epsilon =  args.rms_epsilon
        self.network_type = args.network_type
        
        self.step_no = 0
        self.train_batch_size = args.train_batch_size
        self.discount_factor = args.discount_factor
        self.max_action_no = max_action_no
        self.be = None
        self.action_mat = np.zeros((self.train_batch_size, self.max_action_no))

        self.init_models(self.network_type, max_action_no, learning_rate, rms_decay, rms_epsilon)

    def init_models(self, network_type, max_action_no, learning_rate, rms_decay, rms_epsilon):
        self.sess = self.new_session()

        self.x_in, self.y, self.var_train = self.build_network('policy', network_type, True, max_action_no)
        self.x_target, self.y_target, self.var_target = self.build_network('target', network_type, False, max_action_no)

        # build the variable copy ops
        self.update_target = []
        for i in range(0, len(self.var_target)):
            self.update_target.append(self.var_target[i].assign(self.var_train[i]))

        self.a_in = tf.placeholder(tf.float32, shape=[None, max_action_no])
        print('a %s' % (self.a_in.get_shape()))
        self.y_ = tf.placeholder(tf.float32, [None])
        print('y_ %s' % (self.y_.get_shape()))

        self.y_a = tf.reduce_sum(tf.mul(self.y, self.a_in), reduction_indices=1)
        print('y_a %s' % (self.y_a.get_shape()))

        optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=rms_decay, epsilon=rms_epsilon)
        self.difference = tf.abs(self.y_a - self.y_)
        quadratic_part = tf.clip_by_value(self.difference, 0.0, 1.0)
        linear_part = self.difference - quadratic_part
        #self.errors = (0.5 * tf.square(quadratic_part)) + linear_part
        self.errors = 0.5 * tf.square(self.difference)
        if self.args.prioritized_replay == True:
            self.priority_weight = tf.placeholder(tf.float32, shape=self.errors.get_shape(), name="priority_weight")
            self.errors2 = tf.mul(self.errors, self.priority_weight)
        else:
            self.errors2 = self.errors
        self.loss = tf.reduce_sum(tf.clip_by_value(self.errors2, 0.0, 1.0))
        self.train_step = optimizer.minimize(self.loss)
        self.saver = tf.train.Saver(max_to_keep=25)

        # Initialize variables
        self.sess.run(tf.initialize_all_variables())
        self.sess.run(self.update_target) # is this necessary?

    def clip_reward(self, reward):
            if reward > 0:
                return 1
            elif reward < 0:
                return -1
            else:
                return 0

    def predict(self, history_buffer):
        return self.sess.run([self.y], {self.x_in: history_buffer})[0]
        
    def train(self, minibatch, replay_memory, debug):
        global global_step_no

        if self.args.prioritized_replay == True:
            prestates, actions, rewards, poststates, terminals, replay_indexes, heap_indexes, weights = minibatch
        else:
            prestates, actions, rewards, poststates, terminals = minibatch
        
        self.step_no += 1
        
        y2 = self.y_target.eval(feed_dict={self.x_target: poststates}, session=self.sess)
        
        if self.args.double_dqn == True:
            y3 = self.y.eval(feed_dict={self.x_in: poststates}, session=self.sess)

        self.action_mat.fill(0)
        y_ = np.zeros(self.train_batch_size)
        
        for i in range(self.train_batch_size):
            self.action_mat[i, actions[i]] = 1
            clipped_reward = self.clip_reward(rewards[i])
            if terminals[i]:
                y_[i] = clipped_reward
            else:
                if self.args.double_dqn == True:
                    max_index = np.argmax(y3[i])
                    y_[i] = clipped_reward + self.discount_factor * y2[i][max_index]
                else:
                    y_[i] = clipped_reward + self.discount_factor * np.max(y2[i])

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
        

