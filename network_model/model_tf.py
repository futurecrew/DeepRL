import numpy as np
import os
import random
import math
import time
import threading
import traceback
import pickle
import tensorflow as tf
import numpy as np
import math

def new_session(graph=None):
    config = tf.ConfigProto()
    # Use 25% of GPU memory to prevent from using it all 
    config.gpu_options.per_process_gpu_memory_fraction = 0.25
    return tf.Session(config=config, graph=graph)

class ModelRunnerTF(object):
    def __init__(self, args, max_action_no, thread_no):
        self.args = args
        self.thread_no = thread_no
        self.max_action_no = max_action_no
        self.action_mat = np.zeros((self.args.train_batch_size, self.max_action_no))
        tf.logging.set_verbosity(tf.logging.WARN)
        
        self.sess = new_session()
        self.init_models()
    
    def init_models(self):        
        with tf.device(self.args.device):
            model_policy = Model(self.args, "policy", True, self.max_action_no, self.thread_no)
            model_target = Model(self.args, "target", False, self.max_action_no, self.thread_no)
    
            self.x_in, self.y, self.var_train = model_policy.x, model_policy.y, model_policy.variables
            self.x_target, self.y_target, self.var_target = model_target.x, model_target.y, model_target.variables

            # build the variable copy ops
            self.update_target = []
            for i in range(0, len(self.var_target)):
                self.update_target.append(self.var_target[i].assign(self.var_train[i]))
    
            self.a_in = tf.placeholder(tf.float32, shape=[None, self.max_action_no])
            self.y_ = tf.placeholder(tf.float32, [None])
            self.y_a = tf.reduce_sum(tf.mul(self.y, self.a_in), reduction_indices=1)
    
            optimizer = tf.train.RMSPropOptimizer(self.args.learning_rate, decay=self.args.rms_decay, epsilon=self.args.rms_epsilon)
            self.difference = tf.abs(self.y_a - self.y_)

            quadratic_part = tf.clip_by_value(self.difference, 0.0, 1.0)
            linear_part = self.difference - quadratic_part
            self.errors = (0.5 * tf.square(quadratic_part)) + linear_part
            if self.args.prioritized_replay == True:
                self.priority_weight = tf.placeholder(tf.float32, shape=self.difference.get_shape(), name="priority_weight")
                self.errors = tf.mul(self.errors, self.priority_weight)
            self.loss = tf.reduce_sum(self.errors)
            self.train_step = optimizer.minimize(self.loss)     
                               
            self.saver = tf.train.Saver(max_to_keep=100)
    
            # Initialize variables
            self.sess.run(tf.initialize_all_variables())
            self.sess.run(self.update_target)

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
        if self.args.prioritized_replay == True:
            prestates, actions, rewards, poststates, terminals, replay_indexes, heap_indexes, weights = minibatch
        else:
            prestates, actions, rewards, poststates, terminals = minibatch
        
        y2 = self.y_target.eval(feed_dict={self.x_target: poststates}, session=self.sess)
        
        if self.args.double_dqn == True:
            y3 = self.y.eval(feed_dict={self.x_in: poststates}, session=self.sess)

        data_len = len(actions)        
        action_mat = np.zeros((data_len, self.max_action_no), dtype=np.uint8)
        y_ = np.zeros(data_len)
        
        for i in range(data_len):
            action_mat[i, actions[i]] = 1
            if self.args.clip_reward:
                reward = self.clip_reward(rewards[i])
            else:
                reward = rewards[i]
            if terminals[i]:
                y_[i] = reward
            else:
                if self.args.double_dqn == True:
                    max_index = np.argmax(y3[i])
                    y_[i] = reward + self.args.discount_factor * y2[i][max_index]
                else:
                    y_[i] = reward + self.args.discount_factor * np.max(y2[i])

        if self.args.prioritized_replay == True:
            delta_value, _ = self.sess.run([self.difference, self.train_step], feed_dict={
                self.x_in: prestates,
                self.a_in: action_mat,
                self.y_: y_,
                self.priority_weight: weights
            })
            for i in range(self.args.train_batch_size):
                replay_memory.update_td(heap_indexes[i], abs(delta_value[i]))
                if debug:
                    print 'weight[%s]: %.5f, delta: %.5f, newDelta: %.5f' % (i, weights[i], delta_value[i], weights[i] * delta_value[i]) 
        else:
            self.sess.run(self.train_step, feed_dict={
                self.x_in: prestates,
                self.a_in: action_mat,
                self.y_: y_
            })

    def update_model(self):
        self.sess.run(self.update_target)

    def load(self, fileName):
        self.saver.restore(self.sess, fileName)
        self.update_model()
        
    def save(self, fileName):
        self.saver.save(self.sess, fileName)
        
class Model(object):
    def __init__(self, args, name, trainable, action_no, thread_no):
        self.args = args
        self.network = args.network
        self.screen_height = args.screen_height
        self.screen_width = args.screen_width 
        self.history_len = args.screen_history
        self.action_no = action_no
        self.thread_no = thread_no
        with tf.device(args.device):
            self.build_network(name, args.network, trainable, action_no)
    
    def make_layer_variables(self, shape, trainable, name_suffix, weight_range=-1):
        if weight_range == -1:
            weight_range = 1.0 / math.sqrt(np.prod(shape[0:-1]))
        weights = tf.Variable(tf.random_uniform(shape, minval=-weight_range, maxval=weight_range), trainable=trainable, name='W_' + name_suffix)
        biases  = tf.Variable(tf.random_uniform([shape[-1]], minval=-weight_range, maxval=weight_range), trainable=trainable, name='b_' + name_suffix)
        return weights, biases
    
    def print_log(self, log):
        if self.thread_no == 0:
            print log
            
    def build_network(self, name, network, trainable, num_actions):
        if network == 'nips':
            self.build_network_nips(name, trainable, num_actions)
        else:
            self.build_network_nature(name, trainable, num_actions)
        
    def build_network_nature(self, name, trainable, num_actions):
        self.print_log("Building network nature for %s trainable=%s" % (name, trainable))
    
        with tf.variable_scope(name):
            # First layer takes a screen, and shrinks by 2x
            x = tf.placeholder(tf.uint8, shape=[None, self.screen_height, self.screen_width, self.history_len], name="screens")
            self.print_log(x)
        
            x_normalized = tf.to_float(x) / 255.0
            self.print_log(x_normalized)
    
            # Second layer convolves 32 8x8 filters with stride 4 with relu
            W_conv1, b_conv1 = self.make_layer_variables([8, 8, self.history_len, 32], trainable, "conv1")
    
            h_conv1 = tf.nn.relu(tf.nn.conv2d(x_normalized, W_conv1, strides=[1, 4, 4, 1], padding='VALID') + b_conv1, name="h_conv1")
            self.print_log(h_conv1)
    
            # Third layer convolves 64 4x4 filters with stride 2 with relu
            W_conv2, b_conv2 = self.make_layer_variables([4, 4, 32, 64], trainable, "conv2")
    
            h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 2, 2, 1], padding='VALID') + b_conv2, name="h_conv2")
            self.print_log(h_conv2)
    
            # Fourth layer convolves 64 3x3 filters with stride 1 with relu
            W_conv3, b_conv3 = self.make_layer_variables([3, 3, 64, 64], trainable, "conv3")
    
            h_conv3 = tf.nn.relu(tf.nn.conv2d(h_conv2, W_conv3, strides=[1, 1, 1, 1], padding='VALID') + b_conv3, name="h_conv3")
            self.print_log(h_conv3)
    
            conv_out_size = np.prod(h_conv3._shape[1:]).value
    
            h_conv3_flat = tf.reshape(h_conv3, [-1, conv_out_size], name="h_conv3_flat")
            self.print_log(h_conv3_flat)
    
            # Fifth layer is fully connected with 512 relu units
            W_fc1, b_fc1 = self.make_layer_variables([conv_out_size, 512], trainable, "fc1")
    
            h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1, name="h_fc1")
            self.print_log(h_fc1)
    
            W_fc2, b_fc2 = self.make_layer_variables([512, num_actions], trainable, "fc2")
    
            y = tf.matmul(h_fc1, W_fc2) + b_fc2
            self.print_log(y)
        
        self.x = x
        self.y = y
        self.variables = [W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2]
        
    def build_network_nips(self, name, trainable, num_actions):
        self.print_log("Building network nips for %s trainable=%s" % (name, trainable))
    
        with tf.variable_scope(name):
            # First layer takes a screen, and shrinks by 2x
            x = tf.placeholder(tf.uint8, shape=[None, self.screen_height, self.screen_width, self.history_len], name="screens")
            self.print_log(x)
        
            x_normalized = tf.to_float(x) / 255.0
            self.print_log(x_normalized)
    
            # Second layer convolves 16 8x8 filters with stride 4 with relu
            W_conv1, b_conv1 = self.make_layer_variables([8, 8, self.history_len, 16], trainable, "conv1")
    
            h_conv1 = tf.nn.relu(tf.nn.conv2d(x_normalized, W_conv1, strides=[1, 4, 4, 1], padding='VALID') + b_conv1, name="h_conv1")
            self.print_log(h_conv1)
    
            # Third layer convolves 32 4x4 filters with stride 2 with relu
            W_conv2, b_conv2 = self.make_layer_variables([4, 4, 16, 32], trainable, "conv2")
    
            h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 2, 2, 1], padding='VALID') + b_conv2, name="h_conv2")
            self.print_log(h_conv2)
    
            conv_out_size = np.prod(h_conv2._shape[1:]).value
    
            h_conv2_flat = tf.reshape(h_conv2, [-1, conv_out_size], name="h_conv2_flat")
            self.print_log(h_conv2_flat)
    
            # Fourth layer is fully connected with 256 relu units
            W_fc1, b_fc1 = self.make_layer_variables([conv_out_size, 256], trainable, "fc1")
    
            h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1, name="h_fc1")
            self.print_log(h_fc1)
    
            W_fc2, b_fc2 = self.make_layer_variables([256, num_actions], trainable, "fc2")
    
            y = tf.matmul(h_fc1, W_fc2) + b_fc2
            self.print_log(y)
        
        self.x = x
        self.y = y
        self.variables = [W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2]

    def get_vars(self):
        return self.variables
    
    def prepare_global(self, rms_decay, rms_epsilon):
        global_sess = new_session()
        global_learning_rate = tf.placeholder("float")
        global_optimizer = tf.train.RMSPropOptimizer(global_learning_rate, decay=rms_decay, epsilon=rms_epsilon)
        global_vars = self.get_vars()
    
        return global_sess, global_vars, global_optimizer, global_learning_rate
    
    def init_global(self, global_sess):
        global_sess.run(tf.initialize_all_variables())
