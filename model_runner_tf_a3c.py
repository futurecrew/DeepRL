import numpy as np
import os
import random
import math
import time
import threading
import tensorflow as tf
from model_runner_tf_async import ModelRunnerTFAsync
from network_model import ModelA3C
            
class ModelRunnerTFA3C(ModelRunnerTFAsync):
    def init_models(self):
        self.model = self.new_model('net-' + str(self.thread_no))

        with tf.device(self.args.device):
            self.a_in = tf.placeholder(tf.float32, shape=[None, self.max_action_no])
            self.v_in = tf.placeholder(tf.float32, shape=[None])
            self.td_in = tf.placeholder(tf.float32, shape=[None])
            self.x_in = self.model.x
            self.y_class = self.model.y_class
            self.v = self.model.v
            
            loss = self.get_loss()
            self.init_gradients(loss, self.model.get_vars())
        
    def new_model(self, name):
        return ModelA3C(self.args.device, name, self.network_type, True, self.max_action_no)
    
    def get_loss(self):
        with tf.device(self.args.device):
            """
            y_class_a = tf.reduce_sum(tf.mul(self.y_class, self.a_in), reduction_indices=1)
            self.log_y = tf.log(y_class_a)
            class_loss = -1 * tf.reduce_sum(self.log_y * self.td_in)  
            value_loss = tf.reduce_sum(tf.square(self.v - self.v_in))
            """
    
            # avoid NaN with clipping when value in pi becomes zero
            log_pi = tf.log(tf.clip_by_value(self.y_class, 1e-20, 1.0))
      
            # policy entropy
            entropy = -tf.reduce_sum(self.y_class * log_pi, reduction_indices=1)
      
            # policy loss (output)  (Adding minus, because the original paper's objective function is for gradient ascent, but we use gradient descent optimizer.)
            policy_loss = - tf.reduce_sum( tf.reduce_sum( tf.mul( log_pi, self.a_in ), reduction_indices=1 ) * self.td_in + entropy * 0.01 )
            value_loss = 0.5 * tf.nn.l2_loss(self.v - self.v_in)
            loss = policy_loss + value_loss
        
        return loss
        
    def predict_action_state(self, state):
        y_class, v = self.sess.run([self.y_class, self.v], {self.x_in: state})
        return y_class[0], v[0]

    def predict_state(self, state):
        v = self.sess.run(self.v, {self.x_in: state})
        return v[0]

    def predict(self, state):
        y_class = self.sess.run(self.y_class, {self.x_in: state})
        return y_class[0]

    def train(self, prestates, v_pres, actions, rewards, terminals, v_post, learning_rate):
        data_len = len(actions)

        action_mat = np.zeros((data_len, self.max_action_no), dtype=np.uint8)
        v_in = np.zeros(data_len)
        td_in = np.zeros(data_len)
        
        R = v_post
        for i in range(data_len):
            action_mat[i, actions[i]] = 1
            clipped_reward = self.clip_reward(rewards[i])
            v_in[i] = clipped_reward + self.discount_factor * R
            td_in[i] = v_in[i] - v_pres[i]
            R = v_in[i]

        self.sess.run(self.train_step, feed_dict={
            self.x_in: prestates,
            self.a_in: action_mat,
            self.v_in: v_in,
            self.td_in: td_in
        })
        
        self.sess.run( self.apply_grads,
              feed_dict = { self.global_learning_rate: learning_rate } )
        self.sess.run(self.reset_acc_gradients)
        self.sess.run(self.sync)
        
