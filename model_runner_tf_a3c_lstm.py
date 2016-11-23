import numpy as np
import os
import random
import math
import time
import threading
import tensorflow as tf
from model_runner_tf_async import ModelRunnerTFAsync
from network_model import ModelA3CLstm
            
class ModelRunnerTFA3CLstm(ModelRunnerTFAsync):
    def init_models(self):
        self.model = self.new_model('net-' + str(self.thread_no))

        with tf.device(self.args.device):
            self.a_in = tf.placeholder(tf.float32, shape=[None, self.max_action_no])
            self.v_in = tf.placeholder(tf.float32, shape=[None])
            self.td_in = tf.placeholder(tf.float32, shape=[None])
            self.x_in = self.model.x
            self.y_class = self.model.y_class
            self.v = self.model.v
            
            self.lstm_init_state = self.model.lstm_init_state
            self.lstm_next_state = self.model.lstm_next_state
            self.sequence_length = self.model.sequence_length
            self.lstm_hidden_size = 256
            
            self.reset_lstm_state()
    
            loss = self.get_loss()
            self.init_gradients(loss, self.model.get_vars())
        
    def new_model(self, name):
        return ModelA3CLstm(self.args.device, name, self.network, self.screen_height, self.screen_width, self.args.screen_history, True, self.max_action_no)
    
    def get_loss(self):
        with tf.device(self.args.device):
            """
            y_class_a = tf.reduce_sum(tf.mul(self.y_class, self.a_in), reduction_indices=1)
            self.log_y = tf.log(y_class_a)
            class_loss = -1 * tf.reduce_sum(self.log_y * self.td_in)  
            value_loss = tf.reduce_sum(tf.square(self.v - self.v_in))
            """
    
            print 'self.y_class : %s' % self.y_class
            print 'self.a_in : %s' % self.a_in
            print 'self.td_in : %s' % self.td_in
            print 'self.v : %s' % self.v
            print 'self.v_in : %s' % self.v_in
            
            """
            self.y_class_sequence = self.y_class[:self.sequence_length, :]
            self.a_in_sequence = self.a_in[:self.sequence_length, :]
            self.td_in_sequence = self.td_in[:self.sequence_length]
            self.v_sequence = self.v[:self.sequence_length]
            self.v_in_sequence = self.v_in[:self.sequence_length]
    
            print 'self.y_class_sequence : %s' % self.y_class_sequence
            print 'self.a_in_sequence : %s' % self.a_in_sequence
            print 'self.td_in_sequence : %s' % self.td_in_sequence
            print 'self.v_sequence : %s' % self.v_sequence
            print 'self.v_in_sequence : %s' % self.v_in_sequence
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
        y_class, v, self.lstm_state_value = self.sess.run([self.y_class, self.v, self.lstm_next_state], feed_dict={
                        self.x_in: state,
                        self.lstm_init_state.c : self.lstm_state_value[0],
                        self.lstm_init_state.h : self.lstm_state_value[1],
                        self.sequence_length : [1],
                        })
        return y_class[0], v[0]

    def predict_state(self, state):
        # don't update self.lstm_state_value
        # because this function does not process to a new screen frame
        v, _ = self.sess.run([self.v, self.lstm_next_state], feed_dict={
                        self.x_in: state,
                        self.lstm_init_state.c : self.lstm_state_value[0],
                        self.lstm_init_state.h : self.lstm_state_value[1],
                        self.sequence_length : [1],
                        })
        return v[0]

    def predict(self, state):
        y_class, self.lstm_state_value = self.sess.run([self.y_class, self.lstm_next_state], feed_dict={
                        self.x_in: state,
                        self.lstm_init_state.c : self.lstm_state_value[0],
                        self.lstm_init_state.h : self.lstm_state_value[1],
                        self.sequence_length : [1],
                        })
        return y_class[0]

    def reset_lstm_state(self):
        self.lstm_state_value = self.sess.run(self.lstm_init_state)

    def get_lstm_state(self):
        return self.lstm_state_value

    def set_lstm_state(self, lstm_state_value_c, lstm_state_value_h):
        self.lstm_state_value[0] = lstm_state_value_c
        self.lstm_state_value[1] = lstm_state_value_h

    def train(self, prestates, v_pres, actions, rewards, terminals, v_post, learning_rate, lstm_state_value):
        data_len = len(actions)
        self.lstm_state_value = lstm_state_value
        
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

        # Make input data time sequential for LSTM input
        x_input = prestates[::-1, :, :, :]
        action_mat = action_mat[::-1, :]
        v_in = v_in[::-1]
        td_in = td_in[::-1]
        
        _, self.lstm_state_value = self.sess.run([self.train_step, self.lstm_next_state], feed_dict={
            self.x_in: x_input,
            self.a_in: action_mat,
            self.v_in: v_in,
            self.td_in: td_in,
            self.lstm_init_state.c : self.lstm_state_value[0],
            self.lstm_init_state.h : self.lstm_state_value[1],
            self.sequence_length : [data_len],
        })
        
        self.sess.run( self.apply_grads,
              feed_dict = { self.global_learning_rate: learning_rate } )
        self.sess.run(self.reset_acc_gradients)
        self.sess.run(self.sync)
        
