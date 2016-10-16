import numpy as np
import os
import random
import math
import time
import threading
import tensorflow as tf
from network_model import new_session

class ModelRunnerTFAsync():
    def __init__(self, global_list, settings,  max_action_no, thread_no):
        self.max_action_no = max_action_no
        self.discount_factor = settings['discount_factor']
        self.network_type = settings['network_type']
        self.thread_no = thread_no
        
        if global_list == None:
            self.play_mode = True
        else:
            self.play_mode = False
        
        if self.play_mode == False:
            self.sess, self.global_vars, self.global_optimizer, self.global_learning_rate = global_list
            if self.thread_no == 0:
                self.init_save()
        else:
            self.sess = new_session()

        self.init_models()
        self.saver = tf.train.Saver(max_to_keep=25)
        
    def init_models(self):
        self.x_in, self.y, self.var_train = self.build_network('policy', self.network_type, True, self.max_action_no)
        self.x_target, self.y_target, self.var_target = self.build_network('target', self.network_type, False, self.max_action_no)

        self.a_in = tf.placeholder(tf.float32, shape=[None, self.max_action_no])
        print('a_in %s' % (self.a_in.get_shape()))
        self.y_ = tf.placeholder(tf.float32, [None])
        print('y_ %s' % (self.y_.get_shape()))

        self.y_a = tf.reduce_sum(tf.mul(self.y, self.a_in), reduction_indices=1)
        print('y_a %s' % (self.y_a.get_shape()))

        self.difference = tf.abs(self.y_a - self.y_)
        self.errors = 0.5 * tf.square(self.difference)
        self.priority_weight = tf.placeholder(tf.float32, shape=self.errors.get_shape(), name="priority_weight")
        loss = tf.reduce_sum(self.errors)
        
        self.init_gradients(loss)
        self.sess.run(tf.initialize_all_variables())

    def init_gradients(self, loss, var_train):
        if self.play_mode:
            return
        
        var_refs = [v.ref() for v in var_train]
        train_gradients = tf.gradients(
            loss, var_refs,
            gate_gradients=False,
            aggregation_method=None,
            colocate_gradients_with_ops=False)

        acc_gradient_list = []
        train_step_list = []
        new_grad_vars = []
        self.grad_list = []
        var_list = []
        for grad, var in zip(train_gradients, self.global_vars):
            acc_gradient = tf.Variable(tf.zeros(grad.get_shape()), trainable=False)
            acc_gradient_list.append(acc_gradient)
            train_step_list.append(acc_gradient.assign_add(tf.clip_by_value(grad, -1.0, 1.0)))
            #train_step_list.append(acc_gradient.assign_add(grad))
            new_grad_vars.append((tf.convert_to_tensor(acc_gradient, dtype=tf.float32), var))
            self.grad_list.append(acc_gradient)
            var_list.append(var)
        
        self.train_step = tf.group(*train_step_list)                
        self.reset_acc_gradients = tf.initialize_variables(acc_gradient_list)                       
        self.apply_grads = self.global_optimizer.apply_gradients(new_grad_vars)
        #self.apply_grads = self.global_optimizer.apply_gradients(var_list, self.grad_list)

        # build the sync ops
        sync_list = []
        for i in range(0, len(self.global_vars)):
            sync_list.append(var_train[i].assign(self.global_vars[i]))
        self.sync = tf.group(*sync_list)
    
    def init_save(self):
        save_model = self.new_model('save')
        self.save_vars = save_model.get_vars()
        sync_list = []
        for i in range(0, len(self.global_vars)):
            sync_list.append(self.save_vars[i].assign(self.global_vars[i]))
        self.save_sync = tf.group(*sync_list)
        #self.save_sess = new_session()
        
    def train(self, minibatch, replay_memory, debug):
        if self.settings['prioritized_replay'] == True:
            prestates, actions, rewards, poststates, terminals, replay_indexes, heap_indexes, weights = minibatch
        else:
            prestates, actions, rewards, poststates, terminals = minibatch
        
        self.step_no += 1
        
        y2 = self.y_target.eval(feed_dict={self.x_target: poststates}, session=self.sess)
        
        if self.settings['double_dqn'] == True:
            y3 = self.y.eval(feed_dict={self.x_in: poststates}, session=self.sess)

        self.action_mat.fill(0)
        y_ = np.zeros(self.train_batch_size)
        
        for i in range(self.train_batch_size):
            self.action_mat[i, actions[i]] = 1
            clipped_reward = self.clip_reward(rewards[i])
            if terminals[i]:
                y_[i] = clipped_reward
            else:
                if self.settings['double_dqn'] == True:
                    max_index = np.argmax(y3[i])
                    y_[i] = clipped_reward + self.discount_factor * y2[i][max_index]
                else:
                    y_[i] = clipped_reward + self.discount_factor * np.max(y2[i])

        self.sess.run(self.train_step, feed_dict={
            self.x_in: prestates,
            self.a_in: self.action_mat,
            self.y_: y_
        })

        self.sess.run(self.apply_grads)
        self.sess.run(self.reset_acc_gradients)
        self.sess.run(self.sync)

    def clip_reward(self, reward):
            if reward > 0:
                return 1
            elif reward < 0:
                return -1
            else:
                return 0
                        
    def load(self, fileName):
        self.saver.restore(self.sess, fileName)

    def save(self, fileName):
        self.sess.run(tf.initialize_variables(self.save_vars))
        self.sess.run(self.save_sync)
        self.saver.save(self.sess, fileName)
        
        print 'saved to %s' % fileName
