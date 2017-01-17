import numpy as np
import os
import random
import math
import time
import threading
import tensorflow as tf
from network_model.model_tf import ModelRunnerTF, new_session
from network_model.model_tf import Model

class ModelRunnerTFAsync(ModelRunnerTF):
    def __init__(self, global_list, args, max_action_no, thread_no):
        self.args = args
        self.screen_height = args.screen_height 
        self.screen_width = args.screen_width
        self.max_action_no = max_action_no
        self.discount_factor = args.discount_factor
        self.network = args.network
        self.thread_no = thread_no
        
        if global_list == None:
            self.play_mode = True
        else:
            self.play_mode = False
        
        if self.play_mode == False:
            self.sess, self.global_vars, self.global_optimizer, self.global_learning_rate = global_list
        else:
            self.sess = new_session()

        self.init_models()

        if self.thread_no == 0:
            self.saver = tf.train.Saver(max_to_keep=100)
        
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
    
            self.difference = tf.abs(self.y_a - self.y_)
            self.loss = tf.reduce_sum(tf.square(self.difference))
            
            self.init_gradients(self.loss, self.var_train)
            self.sess.run(tf.initialize_all_variables())

    def init_gradients(self, loss, var_train):
        if self.play_mode:
            return
        
        with tf.device(self.args.device):
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
                train_step_list.append(acc_gradient.assign_add(grad))
                new_grad_vars.append((tf.convert_to_tensor(acc_gradient, dtype=tf.float32), var))
                self.grad_list.append(acc_gradient)
                var_list.append(var)
            
            self.train_step = tf.group(*train_step_list)                
            
            self.reset_acc_gradients = tf.initialize_variables(acc_gradient_list)        
            self.apply_grads = self.global_optimizer.apply_gradients(new_grad_vars)
    
            sync_list = []
            for i in range(0, len(self.global_vars)):
                sync_list.append(var_train[i].assign(self.global_vars[i]))
            self.sync = tf.group(*sync_list)
    
    def init_save(self):
        save_model = self.new_model('save')
        self.save_vars = save_model.get_vars()
        with tf.device(self.args.device):
            sync_list = []
            for i in range(0, len(self.global_vars)):
                sync_list.append(self.save_vars[i].assign(self.global_vars[i]))
            self.save_sync = tf.group(*sync_list)
        
    def train(self, minibatch, replay_memory, learning_rate, debug):
        super(ModelRunnerTFAsync, self).train(minibatch, replay_memory, learning_rate, debug)

        self.sess.run(self.apply_grads, feed_dict={ self.global_learning_rate : learning_rate })
        self.sess.run(self.reset_acc_gradients)
        self.sess.run(self.sync)
    
    def load(self, fileName):
        self.saver.restore(self.sess, fileName)
        
    def copy_from_global_to_local(self):
        self.sess.run(self.sync)
        
                        
def load_global_vars(sess, global_vars, fileName):
    saver = tf.train.Saver(var_list=global_vars)
    saver.restore(sess, fileName)
