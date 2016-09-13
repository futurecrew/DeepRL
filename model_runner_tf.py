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
        self.historyBuffer = np.zeros(batchDimension, dtype=np.float32)
        self.actionMat = np.zeros((self.trainBatchSize, self.maxActionNo))

        # input placeholders
        self.observation = tf.placeholder(tf.float32, shape=[None, batchDimension[2], batchDimension[3], batchDimension[1]], name="observation")
        self.actions = tf.placeholder(tf.float32, shape=[None, maxActionNo], name="actions") # one-hot matrix because tf.gather() doesn't support multidimensional indexing yet
        self.rewards = tf.placeholder(tf.float32, shape=[None], name="rewards")
        self.next_observation = tf.placeholder(tf.float32, shape=[None, batchDimension[2], batchDimension[3], batchDimension[1]], name="next_observation")
        self.terminals = tf.placeholder(tf.float32, shape=[None], name="terminals")
        self.normalized_observation = self.observation / 255.0
        self.normalized_next_observation = self.next_observation / 255.0

        conv_kernel_shapes = []
        conv_kernel_shapes.append([8, 8, 4, 32])
        conv_kernel_shapes.append([4, 4, 32,64])
        conv_kernel_shapes.append([3, 3, 64, 64])
        
        conv_strides = []
        conv_strides.append([1, 4, 4, 1])
        conv_strides.append([1, 2, 2, 1])
        conv_strides.append([1, 1, 1, 1])
        
        dense_layer_shapes = []
        dense_layer_shapes.append([7 * 7 * 64, 512])
        
        num_conv_layers = len(conv_kernel_shapes)
        assert(num_conv_layers == len(conv_strides))
        num_dense_layers = len(dense_layer_shapes)

        last_policy_layer = None
        last_target_layer = None
        self.update_target = []
        self.policy_network_params = []
        self.param_names = []

        # initialize convolutional layers
        for layer in range(num_conv_layers):
            policy_input = None
            target_input = None
            if layer == 0:
                policy_input = self.normalized_observation
                target_input = self.normalized_next_observation
            else:
                policy_input = last_policy_layer
                target_input = last_target_layer

            last_layers = self.conv_relu(policy_input, target_input, 
                conv_kernel_shapes[layer], conv_strides[layer], layer)
            last_policy_layer = last_layers[0]
            last_target_layer = last_layers[1]

        # initialize fully-connected layers
        for layer in range(num_dense_layers):
            policy_input = None
            target_input = None
            if layer == 0:
                input_size = dense_layer_shapes[0][0]
                policy_input = tf.reshape(last_policy_layer, shape=[-1, input_size])
                target_input = tf.reshape(last_target_layer, shape=[-1, input_size])
            else:
                policy_input = last_policy_layer
                target_input = last_target_layer

            last_layers = self.dense_relu(policy_input, target_input, dense_layer_shapes[layer], layer)
            last_policy_layer = last_layers[0]
            last_target_layer = last_layers[1]


        # initialize q_layer
        last_layers = self.dense_linear(
            last_policy_layer, last_target_layer, [dense_layer_shapes[-1][-1], maxActionNo])
        self.policy_q_layer = last_layers[0]
        self.target_q_layer = last_layers[1]

        self.loss = self.build_loss(1.0, maxActionNo, False)

        self.train_op = tf.train.RMSPropOptimizer(
            settings['learning_rate'], decay=settings['rms_decay'], momentum=0.0, epsilon=0.01).minimize(self.loss)

        self.saver = tf.train.Saver(self.policy_network_params)

        """
        if not args.watch:
            param_hists = [tf.histogram_summary(name, param) for name, param in zip(self.param_names, self.policy_network_params)]
            self.param_summaries = tf.merge_summary(param_hists)

        # start tf session
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33333)  # avoid using all vram for GTX 970
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        """
        self.sess = tf.Session()

        """
        if args.watch:
            print("Loading Saved Network...")
            load_path = tf.train.latest_checkpoint(self.path)
            self.saver.restore(self.sess, load_path)
            print("Network Loaded")        
        else:
            self.sess.run(tf.initialize_all_variables())
            print("Network Initialized")
            self.summary_writer = tf.train.SummaryWriter('../records/' + args.game + '/' + args.agent_type + '/' + args.agent_name + '/params', self.sess.graph)
        """
        self.sess.run(tf.initialize_all_variables())
        print("Network Initialized")

    def addToHistoryBuffer(self, state):
        self.historyBuffer[0, :-1] = self.historyBuffer[0, 1:]
        self.historyBuffer[0, -1] = state

    def clearHistoryBuffer(self):
        self.historyBuffer.fill(0)

    def conv_relu(self, policy_input, target_input, kernel_shape, stride, layer_num):
        ''' Build a convolutional layer
        Args:
            input_layer: input to convolutional layer - must be 4d
            target_input: input to layer of target network - must also be 4d
            kernel_shape: tuple for filter shape: (filter_height, filter_width, in_channels, out_channels)
            stride: tuple for stride: (1, vert_stride. horiz_stride, 1)
        '''
        name = 'conv' + str(layer_num + 1)
        with tf.variable_scope(name):

            # weights = tf.Variable(tf.truncated_normal(kernel_shape, stddev=0.01), name=(name + "_weights"))
            weights = self.get_weights(kernel_shape, name)
            # biases = tf.Variable(tf.fill([kernel_shape[-1]], 0.1), name=(name + "_biases"))
            biases = self.get_biases(kernel_shape, name)

            activation = tf.nn.relu(tf.nn.conv2d(policy_input, weights, stride, 'VALID') + biases)

            target_weights = tf.Variable(weights.initialized_value(), trainable=False, name=("target_" + name + "_weights"))
            target_biases = tf.Variable(biases.initialized_value(), trainable=False, name=("target_" + name + "_biases"))

            target_activation = tf.nn.relu(tf.nn.conv2d(target_input, target_weights, stride, 'VALID') + target_biases)

            self.update_target.append(target_weights.assign(weights))
            self.update_target.append(target_biases.assign(biases))

            self.policy_network_params.append(weights)
            self.policy_network_params.append(biases)
            self.param_names.append(name + "_weights")
            self.param_names.append(name + "_biases")

            return [activation, target_activation]


    def dense_relu(self, policy_input, target_input, shape, layer_num):
        ''' Build a fully-connected relu layer 
        Args:
            input_layer: input to dense layer
            target_input: input to layer of target network
            shape: tuple for weight shape (num_input_nodes, num_layer_nodes)
        '''
        name = 'dense' + str(layer_num + 1)
        with tf.variable_scope(name):

            # weights = tf.Variable(tf.truncated_normal(shape, stddev=0.01), name=(name + "_weights"))
            weights = self.get_weights(shape, name)
            # biases = tf.Variable(tf.fill([shape[-1]], 0.1), name=(name + "_biases"))
            biases = self.get_biases(shape, name)

            activation = tf.nn.relu(tf.matmul(policy_input, weights) + biases)

            target_weights = tf.Variable(weights.initialized_value(), trainable=False, name=("target_" + name + "_weights"))
            target_biases = tf.Variable(biases.initialized_value(), trainable=False, name=("target_" + name + "_biases"))

            target_activation = tf.nn.relu(tf.matmul(target_input, target_weights) + target_biases)

            self.update_target.append(target_weights.assign(weights))
            self.update_target.append(target_biases.assign(biases))

            self.policy_network_params.append(weights)
            self.policy_network_params.append(biases)
            self.param_names.append(name + "_weights")
            self.param_names.append(name + "_biases")

            return [activation, target_activation]


    def dense_linear(self, policy_input, target_input, shape):
        ''' Build the fully-connected linear output layer 
        Args:
            input_layer: last hidden layer
            target_input: last hidden layer of target network
            shape: tuple for weight shape (num_input_nodes, num_actions)
        '''
        name = 'q_layer'
        with tf.variable_scope(name):

            # weights = tf.Variable(tf.truncated_normal(shape, stddev=0.01), name=(name + "_weights"))
            weights = self.get_weights(shape, name)
            # biases = tf.Variable(tf.fill([shape[-1]], 0.1), name=(name + "_biases"))
            biases = self.get_biases(shape, name)


            activation = tf.matmul(policy_input, weights) + biases

            target_weights = tf.Variable(weights.initialized_value(), trainable=False, name=("target_" + name + "_weights"))
            target_biases = tf.Variable(biases.initialized_value(), trainable=False, name=("target_" + name + "_biases"))

            target_activation = tf.matmul(target_input, target_weights) + target_biases

            self.update_target.append(target_weights.assign(weights))
            self.update_target.append(target_biases.assign(biases))

            self.policy_network_params.append(weights)
            self.policy_network_params.append(biases)
            self.param_names.append(name + "_weights")
            self.param_names.append(name + "_biases")

            return [activation, target_activation]
        
    def predict(self, historyBuffer):
        ''' Get state-action value predictions for an observation 
        Args:
            observation: the observation
        '''
        a = self.sess.run(self.policy_q_layer, feed_dict={self.observation:historyBuffer.transpose(0, 2, 3, 1)})
        b = np.squeeze(a)
        return b


    def build_loss(self, error_clip, num_actions, double_dqn):
        ''' build loss graph '''
        with tf.name_scope("loss"):

            predictions = tf.reduce_sum(tf.mul(self.policy_q_layer, self.actions), 1)
            
            max_action_values = None
            max_action_values = tf.reduce_max(self.target_q_layer, 1)

            targets = tf.stop_gradient(self.rewards + (self.discountFactor * max_action_values * (1 - self.terminals)))

            difference = tf.abs(predictions - targets)

            if error_clip >= 0:
                quadratic_part = tf.clip_by_value(difference, 0.0, error_clip)
                linear_part = difference - quadratic_part
                errors = (0.5 * tf.square(quadratic_part)) + (error_clip * linear_part)
            else:
                errors = (0.5 * tf.square(difference))

            return tf.reduce_sum(errors)


    def train(self, minibatch, replayMemory, debug):
        if self.settings['prioritized_replay'] == True:
            prestates, actions, rewards, poststates, terminals, replayIndexes, heapIndexes, weights = minibatch
        else:
            prestates, actions, rewards, poststates, terminals = minibatch
        
        self.actionMat.fill(0)
        for i in range(self.trainBatchSize):
            self.actionMat[i, actions[i]] = 1
        
        loss = self.sess.run([self.train_op, self.loss], 
            feed_dict={self.observation:prestates.transpose(0, 2, 3, 1), \
                       self.actions:self.actionMat, \
                       self.rewards:rewards, \
                       self.next_observation:poststates.transpose(0, 2, 3, 1), \
                       self.terminals:terminals})[1]

        return loss

    def get_weights(self, shape, name):
        fan_in = np.prod(shape[0:-1])
        std = 1 / math.sqrt(fan_in)
        return tf.Variable(tf.random_uniform(shape, minval=(-std), maxval=std), name=(name + "_weights"))

    def get_biases(self, shape, name):
        fan_in = np.prod(shape[0:-1])
        std = 1 / math.sqrt(fan_in)
        return tf.Variable(tf.random_uniform([shape[-1]], minval=(-std), maxval=std), name=(name + "_biases"))

    def record_params(self, step):
        summary_string = self.sess.run(self.param_summaries)
        self.summary_writer.add_summary(summary_string, step)
        
    def updateModel(self):
        self.sess.run(self.update_target)

    def load(self, fileName):
        self.saver.restore(self.sess, fileName)
        self.updateModel()
        
    def save(self, fileName):
        self.saver.save(self.sess, fileName)
        

