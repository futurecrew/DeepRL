import tensorflow as tf
import numpy as np
import math

def new_session(graph=None):
    config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    return tf.Session(config=config, graph=graph)

class Model(object):
    def __init__(self, name, network_type, trainable, max_action_no):
        self.network_type = network_type
        self.max_action_no = max_action_no
        self.build_network(name, network_type, trainable, max_action_no)
    
    def make_layer_variables(self, shape, trainable, name_suffix):
        stdv = 1.0 / math.sqrt(np.prod(shape[0:-1]))
        weights = tf.Variable(tf.random_uniform(shape, minval=-stdv, maxval=stdv), trainable=trainable, name='W_' + name_suffix)
        biases  = tf.Variable(tf.random_uniform([shape[-1]], minval=-stdv, maxval=stdv), trainable=trainable, name='b_' + name_suffix)
        return weights, biases
    
    def build_network(self, name, network_type, trainable, num_actions):
        if network_type == 'nips':
            self.build_network_nips(name, trainable, num_actions)
        else:
            self.build_network_nature(name, trainable, num_actions)
        
    def build_network_nature(self, name, trainable, num_actions):
        
        print("Building network for %s trainable=%s" % (name, trainable))
    
        with tf.variable_scope(name):
            # First layer takes a screen, and shrinks by 2x
            x = tf.placeholder(tf.uint8, shape=[None, 84, 84, 4], name="screens")
            print(x)
        
            x_normalized = tf.to_float(x) / 255.0
            print(x_normalized)
    
            # Second layer convolves 32 8x8 filters with stride 4 with relu
            W_conv1, b_conv1 = self.make_layer_variables([8, 8, 4, 32], trainable, "conv1")
    
            h_conv1 = tf.nn.relu(tf.nn.conv2d(x_normalized, W_conv1, strides=[1, 4, 4, 1], padding='VALID') + b_conv1, name="h_conv1")
            print(h_conv1)
    
            # Third layer convolves 64 4x4 filters with stride 2 with relu
            W_conv2, b_conv2 = self.make_layer_variables([4, 4, 32, 64], trainable, "conv2")
    
            h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 2, 2, 1], padding='VALID') + b_conv2, name="h_conv2")
            print(h_conv2)
    
            # Fourth layer convolves 64 3x3 filters with stride 1 with relu
            W_conv3, b_conv3 = self.make_layer_variables([3, 3, 64, 64], trainable, "conv3")
    
            h_conv3 = tf.nn.relu(tf.nn.conv2d(h_conv2, W_conv3, strides=[1, 1, 1, 1], padding='VALID') + b_conv3, name="h_conv3")
            print(h_conv3)
    
            h_conv3_flat = tf.reshape(h_conv3, [-1, 7 * 7 * 64], name="h_conv3_flat")
            print(h_conv3_flat)
    
            # Fifth layer is fully connected with 512 relu units
            W_fc1, b_fc1 = self.make_layer_variables([7 * 7 * 64, 512], trainable, "fc1")
    
            h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1, name="h_fc1")
            print(h_fc1)
    
            W_fc2, b_fc2 = self.make_layer_variables([512, num_actions], trainable, "fc2")
    
            y = tf.matmul(h_fc1, W_fc2) + b_fc2
            print(y)
        
        self.x = x
        self.y = y
        self.variables = [W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2]
        
    def build_network_nips(self, name, trainable, num_actions):
        
        print("Building network for %s trainable=%s" % (name, trainable))
    
        with tf.variable_scope(name):
            # First layer takes a screen, and shrinks by 2x
            x = tf.placeholder(tf.uint8, shape=[None, 84, 84, 4], name="screens")
            print(x)
        
            x_normalized = tf.to_float(x) / 255.0
            print(x_normalized)
    
            # Second layer convolves 16 8x8 filters with stride 4 with relu
            W_conv1, b_conv1 = self.make_layer_variables([8, 8, 4, 16], trainable, "conv1")
    
            h_conv1 = tf.nn.relu(tf.nn.conv2d(x_normalized, W_conv1, strides=[1, 4, 4, 1], padding='VALID') + b_conv1, name="h_conv1")
            print(h_conv1)
    
            # Third layer convolves 32 4x4 filters with stride 2 with relu
            W_conv2, b_conv2 = self.make_layer_variables([4, 4, 16, 32], trainable, "conv2")
    
            h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 2, 2, 1], padding='VALID') + b_conv2, name="h_conv2")
            print(h_conv2)
    
            h_conv2_flat = tf.reshape(h_conv2, [-1, 9 * 9 * 32], name="h_conv2_flat")
            print(h_conv2_flat)
    
            # Fourth layer is fully connected with 256 relu units
            W_fc1, b_fc1 = self.make_layer_variables([9 * 9 * 32, 256], trainable, "fc1")
    
            h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1, name="h_fc1")
            print(h_fc1)
    
            W_fc2, b_fc2 = self.make_layer_variables([256, num_actions], trainable, "fc2")
    
            y = tf.matmul(h_fc1, W_fc2) + b_fc2
            print(y)
        
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

class ModelA3C(Model):
    def build_network_nature(self, name, trainable, num_actions):
        
        print("Building network for %s trainable=%s" % (name, trainable))
    
        with tf.variable_scope(name):
            # First layer takes a screen, and shrinks by 2x
            x = tf.placeholder(tf.uint8, shape=[None, 84, 84, 4], name="screens")
            print(x)
        
            x_normalized = tf.to_float(x) / 255.0
            print(x_normalized)
    
            # Second layer convolves 32 8x8 filters with stride 4 with relu
            W_conv1, b_conv1 = self.make_layer_variables([8, 8, 4, 32], trainable, "conv1")
    
            h_conv1 = tf.nn.relu(tf.nn.conv2d(x_normalized, W_conv1, strides=[1, 4, 4, 1], padding='VALID') + b_conv1, name="h_conv1")
            print(h_conv1)
    
            # Third layer convolves 64 4x4 filters with stride 2 with relu
            W_conv2, b_conv2 = self.make_layer_variables([4, 4, 32, 64], trainable, "conv2")
    
            h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 2, 2, 1], padding='VALID') + b_conv2, name="h_conv2")
            print(h_conv2)
    
            # Fourth layer convolves 64 3x3 filters with stride 1 with relu
            W_conv3, b_conv3 = self.make_layer_variables([3, 3, 64, 64], trainable, "conv3")
    
            h_conv3 = tf.nn.relu(tf.nn.conv2d(h_conv2, W_conv3, strides=[1, 1, 1, 1], padding='VALID') + b_conv3, name="h_conv3")
            print(h_conv3)
    
            h_conv3_flat = tf.reshape(h_conv3, [-1, 7 * 7 * 64], name="h_conv3_flat")
            print(h_conv3_flat)
    
            # Fifth layer is fully connected with 512 relu units
            W_fc1, b_fc1 = self.make_layer_variables([7 * 7 * 64, 512], trainable, "fc1")
    
            h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1, name="h_fc1")
            print(h_fc1)
    
            W_fc2, b_fc2 = self.make_layer_variables([512, num_actions], trainable, "fc2")
    
            y = tf.matmul(h_fc1, W_fc2) + b_fc2
            print(y)
        
        self.x = x
        self.y = y
        self.variables = [W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2]
        
    def build_network_nips(self, name, trainable, num_actions):
        
        print("Building network for %s trainable=%s" % (name, trainable))
    
        with tf.variable_scope(name):
            # First layer takes a screen, and shrinks by 2x
            x_in = tf.placeholder(tf.float32, shape=[None, 84, 84, 4], name="screens")
            print(x_in)
        
            x_normalized = tf.to_float(x_in) / 255.0
            print(x_normalized)
    
            # Second layer convolves 16 8x8 filters with stride 4 with relu
            W_conv1, b_conv1 = self.make_layer_variables([8, 8, 4, 16], trainable, "conv1")
    
            h_conv1 = tf.nn.relu(tf.nn.conv2d(x_normalized, W_conv1, strides=[1, 4, 4, 1], padding='VALID') + b_conv1, name="h_conv1")
            print(h_conv1)
    
            # Third layer convolves 32 4x4 filters with stride 2 with relu
            W_conv2, b_conv2 = self.make_layer_variables([4, 4, 16, 32], trainable, "conv2")
    
            h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 2, 2, 1], padding='VALID') + b_conv2, name="h_conv2")
            print(h_conv2)
    
            h_conv2_flat = tf.reshape(h_conv2, [-1, 9 * 9 * 32], name="h_conv2_flat")
            print(h_conv2_flat)
    
            # Fourth layer is fully connected with 256 relu units
            W_fc1, b_fc1 = self.make_layer_variables([9 * 9 * 32, 256], trainable, "fc1")
    
            h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1, name="h_fc1")
            print(h_fc1)
    
            W_fc2, b_fc2 = self.make_layer_variables([256, num_actions], trainable, "fc2")
    
            y = tf.matmul(h_fc1, W_fc2) + b_fc2
            print(y)
            
            y_class = tf.nn.softmax(y)
            
            W_fc3, b_fc3 = self.make_layer_variables([256, 1], trainable, "fc3")
            v_ = tf.matmul(h_fc1, W_fc3) + b_fc3
            v = tf.reshape(v_, [-1] )
        
        self.x = x_in
        self.y = y
        self.y_class = y_class
        self.v = v
        self.variables = [W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2, W_fc3, b_fc3]
        
class ModelA3CLstm(Model):
    def build_network_nature(self, name, trainable, num_actions):
        
        print("Building network for %s trainable=%s" % (name, trainable))
    
        with tf.variable_scope(name):
            # First layer takes a screen, and shrinks by 2x
            x = tf.placeholder(tf.uint8, shape=[None, 84, 84, 4], name="screens")
            print(x)
        
            x_normalized = tf.to_float(x) / 255.0
            print(x_normalized)
    
            # Second layer convolves 32 8x8 filters with stride 4 with relu
            W_conv1, b_conv1 = self.make_layer_variables([8, 8, 4, 32], trainable, "conv1")
    
            h_conv1 = tf.nn.relu(tf.nn.conv2d(x_normalized, W_conv1, strides=[1, 4, 4, 1], padding='VALID') + b_conv1, name="h_conv1")
            print(h_conv1)
    
            # Third layer convolves 64 4x4 filters with stride 2 with relu
            W_conv2, b_conv2 = self.make_layer_variables([4, 4, 32, 64], trainable, "conv2")
    
            h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 2, 2, 1], padding='VALID') + b_conv2, name="h_conv2")
            print(h_conv2)
    
            # Fourth layer convolves 64 3x3 filters with stride 1 with relu
            W_conv3, b_conv3 = self.make_layer_variables([3, 3, 64, 64], trainable, "conv3")
    
            h_conv3 = tf.nn.relu(tf.nn.conv2d(h_conv2, W_conv3, strides=[1, 1, 1, 1], padding='VALID') + b_conv3, name="h_conv3")
            print(h_conv3)
    
            h_conv3_flat = tf.reshape(h_conv3, [-1, 7 * 7 * 64], name="h_conv3_flat")
            print(h_conv3_flat)
    
            # Fifth layer is fully connected with 512 relu units
            W_fc1, b_fc1 = self.make_layer_variables([7 * 7 * 64, 512], trainable, "fc1")
    
            h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1, name="h_fc1")
            print(h_fc1)
    
            W_fc2, b_fc2 = self.make_layer_variables([512, num_actions], trainable, "fc2")
    
            y = tf.matmul(h_fc1, W_fc2) + b_fc2
            print(y)
        
        self.x = x
        self.y = y
        self.variables = [W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2]
        
    def build_network_nips(self, name, trainable, num_actions):
        
        print("Building network for %s trainable=%s" % (name, trainable))
    
        with tf.variable_scope(name):
            # First layer takes a screen, and shrinks by 2x
            x_in = tf.placeholder(tf.uint8, shape=[None, 84, 84, 4], name="screens")
            print(x_in)
        
            x_normalized = tf.to_float(x_in) / 255.0
            print(x_normalized)
    
            # Second layer convolves 16 8x8 filters with stride 4 with relu
            W_conv1, b_conv1 = self.make_layer_variables([8, 8, 4, 16], trainable, "conv1")
    
            h_conv1 = tf.nn.relu(tf.nn.conv2d(x_normalized, W_conv1, strides=[1, 4, 4, 1], padding='VALID') + b_conv1, name="h_conv1")
            print(h_conv1)
    
            # Third layer convolves 32 4x4 filters with stride 2 with relu
            W_conv2, b_conv2 = self.make_layer_variables([4, 4, 16, 32], trainable, "conv2")
    
            h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 2, 2, 1], padding='VALID') + b_conv2, name="h_conv2")
            print(h_conv2)
    
            h_conv2_flat = tf.reshape(h_conv2, [-1, 9 * 9 * 32], name="h_conv2_flat")
            print(h_conv2_flat)
    
            # Fourth layer is fully connected with 256 relu units
            W_fc1, b_fc1 = self.make_layer_variables([9 * 9 * 32, 256], trainable, "fc1")
    
            h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1, name="h_fc1")
            print(h_fc1)
    
            with tf.variable_scope('LSTM'):
                hidden_size = 256
                batch_size = 5
                self.lstm_state = tf.placeholder(tf.float32, (1, hidden_size * 2))
                self.sequence_length = tf.placeholder(tf.int32)
                cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0)
                h_fc1_split = tf.split(0, batch_size, h_fc1)
                print h_fc1_split
                self.output_list, self.lstm_next_state = tf.nn.rnn(cell, h_fc1_split, initial_state=self.lstm_state, sequence_length=self.sequence_length)
                print('output_list : %s' % self.output_list)
                outputs = tf.pack(self.output_list)
                print('outputs : %s' % outputs)
                outputs2 = tf.squeeze(outputs, [1])
                print('outputs2 : %s' % outputs2)
                #outputs3 = tf.reshape(outputs2, tf.pack([self.sequence_length, hidden_size]))
                self.outputs3 = outputs2[:self.sequence_length[0], :]
                print('outputs3 : %s'  % self.outputs3)
                
                #self.lstm_initial_state = cell.zero_state(batch_size, tf.float32)
                #self.lstm_state = self.lstm_initial_state
                #(h_lstm, self.lstm_final_state) = cell(h_fc1, self.lstm_state)
                
                tvars = tf.trainable_variables()
                lstm_vars = [tvar for tvar in tvars if tvar.name.startswith(name + '/' + 'LSTM')]
        
            W_fc2, b_fc2 = self.make_layer_variables([256, num_actions], trainable, "fc2")
    
            #y = tf.matmul(h_fc1, W_fc2) + b_fc2
            y = tf.matmul(self.outputs3, W_fc2) + b_fc2
            print(y)
            
            y_class = tf.nn.softmax(y)
            
            W_fc3, b_fc3 = self.make_layer_variables([256, 1], trainable, "fc3")
            #v_ = tf.matmul(h_fc1, W_fc3) + b_fc3
            v_ = tf.matmul(self.outputs3, W_fc3) + b_fc3
            v = tf.reshape(v_, [-1] )
        
        self.x = x_in
        self.y = y
        self.y_class = y_class
        self.v = v
        #self.variables = [W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2, W_fc3, b_fc3]
        self.variables = [W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2, W_fc3, b_fc3] + lstm_vars
             
