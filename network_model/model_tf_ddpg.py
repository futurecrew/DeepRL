import numpy as np
import tensorflow as tf
from network_model.model_tf import ModelRunnerTF, new_session
from network_model.model_tf import Model

class ModelRunnerTFDdpg(ModelRunnerTF):
    def __init__(self, args,  action_group_no, thread_no):
        self.args = args
        learning_rate = args.learning_rate
        rms_decay = args.rms_decay
        rms_epsilon =  args.rms_epsilon
        self.network = args.network
        self.thread_no = thread_no
        
        self.train_batch_size = args.train_batch_size
        self.discount_factor = args.discount_factor
        self.action_group_no = action_group_no
        self.action_mat = np.zeros((self.train_batch_size, self.action_group_no))
        tf.logging.set_verbosity(tf.logging.WARN)
        
        self.sess = new_session()
        self.init_models(self.network, action_group_no, learning_rate, rms_decay, rms_epsilon)

    def init_models(self, network, action_group_no, learning_rate, rms_decay, rms_epsilon):        
        with tf.device(self.args.device):
            if self.args.env == 'torcs':
                if self.args.vision:
                    model_policy = ModelTorcsPixel(self.args, "policy", True, action_group_no, self.thread_no)
                    model_target = ModelTorcsPixel(self.args, "target", False, action_group_no, self.thread_no)
                else:
                    model_policy = ModelTorcsLowDim(self.args, "policy", True, action_group_no, self.thread_no)
                    model_target = ModelTorcsLowDim(self.args, "target", False, action_group_no, self.thread_no)
            else:
                raise ValueError('env %s is not supported.' % self.args.env)
            
            self.model_policy = model_policy
    
            self.x_in = model_policy.x_in
            self.action_in = model_policy.action_in
            self.actor_y = model_policy.actor_y
            self.critic_y = model_policy.critic_y
            self.vars = model_policy.variables
            self.actor_vars = model_policy.actor_vars
            
            self.x_in_target = model_target.x_in
            self.action_in_target = model_target.action_in
            self.actor_y_target = model_target.actor_y
            self.critic_y_target = model_target.critic_y
            self.vars_target = model_target.variables
            self.actor_vars_target = model_target.actor_vars

            # build the variable copy ops
            self.update_t = tf.placeholder(tf.float32, 1)
            self.update_target_list = []
            for i in range(0, len(self.vars)):
                self.update_target_list.append(self.vars_target[i].assign(self.update_t * self.vars[i] + (1-self.update_t) * self.vars_target[i]))
            self.update_target = tf.group(*self.update_target_list)
    
            self.critic_y_ = tf.placeholder(tf.float32, [None, 1])
            self.critic_grads_in = tf.placeholder(tf.float32, [None, action_group_no])
    
            optimizer_critic = tf.train.AdamOptimizer(0.001)
            self.critic_grads = tf.gradients(self.critic_y, self.action_in)
            self.difference = tf.abs(self.critic_y_ - self.critic_y)
            quadratic_part = tf.clip_by_value(self.difference, 0.0, 1.0)
            linear_part = self.difference - quadratic_part
            self.errors = (0.5 * tf.square(quadratic_part)) + linear_part
            if self.args.prioritized_replay == True:
                self.priority_weight = tf.placeholder(tf.float32, shape=self.difference.get_shape(), name="priority_weight")
                self.errors = tf.mul(self.errors, self.priority_weight)
            self.loss = tf.reduce_sum(self.errors)
            self.critic_step = optimizer_critic.minimize(self.loss)                 

            optimizer_actor = tf.train.AdamOptimizer(0.0001)
            gvs = optimizer_actor.compute_gradients(self.actor_y, var_list=self.actor_vars, grad_loss=-1 * self.critic_grads_in)
            self.actor_step = optimizer_actor.apply_gradients(gvs)
            
            self.saver = tf.train.Saver(max_to_keep=100)
            self.sess.run(tf.initialize_all_variables())
            self.sess.run(self.update_target, feed_dict={
                self.update_t: [1.0]
            })

    def predict(self, history_buffer):
        return self.sess.run(self.actor_y, {self.x_in: history_buffer})[0]
    
    def print_weights(self):
        #for var in self.actor_vars + self.critic_vars:
        for var in self.vars:
            print ''
            print '[ ' + var.name + ']'
            print self.sess.run(var)
        
    def train(self, minibatch, replay_memory, learning_rate, debug):
        global global_step_no

        if self.args.prioritized_replay == True:
            prestates, actions, rewards, poststates, terminals, replay_indexes, heap_indexes, weights = minibatch
        else:
            prestates, actions, rewards, poststates, terminals = minibatch
        
        actions_post = self.sess.run(self.actor_y_target, feed_dict={
                self.x_in_target: poststates
        })
        
        y2 = self.sess.run(self.critic_y_target, feed_dict={
            self.x_in_target: poststates, 
            self.action_in_target: actions_post
        })
        
        y_ = np.zeros((self.train_batch_size, 1))
        
        for i in range(self.train_batch_size):
            if self.args.clip_reward:
                reward = self.clip_reward(rewards[i])
            else:
                reward = rewards[i]
            if terminals[i]:
                y_[i] = reward
            else:
                y_[i] = reward + self.discount_factor * y2[i]

        self.sess.run([self.critic_step], feed_dict={
            self.x_in: prestates,
            self.action_in: actions,
            self.critic_y_: y_
        })
        
        actor_y_value = self.sess.run(self.actor_y, feed_dict={
            self.x_in: prestates,
        })

        critic_grads_value = self.sess.run(self.critic_grads, feed_dict= {
            self.x_in: prestates,
            self.action_in: actor_y_value
        })
        
        #if debug:
        #    print 'critic_grads_value : %s, %s' % (np.min(critic_grads_value), np.max(critic_grads_value))
        
        self.sess.run(self.actor_step, feed_dict={
            self.x_in: prestates,
            self.critic_grads_in: critic_grads_value[0]
        })

    def update_model(self):
        self.sess.run(self.update_target, feed_dict={
            self.update_t: [0.001]
        })


class ModelTorcsLowDim(Model):
    def build_network(self, name, network, trainable, action_group_no):
        self.print_log('Building network for ModelTorcsLowDim')
    
        with tf.variable_scope(name):
            input_img_size = self.screen_height * self.screen_width * self.history_len
            x_in = tf.placeholder(tf.float32, shape=[None, self.screen_height, self.screen_width, self.history_len], name="screens")
            x_flat = tf.reshape(x_in, (-1, input_img_size))
            self.print_log(x_flat)
        
            # Actor network
            with tf.variable_scope('actor'):
                W_actor_fc1, b_actor_fc1 = self.make_layer_variables([input_img_size, 600], trainable, "actor_fc1")    
                h_actor_fc1 = tf.nn.relu(tf.matmul(x_flat, W_actor_fc1) + b_actor_fc1, name="h_actor_fc1")
                self.print_log(h_actor_fc1)
        
                W_actor_fc2, b_actor_fc2 = self.make_layer_variables([600, 400], trainable, "actor_fc2")    
                h_actor_fc2 = tf.nn.relu(tf.matmul(h_actor_fc1, W_actor_fc2) + b_actor_fc2, name="h_actor_fc2")
                self.print_log(h_actor_fc2)
                
                weight_range = 3 * 10 ** -3
                W_steering, b_steering = self.make_layer_variables([400, 1], trainable, "steering", weight_range)    
                h_steering = tf.tanh(tf.matmul(h_actor_fc2, W_steering) + b_steering, name="h_steering")
                self.print_log(h_steering)
                
                W_acc, b_acc = self.make_layer_variables([400, 1], trainable, "acc", weight_range)
                h_acc = tf.sigmoid(tf.matmul(h_actor_fc2, W_acc) + b_acc, name="h_acc")
                self.print_log(h_acc)
                
                if action_group_no == 3:
                    W_brake, b_brake = self.make_layer_variables([400, 1], trainable, "brake", weight_range)    
                    h_brake = tf.sigmoid(tf.matmul(h_actor_fc2, W_brake) + b_brake, name="h_brake")
                    self.print_log(h_brake)
                    actor_y = tf.concat(1, [h_steering, h_acc, h_brake])
                else:
                    actor_y = tf.concat(1, [h_steering, h_acc])
                self.print_log(actor_y)

            # Critic network    
            action_in = tf.placeholder(tf.float32, shape=[None, action_group_no], name="actions")
            with tf.variable_scope('critic'):
                W_critic_fc1, b_critic_fc1 = self.make_layer_variables([input_img_size, 600], trainable, "critic_fc1")    
                h_critic_fc1 = tf.nn.relu(tf.matmul(x_flat, W_critic_fc1) + b_critic_fc1, name="h_critic_fc1")
                self.print_log(h_critic_fc1)
    
                h_concat = tf.concat(1, [h_critic_fc1, action_in])
            
                W_critic_fc2, b_critic_fc2 = self.make_layer_variables([600 + action_group_no, 400], trainable, "critic_fc2")    
                h_critic_fc2 = tf.nn.relu(tf.matmul(h_concat, W_critic_fc2) + b_critic_fc2, name="h_critic_fc2")
                self.print_log(h_critic_fc2)
    
                W_critic_fc3, b_critic_fc3 = self.make_layer_variables([400, 1], trainable, "critic_fc3", weight_range)
                critic_y = tf.matmul(h_critic_fc2, W_critic_fc3) + b_critic_fc3
                self.print_log(critic_y)
                        
        self.x_in = x_in
        self.action_in = action_in
        self.actor_y = actor_y
        self.critic_y = critic_y

        tvars = tf.trainable_variables()
        self.actor_vars = [tvar for tvar in tvars if tvar.name.startswith(name + '/actor')]
        self.variables = [tvar for tvar in tvars if tvar.name.startswith(name)]
        print 'len(self.actor_vars) : %s' % len(self.actor_vars)
        print 'len(self.variables) : %s' % len(self.variables)
                          

class ModelTorcsPixel(Model):
    def build_network(self, name, network, trainable, action_group_no):
        self.print_log('Building network ModelTorcsPixel')
    
        with tf.variable_scope(name):
            x_in = tf.placeholder(tf.uint8, shape=[None, self.screen_height, self.screen_width, self.history_len], name="screens")
            self.x_normalized = tf.to_float(x_in) / 255.0
            self.print_log(self.x_normalized)
    
            with tf.variable_scope('actor'):
                W_conv1, b_conv1 = self.make_layer_variables([6, 6, self.history_len, 32], trainable, "conv1")
                h_conv1 = tf.nn.relu(tf.nn.conv2d(self.x_normalized, W_conv1, strides=[1, 3, 3, 1], padding='VALID') + b_conv1, name="h_conv1")
                self.print_log(h_conv1)
        
                W_conv2, b_conv2 = self.make_layer_variables([3, 3, 32, 32], trainable, "conv2")
                h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 2, 2, 1], padding='VALID') + b_conv2, name="h_conv2")
                self.print_log(h_conv2)
        
                W_conv3, b_conv3 = self.make_layer_variables([3, 3, 32, 32], trainable, "conv3")
                h_conv3 = tf.nn.relu(tf.nn.conv2d(h_conv2, W_conv3, strides=[1, 2, 2, 1], padding='VALID') + b_conv3, name="h_conv3")
                self.print_log(h_conv3)
        
                conv_out_size = np.prod(h_conv3._shape[1:]).value
                h_conv3_flat = tf.reshape(h_conv3, [-1, conv_out_size], name="h_conv3_flat")
                self.print_log(h_conv3_flat)
        
                W_fc1, b_fc1 = self.make_layer_variables([conv_out_size, 600], trainable, "fc1")
                fc1_temp = tf.matmul(h_conv3_flat, W_fc1) + b_fc1
                h_fc1 = tf.nn.relu(fc1_temp, name="h_fc1")
                self.print_log(h_fc1)
        
                W_fc2, b_fc2 = self.make_layer_variables([600, 400], trainable, "fc2")
                h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2, name="h_fc2")
                self.print_log(h_fc2)
                
                # Actor specific network
                weight_range = 3 * 10 ** -4
                W_steering, b_steering = self.make_layer_variables([400, 1], trainable, "steering", weight_range)    
                h_steering = tf.tanh(tf.matmul(h_fc2, W_steering) + b_steering, name="h_steering")
                self.print_log(h_steering)
                
                W_acc, b_acc = self.make_layer_variables([400, 1], trainable, "acc", weight_range)    
                h_acc = tf.sigmoid(tf.matmul(h_fc2, W_acc) + b_acc, name="h_acc")
                self.print_log(h_acc)
                
                if action_group_no == 3:
                    W_brake, b_brake = self.make_layer_variables([400, 1], trainable, "brake", weight_range)    
                    h_brake = tf.sigmoid(tf.matmul(h_fc2, W_brake) + b_brake, name="h_brake")
                    self.print_log(h_brake)
                    actor_y = tf.concat(1, [h_steering, h_acc, h_brake])
                else:
                    actor_y = tf.concat(1, [h_steering, h_acc])
                self.print_log(actor_y)

            # Critic specific network    
            action_in = tf.placeholder(tf.float32, shape=[None, action_group_no], name="actions")
            with tf.variable_scope('critic'):
                W_critic_fc1, b_critic_fc1 = self.make_layer_variables([action_group_no, 600], trainable, "critic_fc1")
                critic_fc1_temp = tf.matmul(action_in, W_critic_fc1) + b_critic_fc1
                self.print_log(critic_fc1_temp)
                
                self.h_critic_fc1 = tf.nn.relu(fc1_temp + critic_fc1_temp , name="h_critic_fc1")
                self.print_log(self.h_critic_fc1)
        
                self.h_critic_fc2 = tf.nn.relu(tf.matmul(self.h_critic_fc1, W_fc2) + b_fc2, name="h_critic_fc2")
                self.print_log(self.h_critic_fc2)
        
                W_critic_fc3, b_critic_fc3 = self.make_layer_variables([400, 1], trainable, "critic_fc3", weight_range)
                critic_y = tf.matmul(self.h_critic_fc2, W_critic_fc3) + b_critic_fc3
                self.print_log(critic_y)
                            
        self.x_in = x_in
        self.action_in = action_in
        self.actor_y = actor_y
        self.critic_y = critic_y
                          
        tvars = tf.trainable_variables()
        self.actor_vars = [tvar for tvar in tvars if tvar.name.startswith(name + '/actor')]
        self.variables = [tvar for tvar in tvars if tvar.name.startswith(name)]
        print 'len(self.actor_vars) : %s' % len(self.actor_vars)
        print 'len(self.variables) : %s' % len(self.variables)
                          
