import os
import time
import math
import random
import numpy as np
import cv2
import env.torcs.snakeoil3_gym as snakeoil3

class TorcsEnv():
    def __init__(self, vision, bin, port, track, display_screen):
        self.actions = None
        self.display_screen = display_screen
        self.reset_count = 0
        self.client = None
        self.vision = vision
        self.bin = bin
        self.port = port

        if track == -1:
            self.track_file = 'autostart.sh'
        else:
            self.track_file = 'autostart%s.sh' % track
        self.track_file = 'env/torcs/' + self.track_file
        
        self.damage = 0
        self.throttle = True
        self.gear_change = True
        self.initial_run = True
        
        #self.action_group_no = 3         # steering, accel, brake
        self.action_group_no = 2         # steering, accel
        
    def initialize(self):
        self.reset_game(True)
        
    def get_actions(self):
        return [i for i in range(self.action_group_no)]
        
    @property
    def state_dtype(self):
        if self.vision:
            return np.uint8
        else:
            return np.float32
        
    @property
    def continuous_action(self):
        return True
        
    def reset_game(self, relaunch=False):
        self.time_step = 0
        self.damage = 0
        self.conn_error = False
        self.reset_count += 1
        self.total_reward = 0

        if self.client is not None:
            self.client.R.d['meta'] = True
            self.client.respond_to_server()

        if relaunch or self.reset_count % 200 == 0:
            os.system('pkill %s' % self.bin)
            time.sleep(0.5)
            if self.vision is True:
                command = '%s -p %s -nofuel -nolaptime -vision &' % (self.bin, self.port)
            else:
                command = '%s -p %s -nofuel -nolaptime &' % (self.bin, self.port)
            os.system(command)
            time.sleep(0.5)
            os.system('sh %s' % self.track_file)
            time.sleep(0.5)

        # Modify here if you use multiple tracks in the environment
        self.client = snakeoil3.Client(self.bin, p=self.port, vision=self.vision, track_file=self.track_file)  # Open new UDP in vtorcs
        self.client.MAX_STEPS = np.inf

        client = self.client
        client.get_servers_input()  # Get the initial input from torcs

        self.obs = client.S.d  # Get the current full-observation from torcs

        self.last_u = None

    def lives(self):
        return 1
    
    def getScreenRGB(self):
        obs = self.client.S.d
        img_size = len(obs['img'])
        imgR = obs['img'][0:img_size:3]
        imgG = obs['img'][1:img_size:3]
        imgB = obs['img'][2:img_size:3]
        return np.array(imgR, dtype=np.uint8), np.array(imgG, dtype=np.uint8), np.array(imgB, dtype=np.uint8)
    
    def getScreenGrayscale(self, debug_display=False, debug_display_sleep=0):
        if self.vision:
            obs = self.client.S.d
            img_size = len(obs['img'])
            red = np.array(obs['img'][0:img_size:3], dtype=np.uint8).reshape(64, 64)
            green = np.array(obs['img'][1:img_size:3], dtype=np.uint8).reshape(64, 64)
            blue = np.array(obs['img'][2:img_size:3], dtype=np.uint8).reshape(64, 64)
            state = np.array(0.2989*red + 0.5870*green + 0.1140*blue, dtype=np.uint8)
            
            if debug_display:
                cv2.imshow('image', state)
                cv2.waitKey(debug_display_sleep)
        else:
            focus=np.array(self.obs['focus'], dtype=np.float32)/200.
            speedX=np.array(self.obs['speedX'], dtype=np.float32)/300.0
            speedY=np.array(self.obs['speedY'], dtype=np.float32)/300.0
            speedZ=np.array(self.obs['speedZ'], dtype=np.float32)/300.0
            angle=np.array(self.obs['angle'], dtype=np.float32)/3.1416
            damage=np.array(self.obs['damage'], dtype=np.float32)
            opponents=np.array(self.obs['opponents'], dtype=np.float32)/200.
            rpm=np.array(self.obs['rpm'], dtype=np.float32)/10000
            track=np.array(self.obs['track'], dtype=np.float32)/200.
            trackPos=np.array(self.obs['trackPos'], dtype=np.float32)/1.
            wheelSpinVel=np.array(self.obs['wheelSpinVel'], dtype=np.float32)/100.
            
            state = np.hstack((angle, track, trackPos, speedX, speedY, speedZ, wheelSpinVel, rpm))
       
        return state
    
    def act(self, actions):
        self.client.R.d['steer'] = actions[0]
        self.client.R.d['accel'] = actions[1]
        if self.action_group_no == 3:
            self.client.R.d['brake'] = actions[2]
        else:
            self.client.R.d['brake'] = 0
        #self.client.R.d['gear'] = 1
        
        self.client.respond_to_server()
        ret = self.client.get_servers_input()
        
        if ret == False:
            self.conn_error = True
        
        self.obs = self.client.S.d        
        reward = self.get_reward(self.obs)
        
        self.total_reward += reward
        
        return reward
    
    def get_reward(self, obs):
        if abs(obs['angle']) > 3.14 / 2:      # car direction is reverse
            return -100
        #reward = obs['speedX'] * abs(math.cos(obs['angle'])) -  obs['speedX'] * abs(math.sin(obs['angle'])) - obs['speedX'] * abs(obs['trackPos'])
        reward = obs['speedX'] * abs(math.cos(obs['angle'])) -  obs['speedX'] * abs(math.sin(obs['angle'])) - 50 * abs(obs['trackPos'])
        if obs['damage'] > self.damage:
            self.damage = obs['damage']
            if reward > 0:
                return -1
        return reward
    
    def game_over(self):
        obs = self.client.S.d
        if abs(obs['angle']) > 3.14 / 2:      # car direction is reverse
            print 'game_over 1'
            return True
        elif obs['curLapTime'] > 50 and obs['speedX'] * abs(math.cos(obs['angle'])) < 0.1:        # Car is stuck 
            print 'game_over 2'
            return True
        elif self.total_reward < -1000:
            print 'game_over 3'
            return True
        elif self.conn_error == True:
            return True
        else:
            return False
    
    def apply_action_noise(self, action_values, greedy_epsilon):
        noise_0 = self.OU(0.15, action_values[0], 0, 0.4)
        noise_1 = self.OU(0.15, action_values[1], 0.6, 0.4)
                     
        action_values[0] = np.clip(action_values[0] + greedy_epsilon * noise_0, -1, 1)
        action_values[1] = np.clip(action_values[1] + greedy_epsilon * noise_1, 0, 1)

        if self.action_group_no == 3:
            noise_2 = self.OU(0.15, action_values[2], 0.05, 0.1)
            action_values[2] = np.clip(action_values[2] + greedy_epsilon * noise_2, 0, 1)
            """
            if random.random() > 0.1:   # brake
                action_values[2] = 0
            else:
                action_values[2] = np.clip(action_values[2] + greedy_epsilon * noise_2, 0, 1)
            """

    def OU(self, theta, x, mean, gamma):
        return theta * (mean - x) + gamma * np.random.normal(0, 1)
        
    def finish(self):
        os.system('pkill %s' % self.bin)
    
def initialize_args(args):
    if args.vision:
        args.screen_width = 64    # input screen width
        args.screen_height = 64    # input screen height
    else:
        args.screen_width = 29    # input screen width
        args.screen_height = 1    # input screen height
    
    args.screen_history = 1    # input screen history
    args.frame_repeat = 1    # how many frames to repeat in ale for one predicted action
    args.use_env_frame_skip = False    # whether to use ale frame_skip feature
    args.discount_factor = 0.99    # RL discount factor
    args.test_step = 125000    # test for this number of steps
    args.use_random_action_on_reset = False
    args.crop_image = False         # Crop input image or zoom image
    args.run_test = True    # Whether to run test
    args.clip_reward = False
    args.clip_reward_high = 1
    args.clip_reward_low = -1
    args.clip_loss = True

    args.backend = 'TF'    # Deep learning library backend (TF, NEON)
    if args.backend == 'TF':
        args.screen_order = 'hws'   # dimension order in replay memory (height, width, screen)
    elif args.backend == 'NEON':
        args.screen_order = 'shw'   # (screen, height, width)
        args.dnn_initializer = 'fan_in'    # 
        args.use_gpu_replay_mem = False       # Whether to store replay memory in gpu or not to speed up leraning        
    
    if args.drl in ['a3c_lstm', 'a3c']:
        args.asynchronousRL = True
        args.use_annealing = True
        args.train_batch_size = 5
        args.max_replay_memory = 30
        args.max_epoch = 20
        args.epoch_step = 500000    # 4,000,000 global steps / 8 threads
        args.train_start = 0        
        args.train_step = 5                 
        args.train_epsilon_start_step = 0    # start decreasing greedy epsilon from this train step 
        args.learning_rate = 0.0007                 # RL learning rate
        args.rms_decay = 0.99                 # rms decay
        args.rms_epsilon = 0.1                 # rms epsilon
        args.choose_max_action = False
        args.lost_life_game_over = False    # whether to regard lost life as game over
        args.lost_life_terminal = True    # whether to regard lost life as terminal state        
        args.minibatch_random = False       # whether to use random indexing or sequential indexing for minibatch
        args.save_step = 4000000            # save result every this training step
        args.prioritized_replay = False
        args.max_global_step_no = args.epoch_step * args.max_epoch * args.thread_no
    elif args.drl in ['1q']:
        args.asynchronousRL = True

        args.use_annealing = True
        
        args.max_replay_memory = 10000
        args.max_epoch = 20
        args.train_start = 100           # start training after filling this replay memory size
        args.epoch_step = 500000
        args.train_step = 5
        args.train_batch_size = 5
        args.train_epsilon_start_step = 0    # start decreasing greedy epsilon from this train step 
        args.train_epsilon_end_step = 100000    # end decreasing greedy epsilon from this train step 
        
        #args.learning_rate = 0.00025                 # RL learning rate
        args.learning_rate = 0.0007
        
        args.rms_decay = 0.95                
        args.rms_epsilon = 0.01                 
        args.choose_max_action = True

        # DJDJ        
        #args.lost_life_game_over = True    # whether to regard lost life as game over
        args.lost_life_game_over = False    # whether to regard lost life as game over
        
        args.lost_life_terminal = True    # whether to regard lost life as terminal state
        args.minibatch_random = False       # whether to use random indexing or sequential indexing for minibatch
        args.train_min_epsilon = 0.1    # minimum greedy epsilon value for exloration
        args.update_step = 10000    # copy train network into target network every this train step
        args.save_step = 50000            # save result every this training step
        args.prioritized_replay = False
        args.max_global_step_no = args.epoch_step * args.max_epoch * args.thread_no
        args.double_dqn = False                   # whether to use double dqn
        args.test_epsilon = 0.05                    # greedy epsilon for test
        args.prioritized_replay = False
    else:
        args.asynchronousRL = False
        args.use_annealing = False
        if args.vision:
            args.train_batch_size = 16
        else:
            args.train_batch_size = 64
        args.max_replay_memory = 1000000
        args.max_epoch = 200
        args.epoch_step = 250000
        args.train_start = 10        # start training after filling this replay memory size

        # DJDJ        
        #args.train_step = 4                 # Train every this screen step
        args.train_step = 1                 # Train every this screen step
        
        args.learning_rate = 0.00025                 
        args.rms_decay = 0.95                
        args.rms_epsilon = 0.01                 
        args.choose_max_action = True
        args.lost_life_game_over = True    # whether to regard lost life as game over
        args.lost_life_terminal = True    # whether to regard lost life as terminal state
        args.minibatch_random = True
        args.train_min_epsilon = 0.1    # minimum greedy epsilon value for exloration
        args.train_epsilon_start_step = 0    # start decreasing greedy epsilon from this train step 
        args.train_epsilon_end_step = 30000    # end decreasing greedy epsilon from this train step 
        args.update_step = 1    # copy train network into target network every this train step
        args.optimizer = 'RMSProp'    # 
        args.save_step = 10000            # save result every this training step
    
        if args.drl == 'dqn':                     # DQN hyper params
            args.double_dqn = False                   # whether to use double dqn
            args.test_epsilon = 0.05                    # greedy epsilon for test
            args.prioritized_replay = False
        elif args.drl == 'double_dqn':    # Double DQN hyper params
            args.double_dqn = True
            args.train_min_epsilon = 0.01    # 
            args.test_epsilon = 0.001    # 
            args.update_step = 30000    #
            args.prioritized_replay = False 
        elif args.drl == 'prioritized_rank':    # Prioritized experience replay params for RANK
            args.prioritized_replay = True    # 
            
            # DJDJ
            args.learning_rate = 0.00025 / 4    #
            #args.learning_rate = 0.00025 / 8
             
            args.test_epsilon = 0.001    # 
            args.prioritized_mode = 'RANK'    # 
            args.sampling_alpha = 0.7    # 
            args.sampling_beta = 0.5    # 
            args.heap_sort_term = 250000    # 
            args.double_dqn = True
        elif args.drl == 'prioritized_proportion':    # Prioritized experience replay params for PROPORTION
            args.prioritized_replay = True    # 
            args.learning_rate = 0.00025 / 4    # 
            args.test_epsilon = 0.001    # 
            args.prioritized_mode = 'PROPORTION'    # 
            args.sampling_alpha = 0.6    # 
            args.sampling_beta = 0.4    # 
            args.heap_sort_term = 250000    # 
            args.double_dqn = True

    return args
    