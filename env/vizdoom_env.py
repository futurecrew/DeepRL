import numpy as np
import cv2
import time
import copy
import itertools as it
from vizdoom import *

class VizDoomEnv():
    def __init__(self, config, display_screen, use_env_frame_skip, frame_repeat):
        if config == None:
            print 'Need to set vizdoom --config'
            exit()
        self.actions = None
        self.config = config
        self.display_screen = display_screen
        if use_env_frame_skip == True:
            self.frame_repeat = frame_repeat
        else:
            self.frame_repeat = 1

    def initialize(self):
        self.game = DoomGame()
        self.game.load_config(self.config)
        self.game.set_window_visible(self.display_screen)        
        self.game.set_mode(Mode.PLAYER)
        self.game.set_screen_format(ScreenFormat.GRAY8)
        self.game.set_screen_resolution(ScreenResolution.RES_640X480)
        #self.game.set_screen_resolution(ScreenResolution.RES_160X120)
        self.game.init()
        
        self.actions = self.get_action_list(self.game)
        print 'actions no: %s' % len(self.actions)
        print 'actions : %s' % self.actions
        
    def get_actions(self, rom=None):
        if self.actions is None:
            game = DoomGame()
            game.load_config(self.config)
            game.set_window_visible(False)        
            game.set_screen_resolution(ScreenResolution.RES_160X120)
            game.init()
            self.actions = self.get_action_list(game)
            game.close()
        return self.actions
        
    def get_action_list(self, game):
        available_buttons_size = game.get_available_buttons_size()

        # get each action
        actions = []
        null_action = [0] * available_buttons_size
        actions.append(null_action)
        for i in range(available_buttons_size):
            action = copy.deepcopy(null_action)
            action[i] = 1
            actions.append(action)
        
        # get 2 combinations of actions
        for action_index in it.combinations(range(available_buttons_size), 2):
            action = copy.copy(null_action)
            for index in action_index:
                action[index] = 1 
            actions.append(action)
    
        return actions
        
    def reset_game(self):
        self.game.new_episode()
        
    def lives(self):
        return 1 if self.game.is_player_dead() == False else 0
    
    def getScreenRGB(self):
        return self.game.get_state().screen_buffer
    
    def getScreenGrayscale(self, debug_display=False, debug_display_sleep=0):
        screen = self.game.get_state().screen_buffer
        if screen is not None and debug_display:
            cv2.imshow('image', screen)
            cv2.waitKey(debug_display_sleep)
        return screen
    
    def act(self, action):
        return self.game.make_action(action, self.frame_repeat)
    
    def game_over(self):
        return self.game.is_episode_finished()
    
    def finish(self):
        self.game.close()
        
        
def initialize_args(args):
    args.screen_width = 160    # input screen width
    args.screen_height = 120    # input screen height
    args.screen_history = 1    # input screen history
    args.frame_repeat = 12    # how many frames to repeat in ale for one predicted action
    args.use_env_frame_skip = True    # whether to use ale frame_skip feature
    args.discount_factor = 0.99    # RL discount factor
    args.test_step = 4000    # test for this number of steps
    args.use_random_action_on_reset = False
    args.crop_image = False         # Crop input image or zoom image
    args.run_test = True    # Whether to run test
    args.lost_life_game_over = False    # whether to regard lost life as game over
    args.lost_life_terminal = True    # whether to regard lost life as terminal state

    args.clip_reward = False
    args.clip_loss = False
    """
    args.clip_reward = True
    args.clip_reward_high = 1
    args.clip_reward_low = -1
    args.clip_loss = True
    """

    args.rms_decay = 0.9                
    args.rms_epsilon =1e-10          
    #args.rms_decay = 0.99                 # rms decay
    #args.rms_epsilon = 0.1                 # rms epsilon
          
    args.train_start = 10           # start training after filling this replay memory size

    args.backend = 'TF'    # Deep learning library backend (TF, NEON)    
    if args.backend == 'TF':
        args.screen_order = 'hws'   # dimension order in replay memory (height, width, screen)
    elif args.backend == 'NEON':
        args.screen_order = 'shw'   # (screen, height, width)
        args.dnn_initializer = 'fan_in'    # 
        args.use_gpu_replay_mem = False       # Whether to store replay memory in gpu or not to speed up leraning        

    if args.drl in ['a3c_lstm', 'a3c']:
        args.asynchronousRL = True
        args.max_replay_memory = 10000
        args.max_epoch = 20
        args.epoch_step = 4000
        args.train_step = 5
        args.train_batch_size = 5
        #args.train_epsilon_start_step = 0.2 * args.max_epoch * args.epoch_step    # start decreasing greedy epsilon from this train step 
        args.train_epsilon_start_step = 0        
        args.learning_rate = 0.0007                 # RL learning rate
        args.choose_max_action = False
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
        args.epoch_step = 4000
        args.train_step = 3
        args.train_batch_size = 3
        args.train_epsilon_start_step = args.max_epoch * args.epoch_step * 0.1    # start decreasing greedy epsilon from this train step 
        args.train_epsilon_end_step = args.max_epoch * args.epoch_step * 0.6    # end decreasing greedy epsilon from this train step 
        args.learning_rate = 0.00025                 # RL learning rate
        args.choose_max_action = True
        args.minibatch_random = False       # whether to use random indexing or sequential indexing for minibatch
        args.train_min_epsilon = 0.1    # minimum greedy epsilon value for exloration
        args.update_step = 100    # copy train network into target network every this train step
        args.save_step = 50000            # save result every this training step
        args.prioritized_replay = False
        args.max_global_step_no = args.epoch_step * args.max_epoch * args.thread_no
        args.double_dqn = False                   # whether to use double dqn
        args.test_epsilon = 0.0                    # greedy epsilon for test
        args.prioritized_replay = False
    else:
        args.asynchronousRL = False
        args.train_batch_size = 64
        args.max_replay_memory = 10000
        args.max_epoch = 20
        args.epoch_step = 2000
        args.train_start = 10        # start training after filling this replay memory size
        args.train_step = 1                 # Train every this screen step
        args.learning_rate = 0.00025                 
        args.choose_max_action = True
        args.minibatch_random = True
        args.train_min_epsilon = 0.1    # minimum greedy epsilon value for exloration
        args.update_step = 100    # copy train network into target network every this train step
        args.optimizer = 'RMSProp'    # 
        args.save_step = 50000            # save result every this training step
        args.train_epsilon_start_step = args.max_epoch * args.epoch_step * 0.1    # start decreasing greedy epsilon from this train step 
        args.train_epsilon_end_step = args.max_epoch * args.epoch_step * 0.6    # end decreasing greedy epsilon from this train step 
        args.test_epsilon = 0.0                    # greedy epsilon for test

        if args.drl == 'dqn':                     # DQN hyper params
            args.double_dqn = False                   # whether to use double dqn
            args.prioritized_replay = False
        elif args.drl == 'double_dqn':    # Double DQN hyper params
            args.double_dqn = True
            args.train_min_epsilon = 0.01    # 
            args.update_step = 300    #
            args.prioritized_replay = False 
        elif args.drl == 'prioritized_rank':    # Prioritized experience replay params for RANK
            args.prioritized_replay = True    # 
            args.learning_rate = 0.00025 / 4    # 
            args.prioritized_mode = 'RANK'    # 
            args.sampling_alpha = 0.7    # 
            args.sampling_beta = 0.5    # 
            args.heap_sort_term = 250000    # 
            args.double_dqn = True
            args.use_priority_weight = True    # whether to priority weight
        elif args.drl == 'prioritized_proportion':    # Prioritized experience replay params for PROPORTION
            args.prioritized_replay = True    # 
            args.learning_rate = 0.00025 / 4    # 
            args.prioritized_mode = 'PROPORTION'    # 
            args.sampling_alpha = 0.6    # 
            args.sampling_beta = 0.4    # 
            args.heap_sort_term = 250000    # 
            args.double_dqn = True
            args.use_priority_weight = True    # whether to priority weight
            
    return args
