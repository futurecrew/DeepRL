import random
import numpy as np
from ale_python_interface import ALEInterface

class AleEnv():
    def __init__(self, rom, display_screen, use_env_frame_skip, frame_repeat):
        self.actions = None
        self.rom = rom
        self.display_screen = display_screen
        self.use_env_frame_skip = use_env_frame_skip
        self.frame_repeat = frame_repeat
        
    def initialize(self):
        self.ale = ALEInterface()
        self.ale.setInt("random_seed", random.randint(1, 1000))
        if self.display_screen:
            self.ale.setBool('display_screen', True)

        if self.use_env_frame_skip == True:
            self.ale.setInt('frame_skip', self.frame_repeat)
            self.ale.setBool('color_averaging', True)        
 
        self.ale.setFloat('repeat_action_probability', 0)
        self.ale.loadROM(self.rom)
        self.actions = self.ale.getMinimalActionSet()
        print 'actions: %s' % self.actions
        (self.screen_width,self.screen_height) = self.ale.getScreenDims()
        print("width/height: " +str(self.screen_width) + "/" + str(self.screen_height))
        
        self.initialized = True
        
    def get_actions(self, rom=None):
        if self.actions is None and rom != None:
            ale = ALEInterface()
            ale.loadROM(rom)
            self.actions = ale.getMinimalActionSet()
        return self.actions
        
    @property
    def state_dtype(self):
        return np.uint8
        
    @property
    def continuous_action(self):
        return False
    
    def reset_game(self):
        self.ale.reset_game()
        
    def lives(self):
        return self.ale.lives()
    
    def getScreenRGB(self):
        return self.ale.getScreenRGB()
    
    def getState(self, debug_display=False, debug_input=None):
        screen = self.ale.getScreenGrayscale()
        if screen is not None and debug_display:
            debug_input.show(screen.reshape(screen.shape[0], screen.shape[1]))
        return screen.reshape(self.screen_height, self.screen_width)
    
    def act(self, action):
        return self.ale.act(action)
    
    def game_over(self):
        return self.ale.game_over()
    
    def finish(self):
        return    
    
def initialize_args(args):
    args.screen_width = 84    # input screen width
    args.screen_height = 84    # input screen height
    args.screen_history = 4    # input screen history
    args.frame_repeat = 4    # how many frames to repeat in ale for one predicted action
    args.use_env_frame_skip = False    # whether to use ale frame_skip feature
    args.discount_factor = 0.99    # RL discount factor
    args.test_step = 125000    # test for this number of steps
    args.use_random_action_on_reset = True
    args.crop_image = False         # Crop input image or zoom image
    args.run_test = True    # Whether to run test
    args.clip_reward = True
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
    
    if args.network == '':
        args.network = 'nature'
    if args.network != 'nature' and args.network != 'nips':
        raise ValueError('network should be either nips or nature')
    
    if args.drl in ['a3c_lstm', 'a3c']:
        args.asynchronousRL = True
        args.train_batch_size = 5
        if args.max_replay_memory == -1:
            args.max_replay_memory = 30
        args.max_epoch = 20
        args.epoch_step = 500000    # 4,000,000 global steps / 8 threads
        args.train_start = 0        
        args.train_step = 5                 
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
        if args.max_replay_memory == -1:
            args.max_replay_memory = 10000
        args.max_epoch = 20
        args.train_start = 100           # start training after filling this replay memory size
        args.epoch_step = 500000
        args.train_step = 5
        args.train_batch_size = 5
        args.train_epsilon_start_step = 0    # start decreasing greedy epsilon from this train step 
        args.train_epsilon_end_step = 1000000    # end decreasing greedy epsilon from this train step 
        args.learning_rate = 0.0007
        args.rms_decay = 0.95                
        args.rms_epsilon = 0.01                 
        args.choose_max_action = True
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
        args.train_batch_size = 32
        if args.max_replay_memory == -1:
            args.max_replay_memory = 1000000
        args.max_epoch = 200
        args.epoch_step = 250000
        args.train_start = 50000        # start training after filling this replay memory size
        args.train_step = 4                 # Train every this screen step
        args.learning_rate = 0.00025                 
        args.rms_decay = 0.95                
        args.rms_epsilon = 0.01                 
        args.choose_max_action = True
        args.lost_life_game_over = True    # whether to regard lost life as game over
        args.lost_life_terminal = True    # whether to regard lost life as terminal state
        args.minibatch_random = True
        args.train_min_epsilon = 0.1    # minimum greedy epsilon value for exloration
        args.train_epsilon_start_step = 0    # start decreasing greedy epsilon from this train step 
        args.train_epsilon_end_step = 1000000    # end decreasing greedy epsilon from this train step 
        args.update_step = 10000    # copy train network into target network every this train step
        args.optimizer = 'RMSProp'    # 
        args.save_step = 50000            # save result every this training step
        
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
            args.learning_rate = 0.00025 / 4    #
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
    
