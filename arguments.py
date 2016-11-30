import argparse

def get_game_name(rom):
    if '/' in rom:
        game = rom.split('/')[-1]
    else:
        game = rom
    if '.' in game:
        game = game.split('.')[0]
    return game

def get_args():
    parser = argparse.ArgumentParser()    
    
    parser.add_argument('rom', type=str, help='ALE rom file')    
    parser.add_argument('--multi-thread-no', type=int, default=1, help='Number of multiple threads for Asynchronous RL')
    parser.add_argument('--network', type=str, default='nips', choices=['nips', 'nature'], help='network model nature or nips') 
    parser.add_argument('--drl', type=str, default='dqn', choices=['dqn', 'double_dqn', 'prioritized_rank', 'prioritized_proportion', 'a3c_lstm', 'a3c', '1q'])
    parser.add_argument('--snapshot', type=str, default=None, help='trained file to resume training or to replay') 
    parser.add_argument('--device', type=str, default='', help='gpu or cpu')
    parser.add_argument('--show-screen', action='store_true', help='whether to show display or not')
    parser.add_argument('--config', type=str, default=None, help='config file for vizdoom')
    parser.set_defaults(show_screen=False)
    
    args = parser.parse_args()
    
    if args.rom == 'vizdoom':
        args.env = 'vizdoom'
    else:
        args.env = 'ale'
    
    args.game = get_game_name(args.rom)
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

    args.backend = 'TF'    # Deep learning library backend (TF, NEON)    
    if args.backend == 'TF':
        args.screen_order = 'hws'   # dimension order in replay memory (height, width, screen)
    elif args.backend == 'NEON':
        args.screen_order = 'shw'   # (screen, height, width)
        args.dnn_initializer = 'fan_in'    # 
        args.use_gpu_replay_mem = False       # Whether to store replay memory in gpu or not to speed up leraning        
    
    if args.drl in ['a3c_lstm', 'a3c', '1q']:
        args.asynchronousRL = True
        args.train_batch_size = 5
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
        args.max_global_step_no = args.epoch_step * args.max_epoch * args.multi_thread_no

        # DJDJ
        if args.env == 'vizdoom':
            args.screen_width = 45    # input screen width
            args.screen_height = 30    # input screen height
            #args.train_batch_size = 64
            #args.learning_rate = 0.00025                 # RL learning rate
            args.use_env_frame_skip = True
            args.frame_repeat = 12
            #args.train_step = 1                 # Train every this screen step
            args.max_epoch = 20
            args.epoch_step = 2000
            args.max_replay_memory = 10000
            args.train_start = 10        # start training after filling this replay memory size
            args.test_step = 2000
            args.train_epsilon_start_step = 6000    # start decreasing greedy epsilon from this train step 
            args.train_epsilon_end_step = 24000    # end decreasing greedy epsilon from this train step 
            args.screen_history = 1    # input screen history
            args.test_epsilon = 0.0                    # greedy epsilon for test
            args.rms_decay = 0.9                
            args.rms_epsilon = 1e-10                   
            args.use_random_action_on_reset = False
            args.update_step = 100    # copy train network into target network every this train step
        
    else:
        args.asynchronousRL = False
        args.train_batch_size = 32
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
        args.train_epsilon_end_step = 100000    # end decreasing greedy epsilon from this train step 
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

        # DJDJ
        if args.env == 'vizdoom':
            args.screen_width = 45    # input screen width
            args.screen_height = 30    # input screen height
            args.train_batch_size = 64
            args.learning_rate = 0.00025                 # RL learning rate
            args.use_env_frame_skip = True
            args.frame_repeat = 12
            args.train_step = 1                 # Train every this screen step
            args.max_epoch = 20
            args.epoch_step = 2000
            args.max_replay_memory = 10000
            args.train_start = 0        # start training after filling this replay memory size
            args.test_step = 2000
            args.train_epsilon_start_step = 6000    # start decreasing greedy epsilon from this train step 
            args.train_epsilon_end_step = 24000    # end decreasing greedy epsilon from this train step 
            args.screen_history = 1    # input screen history
            args.test_epsilon = 0.0                    # greedy epsilon for test
            args.rms_decay = 0.9                
            args.rms_epsilon = 1e-10                   
            args.use_random_action_on_reset = False
            args.update_step = 100    # copy train network into target network every this train step
            
    return args
