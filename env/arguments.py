import sys
import argparse
import importlib

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
    
    parser.add_argument('env', type=str, help='ALE rom file or environment name (e.g. torcs)')    
    parser.add_argument('--thread-no', type=int, default=1, help='Number of multiple threads for Asynchronous RL')
    parser.add_argument('--network', type=str, default='', help='network model nature or nips') 
    parser.add_argument('--drl', type=str, default='dqn', choices=['dqn', 'double_dqn', 'prioritized_rank', 'a3c_lstm', 'a3c', '1q'])
    parser.add_argument('--ddpg', action='store_true', help='whether to use DDPG or not')
    parser.add_argument('--snapshot', type=str, default=None, help='trained file to resume training or to replay') 
    parser.add_argument('--device', type=str, default='', help='gpu or cpu')
    parser.add_argument('--show-screen', action='store_true', help='whether to show display or not')
    parser.add_argument('--max-replay-memory', type=int, default=-1, help='maximum size of replay memory')

    if len(sys.argv) < 2:
        raise ValueError('Need arguments to run.')
    
    env = sys.argv[1]
    if env.endswith('.bin'):
        module = importlib.import_module('env.ale.ale_env')
    else:
        if env == 'torcs':
            parser.add_argument('--vision', action='store_true', help='use vision input or not')
            parser.add_argument('--bin', type=str, default='torcs', help='torcs executable') 
            parser.add_argument('--port', type=int, default=3001, help='port to be used for torcs server')
            parser.add_argument('--track', type=int, default=-1, help='track file no')
        elif env == 'vizdoom':
            parser.add_argument('--config', type=str, default=None, help='config file for vizdoom')
    
        module = importlib.import_module('env.%s.%s_env' % (env, env))

    parser.set_defaults(show_screen=False)
    parser.set_defaults(ddpg=False)
    
    args = parser.parse_args()
    args.game = get_game_name(env)
    args.snapshot_folder = 'snapshot/' + args.game
    args.use_color_input = False

    init_func = getattr(module, 'initialize_args')
    init_func(args)
    
    return args

def get_env(args, initialize, show_screen):
    if args.env.endswith('.bin'):
        from env.ale.ale_env import AleEnv
        env = AleEnv(args.env, show_screen, args.use_env_frame_skip, args.frame_repeat)
    elif args.env == 'torcs':
        from env.torcs.torcs_env import TorcsEnv
        env = TorcsEnv(args.vision, args.bin, args.port, args.track, show_screen)
    elif args.env == 'vizdoom':
        from env.vizdoom.vizdoom_env import VizDoomEnv
        env = VizDoomEnv(args.config, args.use_color_input, show_screen, args.use_env_frame_skip, args.frame_repeat)

    if initialize:
        env.initialize()
    return env

