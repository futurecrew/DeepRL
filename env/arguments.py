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
    parser.add_argument('--thread-no', type=int, default=1, help='Number of multiple threads for Asynchronous RL')
    parser.add_argument('--network', type=str, default='nips', choices=['nips', 'nature'], help='network model nature or nips') 
    parser.add_argument('--drl', type=str, default='dqn', choices=['dqn', 'double_dqn', 'prioritized_rank', 'prioritized_proportion', 'a3c_lstm', 'a3c', '1q'])
    parser.add_argument('--snapshot', type=str, default=None, help='trained file to resume training or to replay') 
    parser.add_argument('--device', type=str, default='', help='gpu or cpu')
    parser.add_argument('--show-screen', action='store_true', help='whether to show display or not')
    parser.add_argument('--config', type=str, default=None, help='config file for vizdoom')
    parser.set_defaults(show_screen=False)
    
    args = parser.parse_args()
    args.game = get_game_name(args.rom)
    
    if args.rom == 'vizdoom':
        args.env = 'vizdoom'
        from env.vizdoom.vizdoom_env import initialize_args
        initialize_args(args)
    else:
        args.env = 'ale'
        from env.ale.ale_env import initialize_args
        initialize_args(args)
    
    return args
