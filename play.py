from arguments import get_args
from deep_rl_train import DeepRLPlayer
        
if __name__ == '__main__':
    args = get_args()
    if args.replay_file == None:
        print 'Usage: python player.py [path_to_rom_file] --replay-file [path_to_snapshot_file]'
        exit()
    args.show_screen = True
    
    print 'Play using data_file: %s' % args.replay_file
    player = DeepRLPlayer(args, args.replay_file)
    player.model_runner.load(args.replay_file + '.weight')
    player.debug = True
    player.test(0)
    