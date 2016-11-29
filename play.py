from arguments import get_args
from deep_rl_train import DeepRLPlayer
        
if __name__ == '__main__':
    args = get_args()
    if args.snapshot == None:
        print 'Usage: python player.py [path/to/rom/file] --snapshot [path/to/snapshot/file]'
        exit()
    args.show_screen = True
    
    print 'Play using data_file: %s' % args.snapshot
    player = DeepRLPlayer(args, args.snapshot)
    player.model_runner.load(args.snapshot + '.weight')
    player.debug = True
    player.test(0)
    