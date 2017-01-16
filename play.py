import argparse
import pickle
from env.arguments import get_args
from deep_rl_train import DeepRLPlayer
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()    
    parser.add_argument('snapshot', type=str, help='snapshot')    
    args = parser.parse_args()
    
    if args.snapshot == None:
        print 'Usage: python player.py [path/to/snapshot/file]'
        exit()
    
    print 'Play using data_file: %s' % args.snapshot
    with open(args.snapshot + '.pickle') as f:
        player = pickle.load(f)
        player.set_global_list(None)
        player.args.show_screen = True
        player.args.test_step = 100000
        player.thread_no = 0
        player.initialize_post()
        player.model_runner.load(args.snapshot + '.weight')
        player.test(0)
    