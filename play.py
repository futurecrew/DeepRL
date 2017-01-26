import argparse
import pickle
from env.arguments import get_args
from train import Trainer
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()    
    parser.add_argument('snapshot', type=str, help='snapshot')    
    args = parser.parse_args()
    
    if args.snapshot == None:
        print 'Usage: python play.py [path/to/snapshot/file]'
        exit()
    
    print 'Play using data_file: %s' % args.snapshot
    with open(args.snapshot + '.pickle') as f:
        trainer = pickle.load(f)
        trainer.set_global_list(None)
        trainer.args.show_screen = True
        trainer.args.test_step = 100000
        trainer.thread_no = 0
        trainer.initialize_post()
        trainer.model_runner.load(args.snapshot + '.weight')
        trainer.test(0)
    