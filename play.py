from arguments import get_args
from deep_rl_train import DeepRLPlayer
        
if __name__ == '__main__':
    args = get_args()
    args.replay_file = 'snapshot/%s/%s' % (args.game, '20161030_144636/dqn_2404871')
    if args.replay_file == None:
        print 'Usage: python player.py replay_file'
        exit()

    print 'Play using data_file: %s' % args.replay_file
    player = DeepRLPlayer(args, args.replay_file)
    player.model_runner.load(args.replay_file + '.weight')
    player.debug = True
    player.test(0)
    