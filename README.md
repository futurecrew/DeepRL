# DeepRL

This is to implement deep reinforcement learning algorithms including following papers.
  - Deep Q Network (Human-level control through deep reinforcement learning) 
  - Deep Reinforcement Learning with Double Q-learning
  - Asynchronous Methods for Deep Reinforcement Learning
  - Prioritized Experience Replay (in working)

<img src="https://github.com/only4hj/DeepRL/blob/master/snapshot/space_invaders_a3c_lstm.gif" width="300">


## Test scores
<img src="https://github.com/only4hj/DeepRL/blob/master/snapshot/space_invaders_a3c.png" width="500">

In my PC (i7 CPU, Titan-X Maxwell),
  - A3C FF took 20 hours (4.00M global steps/hour)
  - A3C LSTM took 44 hours (1.84M global steps/hour)

## Requirements
  - Python-2.7
  - Arcade-Learning-Environment
  - Tensorflow-1.0
  - cv2
  - Vizdoom (in working)
  
## How to train
```
DQN        : python deep_rl_train.py /path/to/rom --drl dqn
Double DQN : python deep_rl_train.py /path/to/rom --drl double_dqn
A3C FF     : python deep_rl_train.py /path/to/rom --drl a3c --multi-thread-no 8
A3C LSTM   : python deep_rl_train.py /path/to/rom --drl a3c_lstm --multi-thread-no 8
```
  
## How to retrain
```
python deep_rl_train.py /path/to/rom --drl a3c --multi-thread-no 8 --snapshot path/to/snapshot_file
ex) python deep_rl_train.py /rom/breakout.bin --drl a3c --snapshot snapshot/breakout/20161114_003838/a3c_6250000
```

## How to play
```
python play.py /path/to/rom --snapshot path/to/snapshot_file
ex) python play.py /rom/space_invaders.bin --snapshot snapshot/space_invaders/20161114_003838/a3c_79993828
```

## Reference projects
  - https://github.com/tambetm/simple_dqn
  - https://github.com/miyosuda/async_deep_reinforce
  - https://github.com/muupan/async-rl
