# DeepRL

This project implements deep reinforcement learning algorithms including following papers.
  - Deep Q Network (Human-level control through deep reinforcement learning) 
  - Deep Reinforcement Learning with Double Q-learning
  - Asynchronous Methods for Deep Reinforcement Learning
  - Prioritized Experience Replay

<img src="https://github.com/only4hj/DeepRL/blob/master/snapshot/space_invaders_a3c_lstm.gif" width="280">
<img src="https://github.com/only4hj/DeepRL/blob/master/snapshot/breakout_a3c.gif" width="280">
<img src="https://github.com/only4hj/DeepRL/blob/master/snapshot/hero.gif" width="280">



## Test scores
<nobr>
<img src="https://github.com/only4hj/DeepRL/blob/master/snapshot/space_invaders_a3c.png" width="420">
<img src="https://github.com/only4hj/DeepRL/blob/master/snapshot/breakout_a3c.png" width="420">
</nobr>
<img src="https://github.com/only4hj/DeepRL/blob/master/snapshot/hero_priority.png" width="420">


In my PC (i7 CPU, Titan-X Maxwell),
  - DQN took 96 hours for 80M steps (shown 11M steps)
  - Double-Q took 112 hours for 80M steps (shown 11M steps)
  - Prioritized took 112 hours for 80M steps (shown 11M steps)
  - A3C FF took 20 hours for 80M global steps
  - A3C LSTM took 44 hours for 80M global steps

## Requirements
  - Python-2.7
  - Numpy
  - Arcade-Learning-Environment
  - Tensorflow-0.11
  - opencv2
  - Vizdoom (in working)
  <br><br>
  See <a href="INSTALL.md">this</a> for installation.
  
## How to train
```
DQN         : python deep_rl_train.py /path/to/rom --drl dqn
Double DQN  : python deep_rl_train.py /path/to/rom --drl double_dqn
Prioritized : python deep_rl_train.py /path/to/rom --drl prioritized_rank
A3C FF      : python deep_rl_train.py /path/to/rom --drl a3c --thread-no 8
A3C LSTM    : python deep_rl_train.py /path/to/rom --drl a3c_lstm --thread-no 8
```
  
## How to retrain
```
python deep_rl_train.py /path/to/rom --drl a3c --thread-no 8 --snapshot path/to/snapshot_file
ex) python deep_rl_train.py /rom/breakout.bin --drl a3c --thread-no 8 --snapshot snapshot/breakout/20161114_003838/a3c_6250000
```

## How to play
```
python play.py path/to/snapshot_file
ex) python play.py snapshot/space_invaders/20161114_003838/a3c_79993828
```

## Debug console commands
While training you can send several debug commands in the console.
- p : print debug message or not
- u : pause training or not
- quit : finish training
- d : display screen or not
- - : make displaying faster
- + : make displaying slower


## Reference projects
  - https://github.com/tambetm/simple_dqn
  - https://github.com/miyosuda/async_deep_reinforce
  - https://github.com/muupan/async-rl
