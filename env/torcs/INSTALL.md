## How to install Torcs environment in Ubuntu

- This project is tested in python-2.7 and Ubuntu 14.04.
  <br>

- xautomation (http://linux.die.net/man/7/xautomation)
  <br>
  sudo apt-get update
  <br>
  sudo apt-get install xautomation
- sudo apt-get install libglib2.0-dev  libgl1-mesa-dev libglu1-mesa-dev  freeglut3-dev  libplib-dev  libopenal-dev libalut-dev libxi-dev libxmu-dev libxrender-dev  libxrandr-dev libpng12-dev 
- Gym-TORCS
  <br>
  Download source from https://github.com/ugo-nama-kun/gym_torcs
  <br>
  Untar the archive
  <br>
  cd into the vtorcs-RL-color directory
  <br>
  ./configure
  <br>
  make
  <br>
  sudo make install
  <br>
  sudo make datainstall

- Initialization of the Race
  <br>
  Go to the link below and follow instructions.
  <br>
  https://github.com/ugo-nama-kun/gym_torcs#initialization-of-the-race
  
## How to train
```
Without vision   : python train.py torcs --ddpg
With vision only : python train.py torcs --ddpg --vision
```
