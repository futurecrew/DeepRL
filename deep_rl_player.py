#!/usr/bin/env python

# deep_rl_player.py
# Author: Daejoong Kim
import os
from ale_python_interface import ALEInterface
import numpy as np
import random
import scipy.ndimage as ndimage
import cv2
import pickle
import threading
import time
import util
import png
from replay_memory import ReplayMemory
from sampling_manager import SamplingManager

class DeepRLPlayer:
    def __init__(self, settings, play_file=None, thread_no=None):
        self.settings = settings
        self.play_file = play_file
        self.current_state = None
        self.thread_no = thread_no
        self.gray_pixels = np.zeros((84, 84), np.float)
        self.zero_history = []
        for _ in range(4):
            self.zero_history.append(np.zeros((84, 84), dtype=np.uint8))

        self.batch_dimension = (self.settings['train_batch_size'], 
                                      self.settings['screen_history'],
                                      self.settings['screen_height'], 
                                      self.settings['screen_width'])

        self.train_step = 0
        self.epoch_done = 0
        self.train_start = time.strftime('%Y%m%d_%H%M%S')

        if self.play_file is None:
            log_file="output/%s_%s.log" % (settings['game'], self.train_start)            
            util.Logger(log_file)
        
        if os.path.exists('output') == False:
            os.makedirs('output')
        if os.path.exists('snapshot') == False:
            os.makedirs('snapshot')
            
        game_folder = self.settings['rom'].split('/')[-1]
        if '.' in game_folder:
            game_folder = game_folder.split('.')[0]
        self.snapshot_folder = 'snapshot/' + game_folder
        if os.path.exists(self.snapshot_folder) == False:
            os.makedirs(self.snapshot_folder)
        
        self.print_env()
        self.initialize_post()
        
    def initialize_post(self):
        """ initialization that should be run on __init__() or after deserialization """
        if self.settings['show_screen'] or self.play_file is not None:
            display_screen = True
        else:
            display_screen = False

        self.initialize_ale(display_screen)
        self.initialize_model()
        self.initialize_replay_memory()
        
        DebugInput(self).start()
        self.debug = False
        
    def initialize_ale(self, display_screen=False):
        self.ale = ALEInterface()
        
        #max_frames_per_episode = self.ale.getInt("max_num_frames_per_episode");
        self.ale.setInt("random_seed", random.randint(1, 1000))
        
        #random_seed = self.ale.getInt("random_seed")
        #print("random_seed: " + str(random_seed))

        if display_screen:
            self.ale.setBool('display_screen', True)
            
        self.ale.setFloat('repeat_action_probability', 0)
        
        self.ale.loadROM(self.settings['rom'])
        self.legal_actions = self.ale.getMinimalActionSet()
        print 'legal_actions: %s' % self.legal_actions
        
        (self.screen_width,self.screen_height) = self.ale.getScreenDims()
        print("width/height: " +str(self.screen_width) + "/" + str(self.screen_height))
        
        (display_width,display_height) = (1024,420)
        
        ram_size = self.ale.getRAMSize()
        ram = np.zeros((ram_size),dtype=np.uint8)    
        
    def initialize_model(self):
        if self.settings['backend'] == 'NEON':
            from model_runner_neon import ModelRunnerNeon
            self.model_runner = ModelRunnerNeon(
                                    self.settings, 
                                    max_action_no = len(self.legal_actions),
                                    batch_dimension = self.batch_dimension
                                    )
        elif self.settings['backend'] == 'TF':
            from model_runner_tf import ModelRunnerTF
            self.model_runner = ModelRunnerTF(
                                    self.settings, 
                                    max_action_no = len(self.legal_actions),
                                    batch_dimension = self.batch_dimension
                                    )
        else:
            print "settings['backend'] should be NEON or TF."

    def initialize_replay_memory(self):
        uniform_replay_memory = ReplayMemory(self.model_runner.be,
                                     self.settings['use_gpu_replay_mem'],
                                     self.settings['max_replay_memory'], 
                                     self.settings['train_batch_size'],
                                     self.settings['screen_history'],
                                     self.settings['screen_width'],
                                     self.settings['screen_height'],
                                     self.settings['minibatch_random'])
        if self.settings['prioritized_replay'] == True:
            self.replay_memory = SamplingManager(uniform_replay_memory,
                                         self.settings['use_gpu_replay_mem'],
                                         self.settings['max_replay_memory'], 
                                         self.settings['train_batch_size'],
                                         self.settings['screen_history'],
                                         self.settings['prioritized_mode'],
                                         self.settings['sampling_alpha'],
                                         self.settings['sampling_beta'],
                                         self.settings['heap_sort_term'])
        else:
            self.replay_memory = uniform_replay_memory
                                                 
    def get_action_from_model(self, mode):
        if mode == 'TEST':
            greedy_epsilon = self.settings['test_epsilon']
        else:
            min_epsilon = settings['train_min_epsilon']
            train_frequency = self.settings['train_step']
            if self.train_step * train_frequency <= 10**6:
                greedy_epsilon = 1.0 - (1.0 - min_epsilon) / 10**6 * self.train_step * train_frequency
            else:
                greedy_epsilon = min_epsilon
             
        if random.random() < greedy_epsilon:
            return random.randrange(0, len(self.legal_actions)), greedy_epsilon
        else:
            action_values = self.model_runner.predict(self.model_runner.history_buffer)
            action_index = np.argmax(action_values)
            return action_index, greedy_epsilon
        
    def print_env(self):
        print 'Start time: %s' % time.strftime('%Y.%m.%d %H:%M:%S')
        print '[ Running Environment ]'
        for key in self.settings.keys():
            print '{} : '.format(key).ljust(30) + '{}'.format(self.settings[key])
        
    def reset_game(self):
        self.model_runner.clear_history_buffer()
        self.ale.reset_game()
        self.current_state = None
        for _ in range(random.randint(4, 30)):
            reward, state, terminal, game_over = self.do_actions(0, 'TRAIN')
            self.model_runner.add_to_history_buffer(state)
    
    def save_screen_as_png(self, file_name, screen):
        pngfile = open(file_name, 'wb')
        png_writer = png.Writer(screen.shape[1], screen.shape[0], greyscale=True)
        png_writer.write(pngfile, screen)
        pngfile.close()

    def do_actions(self, action_index, mode):
        action = self.legal_actions[action_index]
        reward = 0
        terminal = False 
        lives = self.ale.lives()
        frame_repeat = self.settings['frame_repeat']

        if 'ndimage.zoom' in self.settings and self.settings['ndimage.zoom']:        
            state = self.ale.getScreenRGB()
            for _ in range(frame_repeat):
                prev_state = state
                reward += self.ale.act(action)
                state = self.ale.getScreenRGB()
                game_over = self.ale.game_over()
                if self.ale.lives() < lives or game_over:
                    terminal = True
                    if mode == 'TRAIN' and self.settings['lost_life_game_over'] == True:
                        game_over = True
                    break
            max_state = np.maximum(prev_state, state)
            
            screen = np.dot(max_state, np.array([.299, .587, .114])).astype(np.uint8)
            screen = ndimage.zoom(screen, (0.4, 0.525))
            screen.resize((84, 84))
            return reward, screen, terminal, game_over
        else:
            if self.current_state is None:
                self.current_state = self.ale.getScreenGrayscale()
            for _ in range(frame_repeat):
                prev_state = self.current_state
                reward += self.ale.act(action)
                self.current_state = self.ale.getScreenGrayscale()
                game_over = self.ale.game_over()
                if self.ale.lives() < lives or game_over:
                    terminal = True
                    if mode == 'TRAIN' and self.settings['lost_life_game_over'] == True:
                        game_over = True
                    break
            max_state = np.maximum(prev_state, self.current_state)
            resized = cv2.resize(max_state, (84, 84))
            return reward, resized, terminal, game_over
    
    def generate_replay_memory(self, count):
        print 'Generating %s replay memory' % count
        start_time = time.time()
        self.reset_game()
        for _ in range(count):
            action_index, greedy_epsilon = self.get_action_from_model('TRAIN')
            reward, state, terminal, game_over = self.do_actions(action_index, 'TRAIN')
            self.replay_memory.add(action_index, reward, state, terminal)
            self.model_runner.add_to_history_buffer(state)
                
            if(game_over):
                self.reset_game()
        
        print 'Generating replay memory took %.0f sec' % (time.time() - start_time)
        
    def test(self, epoch):
        episode = 0
        total_reward = 0
        test_start_time = time.time()
        self.reset_game()
        
        episode_reward = 0
        for step_no in range(self.settings['test_step']):
            action_index, greedy_epsilon = self.get_action_from_model('TEST')
                
            reward, state, terminal, game_over = self.do_actions(action_index, 'TEST')
                
            episode_reward += reward

            self.model_runner.add_to_history_buffer(state)
            
            if(game_over):
                episode += 1
                total_reward += episode_reward
                self.reset_game()
                episode_reward = 0
            
                if self.debug:
                    print "[ Test  %s ] %s steps, avg score: %.1f. ep: %d, elapsed: %.0fs. last e: %.3f" % \
                          (epoch, step_no, float(total_reward) / episode, episode, 
                           time.time() - test_start_time,
                           greedy_epsilon)
        
        episode = max(episode, 1)          
        print "[ Test  %s ] avg score: %.1f. elapsed: %.0fs. last e: %.3f" % \
              (epoch, float(total_reward) / episode, 
               time.time() - test_start_time,
               greedy_epsilon)
                  
    def train(self, replay_memory_no=None):
        if replay_memory_no == None:
            replay_memory_no = self.settings['train_start']
        if replay_memory_no > 0:
            self.generate_replay_memory(replay_memory_no)
        
        print 'Start training'
        start_time = time.time()
        
        for epoch in range(self.epoch_done + 1, self.settings['max_epoch'] + 1):
            epoch_total_reward = 0
            episode_total_reward = 0
            epoch_start_time = time.time()
            episode_start_time = time.time()
            self.reset_game()
            episode = 1

            for step_no in range(self.settings['epoch_step']):
                action_index, greedy_epsilon = self.get_action_from_model('TRAIN')
                
                reward, state, terminal, game_over = self.do_actions(action_index, 'TRAIN')

                episode_total_reward += reward
                epoch_total_reward += reward

                self.replay_memory.add(action_index, reward, state, terminal)
                    
                if step_no % self.settings['train_step'] == 0:
                    minibatch = self.replay_memory.get_minibatch()
                    self.model_runner.train(minibatch, self.replay_memory, self.debug)
                    self.train_step += 1
                
                    if (self.train_step % self.settings['save_step'] == 0) and (self.thread_no == None or self.thread_no == 0):
                        self.save()
                     
                self.model_runner.add_to_history_buffer(state)
                
                if game_over:
                    if episode % 500 == 0:
                        print "Ep %s, score: %s, step: %s, elapsed: %.1fs, avg: %.1f, train=%s, t_elapsed: %.1fs" % (
                                                                                episode, episode_total_reward,
                                                                                step_no, (time.time() - episode_start_time),
                                                                                float(epoch_total_reward) / episode,
                                                                                self.train_step,
                                                                                (time.time() - start_time))
                    episode_start_time = time.time()
                    
                    episode += 1
                    episode_total_reward = 0
                    
                    self.reset_game()
                    
                    #if self.settings['multi_thread_no'] > 1 and episode % self.settings['multi_thread_copy_step']  == 0:
                    #    self.queueManager.sendParams()

                if step_no > 0 and step_no % self.settings['update_step'] == 0:
                    self.model_runner.update_model()
                
            print "[ Train %s ] avg score: %.1f. elapsed: %.0fs. last e: %.3f, train=%s" % \
                  (epoch, float(epoch_total_reward) / episode, 
                   time.time() - epoch_start_time,
                   greedy_epsilon, self.train_step)
             
            # Test once every epoch
            if self.thread_no == None or self.thread_no == 0:
                self.test(epoch)
                    
            self.epoch_done = epoch
                
        self.model_runner.finish_train()
    
    def save(self):
        timesnapshot_folder = self.snapshot_folder + '/' + self.train_start
        if os.path.exists(timesnapshot_folder) == False:
            os.makedirs(timesnapshot_folder)
        
        file_name = '%s/dqn_%s' % (timesnapshot_folder, self.train_step)
        with open(file_name + '.pickle', 'wb') as f:
            pickle.dump(self, f)
            self.model_runner.save(file_name + '.weight')
            #print '%s dumped' % file_name
        
    def __getstate__(self):
        self.replay_memory_no = self.replay_memory.count
        d = dict(self.__dict__)
        del d['ale']
        del d['replay_memory']
        del d['model_runner']
        return d
        
class DebugInput(threading.Thread):
    def __init__(self, player):
        threading.Thread.__init__(self)
        self.player = player
        self.running = True
    
    def run(self):
        time.sleep(5)
        while (self.running):
            input = raw_input('')
            if input == 'd':
                self.player.debug = not self.player.debug
                print 'Debug mode : %s' % self.player.debug
                
    def finish(self):
        self.running = False
        
global_data = []

def fork_thread(settings, thread_no):
    print 'fork_thread %s' % thread_no
    player = DeepRLPlayer(settings, thread_no= thread_no)
    player.train()
    
def train(settings, save_file=None):
    if save_file is not None:        # retrain
        with open(save_file + '.pickle') as f:
            player = pickle.load(f)
            player.train_start = time.strftime('%Y%m%d_%H%M%S')
            log_file="output/%s_%s.log" % (settings['game'], player.train_start)            
            util.Logger(log_file)
            print 'Resume trainig: %s' % save_file
            player.print_env()
            player.initialize_post()
            player.model_runner.load(save_file + '.weight')
            player.train(replay_memory_no = player.replay_memory_no)
    else:
        multithread_no = settings['multi_thread_no']
        if multithread_no > 1:
            threadList = []                
            for i in range(multithread_no):        
                print 'creating a thread[%s]' % i
                t = threading.Thread(target=fork_thread, args=(settings, i))
                t.start()
                threadList.append(t)
                
            while True:
                time.sleep(1000)
            
        else:
            player = DeepRLPlayer(settings)
            player.train_step = 0
            player.train()

def play(settings, play_file=None):
    print 'Play using data_file: %s' % play_file
    player = DeepRLPlayer(settings, play_file)
    player.model_runner.load(play_file + '.weight')
    player.test(0)
    
if __name__ == '__main__':    
    settings = {}

    #settings['game'] = 'breakout'
    #settings['game'] = 'space_invaders'
    #settings['game'] = 'enduro'
    #settings['game'] = 'kung_fu_master'
    #settings['game'] = 'krull'
    settings['game'] = 'hero'
    #settings['game'] = 'qbert'
    #settings['game'] = 'seaquest'

    settings['rom'] = '/media/big/download/roms/%s.bin' % settings['game']    
    settings['frame_repeat'] = 4    
    settings['show_screen'] = False
    settings['use_keyboard'] = False
    settings['train_batch_size'] = 32
    settings['max_replay_memory'] = 1000000
    settings['max_epoch'] = 200
    settings['epoch_step'] = 250000
    settings['discount_factor'] = 0.99
    settings['train_min_epsilon'] = 0.1           # Minimum greey epsilon value for exloration
    settings['update_step'] = 10000               # Copy train network into target network every this train step
    settings['train_start'] = 50000                   # Start training after filling this replay memory size
    settings['train_step'] = 4                            # Train every this screen step
    settings['test_step'] = 125000                   # Test for this number of steps
    settings['test_epsilon'] = 0.05                   # Greed epsilon for test
    settings['save_step'] = 50000                    # Save result every this training step
    settings['screen_width'] = 84
    settings['screen_height'] = 84
    settings['screen_history'] = 4
    settings['learning_rate'] = 0.00025
    settings['rms_decay'] = 0.95
    settings['lost_life_game_over'] = True
    settings['update_step_in_step_no'] = True
    settings['double_dqn'] = False
    settings['prioritized_replay'] = False
    settings['use_priority_weight'] = True
    settings['minibatch_random'] = True        # Whether to use random indexing for minibatch or not 
    settings['multi_thread_no'] = 0                # Number of multiple threads for Asynchronous RL

    #settings['backend'] = 'NEON'
    settings['backend'] = 'TF'
    
    settings['tf_version'] = 'v1'
    settings['clip_delta'] = True
    settings['use_self.current_state'] = True
    settings['use_successive_two_frames'] = True
    settings['dnn_initializer'] = 'fan_in'
    #settings['dnn_initializer'] = 'xavier'
    settings['optimizer'] = 'RMSProp'
    #settings['ndimage.zoom'] = True

    #settings['use_gpu_replay_mem'] = True           # Whether to store replay memory in gpu or not to speed up leraning
    settings['use_gpu_replay_mem'] = False

    
    # Double DQN hyper params
    settings['double_dqn'] = True
    settings['train_min_epsilon'] = 0.01
    settings['test_epsilon'] = 0.001
    settings['update_step'] = 30000


    # Prioritized experience replay params for RANK
    settings['prioritized_replay'] = True
    settings['learning_rate'] = 0.00025 / 4
    settings['prioritized_mode'] = 'RANK'
    settings['sampling_alpha'] = 0.7
    settings['sampling_beta'] = 0.5
    settings['heap_sort_term'] = 250000


    """
    # Prioritized experience replay params for PROPORTION
    settings['prioritized_replay'] = True
    settings['learning_rate'] = 0.00025 / 4
    #settings['learning_rate'] = 0.00025 * 5
    settings['prioritized_mode'] = 'PROPORTION'
    settings['sampling_alpha'] = 0.6
    settings['sampling_beta'] = 0.4
    settings['heap_sort_term'] = 250000
    """
    
    """
    # Asynchronous RL
    settings['train_start'] = settings['train_batch_size'] + settings['screen_history'] - 1 
    settings['max_replay_memory'] = settings['train_start'] + 100
    settings['minibatch_random'] = False
    settings['multi_thread_no'] = 2
    settings['multi_thread_sync_step'] = 10
    """
    
    data_file = None    
    #data_file = 'snapshot/breakout/dqn_neon_3100000.prm'
    #data_file = 'snapshot/%s/%s' % (settings['game'], '20160831_200101/dqn_200000')
    
    train(settings, data_file)
    #play(settings, data_file)
