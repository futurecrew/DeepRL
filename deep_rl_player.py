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
import argparse
import png
from replay_memory import ReplayMemory
from sampling_manager import SamplingManager
from model_runner_neon import ModelRunnerNeon
from model_runner_tf_a3c import ModelRunnerTFA3C
from model_runner_tf_a3c_lstm import ModelRunnerTFA3CLstm
from model_runner_tf_async import ModelRunnerTFAsync
from model_runner_tf import ModelRunnerTF
from network_model import ModelA3C, ModelA3CLstm

class DeepRLPlayer:
    def __init__(self, args, play_file=None, thread_no=0, global_list=None):
        self.args = args
        self.play_file = play_file
        self.current_state = None
        self.thread_no = thread_no
        self.global_list = global_list
        self.gray_pixels = np.zeros((84, 84), np.float)
        self.zero_history = []
        for _ in range(4):
            self.zero_history.append(np.zeros((84, 84), dtype=np.uint8))

        if self.args.screen_order == 'hws':
            self.batch_dimension = (self.args.train_batch_size, 
                                      self.args.screen_height, 
                                      self.args.screen_width,
                                      self.args.screen_history)
        else:
            self.batch_dimension = (self.args.train_batch_size, 
                                      self.args.screen_history,
                                      self.args.screen_height, 
                                      self.args.screen_width)

        self.train_step = 0
        self.epoch_done = 0
        self.next_test_thread_no = 0
        self.train_start = time.strftime('%Y%m%d_%H%M%S')

        if self.play_file is None and self.thread_no == 0:
            log_file="output/%s_%s.log" % (args.game, self.train_start)            
            util.Logger(log_file)
        
        if os.path.exists('output') == False:
            os.makedirs('output')
        if os.path.exists('snapshot') == False:
            os.makedirs('snapshot')
            
        game_folder = self.args.rom.split('/')[-1]
        if '.' in game_folder:
            game_folder = game_folder.split('.')[0]
        self.snapshot_folder = 'snapshot/' + game_folder
        if os.path.exists(self.snapshot_folder) == False:
            os.makedirs(self.snapshot_folder)
        
        self.print_env()
        self.initialize_post()
        
    def initialize_post(self):
        """ initialization that should be run on __init__() or after deserialization """
        if (self.args.show_screen and self.thread_no == 0) or self.play_file is not None:
            display_screen = True
        else:
            display_screen = False

        self.initialize_ale(display_screen)
        self.initialize_model()
        self.initialize_replay_memory()
        
        #DebugInput(self).start()
        self.debug = False
        
    def initialize_ale(self, display_screen=False):
        self.ale = ALEInterface()
        self.ale.setInt("random_seed", random.randint(1, 1000))
        if display_screen:
            self.ale.setBool('display_screen', True)

        if self.args.use_ale_frame_skip == True:
            self.ale.setInt('frame_skip', self.args.frame_repeat)
            self.ale.setBool('color_averaging', True)        
 
        self.ale.setFloat('repeat_action_probability', 0)
        self.ale.loadROM(self.args.rom)
        self.legal_actions = self.ale.getMinimalActionSet()
        print 'legal_actions: %s' % self.legal_actions
        
        (self.screen_width,self.screen_height) = self.ale.getScreenDims()
        print("width/height: " +str(self.screen_width) + "/" + str(self.screen_height))
        
        (display_width,display_height) = (1024,420)
        
        ram_size = self.ale.getRAMSize()
        ram = np.zeros((ram_size),dtype=np.uint8)    
        
    def initialize_model(self):
        if self.args.backend == 'NEON':
            self.model_runner = ModelRunnerNeon(
                                    self.args, 
                                    max_action_no = len(self.legal_actions),
                                    batch_dimension = self.batch_dimension
                                    )
        elif self.args.backend == 'TF':
            if self.args.asynchronousRL == True:
                if self.args.asynchronousRL_type == 'A3C':            
                    self.model_runner = ModelRunnerTFA3C(
                                    self.global_list,
                                    self.args, 
                                    max_action_no = len(self.legal_actions),
                                    thread_no = self.thread_no
                                    )
                elif self.args.asynchronousRL_type == 'A3C_LSTM':            
                    self.model_runner = ModelRunnerTFA3CLstm(
                                    self.global_list,
                                    self.args, 
                                    max_action_no = len(self.legal_actions),
                                    thread_no = self.thread_no
                                    )
                else:
                    self.model_runner = ModelRunnerTFAsync(
                                    self.global_list,
                                    self.args, 
                                    max_action_no = len(self.legal_actions),
                                    thread_no = self.thread_no
                                    )
            else:
                self.model_runner = ModelRunnerTF(
                                    self.args, 
                                    max_action_no = len(self.legal_actions),
                                    batch_dimension = self.batch_dimension
                                    )
        else:
            print "args.backend should be NEON or TF."

    def initialize_replay_memory(self):
        uniform_replay_memory = ReplayMemory(
                                     self.args.max_replay_memory, 
                                     self.args.train_batch_size,
                                     self.args.screen_history,
                                     self.args.screen_width,
                                     self.args.screen_height,
                                     self.args.minibatch_random,
                                     self.args.screen_order)
        if self.args.prioritized_replay == True:
            self.replay_memory = SamplingManager(uniform_replay_memory,
                                         self.args.max_replay_memory, 
                                         self.args.train_batch_size,
                                         self.args.screen_history,
                                         self.args.prioritized_mode,
                                         self.args.sampling_alpha,
                                         self.args.sampling_beta,
                                         self.args.heap_sort_term)
        else:
            self.replay_memory = uniform_replay_memory

    def get_greedy_epsilon(self, mode):
        if mode == 'TEST':
            greedy_epsilon = self.args.test_epsilon
        else:
            min_epsilon = args.train_min_epsilon
            train_frequency = self.args.train_step
            if self.train_step * train_frequency <= 10**6:
                greedy_epsilon = 1.0 - (1.0 - min_epsilon) / 10**6 * self.train_step * train_frequency
            else:
                greedy_epsilon = min_epsilon
        return greedy_epsilon
           
    def choose_action(self, action_values):
        rand_value = random.random()
        sum_value = 0
        action_index = 0
        for i, action_value in enumerate(action_values):
            sum_value += action_value
            if rand_value <= sum_value:
                action_index = i
                break
        return action_index
            
    def get_action_index(self, mode):
        state = self.replay_memory.history_buffer
        if self.args.choose_max_action:
            greedy_epsilon = self.get_greedy_epsilon(mode)
            if random.random() < greedy_epsilon:
                return random.randrange(0, len(self.legal_actions)), greedy_epsilon
            else:
                action_values = self.model_runner.predict(state)
                action_index = np.argmax(action_values)
                return action_index, greedy_epsilon
        else:
            action_values = self.model_runner.predict(state)
            action_index = self.choose_action(action_values)
            """
            if mode == 'TRAIN':
                action_index = self.choose_action(action_values)
            else:
                action_index = np.argmax(action_values)
            """
            return action_index, 0
                                                 
    def get_action_state_value(self, mode):
        state = self.replay_memory.history_buffer
        action_values, state_value = self.model_runner.predict_action_state(state)
        if self.args.choose_max_action:
            action_index =  np.argmax(action_values)
        else:        
            action_index = self.choose_action(action_values)
        return action_index, state_value

    def get_state_value(self):
        state = self.replay_memory.history_buffer
        return self.model_runner.predict_state(state)
    
    def print_env(self):
        if self.args.asynchronousRL == False or self.thread_no == 0:
            print 'Start time: %s' % time.strftime('%Y.%m.%d %H:%M:%S')
            print '[ Running Environment ]'
            for arg in vars(self.args):
                print '{} : '.format(arg).ljust(30) + '{}'.format(getattr(self.args, arg))
        
    def reset_game(self):
        self.replay_memory.clear_history_buffer()
        self.ale.reset_game()
        self.current_state = None
        action_index = 0
        
        if self.args.asynchronousRL_type == 'A3C_LSTM':
            self.model_runner.reset_lstm_state()
        
        for _ in range(random.randint(4, 30)):
            reward, state, terminal, game_over = self.do_actions(action_index, 'TRAIN')
            self.replay_memory.add(action_index, reward, state, terminal)
    
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
        frame_repeat = self.args.frame_repeat

        if self.args.crop_image:        
            state = self.ale.getScreenRGB()
            for _ in range(frame_repeat):
                prev_state = state
                reward += self.ale.act(action)
                state = self.ale.getScreenRGB()
                game_over = self.ale.game_over()
                if (self.args.lost_life_terminal == True and self.ale.lives() < lives) or game_over:
                    terminal = True
                    if mode == 'TRAIN' and self.args.lost_life_game_over == True:
                        game_over = True
                    break
            new_state = np.maximum(prev_state, state)
            
            screen = np.dot(new_state, np.array([.299, .587, .114])).astype(np.uint8)
            screen = ndimage.zoom(screen, (0.4, 0.525))
            screen.resize((84, 84))
            return reward, screen, terminal, game_over
        else:
            if self.args.use_ale_frame_skip == True:
                reward += self.ale.act(action)
                new_state = self.ale.getScreenGrayscale()
                game_over = self.ale.game_over()
                if (self.args.lost_life_terminal == True and self.ale.lives() < lives) or game_over:
                    terminal = True
                    if mode == 'TRAIN' and self.args.lost_life_game_over == True:
                        game_over = True
            else:
                if self.current_state is None:
                    self.current_state = self.ale.getScreenGrayscale()                
                for _ in range(frame_repeat):
                    prev_state = self.current_state
                    reward += self.ale.act(action)
                    self.current_state = self.ale.getScreenGrayscale()
                    game_over = self.ale.game_over()
                    if (self.args.lost_life_terminal == True and self.ale.lives() < lives) or game_over:
                        terminal = True
                        if mode == 'TRAIN' and self.args.lost_life_game_over == True:
                            game_over = True
                        break
                new_state = np.maximum(prev_state, self.current_state)
                
            resized = cv2.resize(new_state, (84, 84))
            return reward, resized, terminal, game_over
    
    def generate_replay_memory(self, count):
        if self.thread_no == 0:
            print 'Generating %s replay memory' % count
        start_time = time.time()
        self.reset_game()
        for _ in range(count):
            action_index, greedy_epsilon = self.get_action_index('TRAIN')
            reward, state, terminal, game_over = self.do_actions(action_index, 'TRAIN')
            self.replay_memory.add(action_index, reward, state, terminal)
                
            if(game_over):
                self.reset_game()
        
        if self.thread_no == 0:
            print 'Generating replay memory took %.0f sec' % (time.time() - start_time)
        
    def test(self, epoch):
        episode = 0
        total_reward = 0
        test_start_time = time.time()
        self.reset_game()
        
        episode_reward = 0
        for step_no in range(self.args.test_step):
            action_index, greedy_epsilon = self.get_action_index('TEST')
                
            reward, state, terminal, game_over = self.do_actions(action_index, 'TEST')
                
            episode_reward += reward

            self.replay_memory.add_to_history_buffer(state)
            
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
            replay_memory_no = self.args.train_start
        if replay_memory_no > 0:
            self.generate_replay_memory(replay_memory_no)
        
        print 'Start training'
        start_time = time.time()
        for epoch in range(self.epoch_done + 1, self.args.max_epoch + 1):
            epoch_total_reward = 0
            episode_total_reward = 0
            epoch_start_time = time.time()
            episode_start_time = time.time()
            self.reset_game()
            episode = 1

            for step_no in range(1, self.args.epoch_step + 1):
                action_index, greedy_epsilon = self.get_action_index('TRAIN')
                reward, state, terminal, game_over = self.do_actions(action_index, 'TRAIN')

                episode_total_reward += reward
                epoch_total_reward += reward

                self.replay_memory.add(action_index, reward, state, terminal)
                    
                if step_no % self.args.train_step == 0:
                    minibatch = self.replay_memory.get_minibatch()
                    self.model_runner.train(minibatch, self.replay_memory, self.debug)
                    self.train_step += 1
                
                    if self.train_step % self.args.save_step == 0 and self.thread_no == 0:
                        self.save()
                     
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
                    
                if step_no > 0 and step_no % self.args.update_step == 0:
                    self.model_runner.update_model()
                
            print "[ Train %s ] avg score: %.1f. elapsed: %.0fs. last e: %.3f, train=%s" % \
                  (epoch, float(epoch_total_reward) / episode, 
                   time.time() - epoch_start_time,
                   greedy_epsilon, self.train_step)
             
            # Test once every epoch
            if args.asynchronousRL == False:
                self.test(epoch)
            else:
                if self.thread_no == self.next_test_thread_no:
                    self.test(epoch)
                self.next_test_thread_no = (self.next_test_thread_no + 1) % self.args.multi_thread_no
                    
            self.epoch_done = epoch
                
        self.model_runner.finish_train()

    def _anneal_learning_rate(self, max_global_step_no, global_step_no):
        learning_rate = self.args.learning_rate * (max_global_step_no - global_step_no) / max_global_step_no
        if learning_rate < 0.0:
            learning_rate = 0.0
        return learning_rate

    def train_async_a3c(self, replay_memory_no=None):
        global global_step_no
        max_global_step_no = self.args.max_epoch * self.args.epoch_step * self.args.multi_thread_no
        last_time = 0
        last_global_step_no = 0
        
        if replay_memory_no == None:
            replay_memory_no = self.args.train_start
        if replay_memory_no > 0:
            self.generate_replay_memory(replay_memory_no)
        
        if self.thread_no == 0:
            print 'max_global_step_no : %s' % max_global_step_no
            print 'Start training async_a3c'
        start_time = time.time()
        
        for epoch in range(self.epoch_done + 1, self.args.max_epoch + 1):
            epoch_total_reward = 0
            episode_total_reward = 0
            epoch_start_time = time.time()
            episode_start_time = time.time()
            self.reset_game()
            episode = 1
            step_no = 1
            
            while step_no < self.args.epoch_step:
                v_pres = []
    
                lstm_state_value = self.model_runner.get_lstm_state()            
                for i in range(self.args.train_step):
                    action_index, state_value = self.get_action_state_value('TRAIN')
                    reward, state, terminal, game_over = self.do_actions(action_index, 'TRAIN')

                    self.replay_memory.add(action_index, reward, state, terminal)
                    v_pres.append(state_value)
    
                    episode_total_reward += reward
                    epoch_total_reward += reward
    
                    if game_over:
                        break
                
                v_pres.reverse()
                data_len = i + 1
                step_no += data_len
                global_step_no += data_len
                
                if terminal:
                    v_post = 0
                else:
                    v_post = self.get_state_value()
                prestates, actions, rewards, _, terminals = self.replay_memory.get_minibatch(data_len)
                learning_rate = self._anneal_learning_rate(max_global_step_no, global_step_no)

                self.model_runner.set_lstm_state(lstm_state_value)            
                self.model_runner.train(prestates, v_pres, actions, rewards, terminals, v_post, learning_rate)

                self.train_step += 1
                
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

                    if self.thread_no == 0:
                        current_time = time.time()
                        if current_time - last_time > 3600:
                            steps_per_sec = float(global_step_no - last_global_step_no) / (current_time - last_time)
                            if last_time != 0:
                                print '%.0f global_step/sec. %.2fM global_step/hour' % (steps_per_sec, steps_per_sec * 3600 / 10**6)
                            last_time = current_time
                            last_global_step_no = global_step_no

            print "[ Train %s ] avg score: %.1f. elapsed: %.0fs. learning_rate=%.5f" % \
                  (epoch, float(epoch_total_reward) / episode, 
                   time.time() - epoch_start_time, learning_rate)
                
            if self.thread_no == 0:
                self.save()
             
            # Test once every epoch
            if args.run_test == True:
                if args.asynchronousRL == False:
                    self.test(epoch)
                else:
                    if self.thread_no == self.next_test_thread_no:
                        self.test(epoch)
                    self.next_test_thread_no = (self.next_test_thread_no + 1) % self.args.multi_thread_no

            self.epoch_done = epoch
                
        print 'thread %s finished' % self.thread_no

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
        del d['global_list']
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
global_step_no = 0

def train(args, save_file=None):
    if save_file is not None:        # retrain
        with open(save_file + '.pickle') as f:
            player = pickle.load(f)
            player.train_start = time.strftime('%Y%m%d_%H%M%S')
            log_file="output/%s_%s.log" % (args.game, player.train_start)            
            util.Logger(log_file)
            print 'Resume trainig: %s' % save_file
            player.print_env()
            player.initialize_post()
            player.model_runner.load(save_file + '.weight')
            player.train(replay_memory_no = player.replay_memory_no)
    else:
        if args.asynchronousRL == True:
            threadList = []
            playerList = []

            ale = ALEInterface()
            ale.loadROM(args.rom)
            legal_actions = ale.getMinimalActionSet()
            
            # initialize global model
            if args.asynchronousRL_type == 'A3C':
                model = ModelA3C('global', args.network_type, True,  len(legal_actions))
            elif args.asynchronousRL_type == 'A3C_LSTM':     
                model = ModelA3CLstm('global', args.network_type, True,  len(legal_actions))

            global_list = model.prepare_global(args.rms_decay, args.rms_epsilon)

            for i in range(args.multi_thread_no):        
                print 'creating a thread[%s]' % i
                player = DeepRLPlayer(args, thread_no= i, global_list=global_list)
                playerList.append(player)

            model.init_global(global_list[0])

            for player in playerList:
                if args.asynchronousRL_type.startswith('A3C'):
                    target_func = player.train_async_a3c
                else:
                    target_func = player.train
                t = threading.Thread(target=target_func, args=())
                t.start()
                threadList.append(t)
                
            for thread in threadList:
                thread.join()
            
        else:
            player = DeepRLPlayer(args)
            player.train_step = 0
            player.train()

def play(args, play_file=None):
    print 'Play using data_file: %s' % play_file
    player = DeepRLPlayer(args, play_file)
    player.model_runner.load(play_file + '.weight')
    player.test(0)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()    

    #parser.add_argument('--game', type=str, default='breakout', help='game name')
    #parser.add_argument('--game', type=str, default='space_invaders', help='game name')
    #parser.add_argument('--game', type=str, default='enduro', help='game name')
    parser.add_argument('--game', type=str, default='kung_fu_master', help='game name')
    #parser.add_argument('--game', type=str, default='krull', help='game name')
    #parser.add_argument('--game', type=str, default='hero', help='game name')
    #parser.add_argument('--game', type=str, default='qbert', help='game name')       # Something wrong with reward?
    #parser.add_argument('--game', type=str, default='seaquest', help='game name')
    #parser.add_argument('--game', type=str, default='pong', help='game name')
    #parser.add_argument('--game', type=str, default='beam_rider', help='game name')
    
    # Asynchronous RL
    parser.add_argument('--asynchronousRL', type=bool, default=True, help='')
    parser.add_argument('--asynchronousRL-type', type=str, default='A3C_LSTM', help='')
    #parser.add_argument('--asynchronousRL-type', type=str, default='A3C', help='')
    #parser.add_argument('--asynchronousRL-type', type=str, default='1Q', help='')
    
    #parser.add_argument('--network-type', type=str, default='nature', help='network model nature or nips')
    parser.add_argument('--network-type', type=str, default='nips', help='') 
    parser.add_argument('--dqn-type', type=str, default='dqn', help='dqn, double-dqn, prioritized-rank, prioritized-proportion') 

    args = parser.parse_args()
    args.rom ='/media/big/download/roms/%s.bin' % args.game

    args.screen_width = 84    # input screen width
    args.screen_height = 84    # input screen height
    args.screen_history = 4    # input screen history
    args.frame_repeat = 4    # how many frames to repeat in ale for one predicted action
    args.show_screen = False    # whether to show ale display
    args.use_ale_frame_skip = False    # whether to use ale frame_skip feature
    args.discount_factor = 0.99    # RL discount factor
    args.test_step = 125000    # test for this number of steps
    args.lost_life_game_over = False    # whether to regard lost life as game over
    args.lost_life_terminal = False    # whether to regard lost life as terminal state
    args.crop_image = False         # Crop input image or zoom image
    args.run_test = True    # 
    args.backend = 'TF'    # Deep learning library backend (TF, NEON)
    
    if args.backend == 'TF':
        args.screen_order = 'hws'   # dimension order in replay memory (height, width, screen)
    elif args.backend == 'NEON':
        args.screen_order = 'shw'   # (screen, height, width)
        args.dnn_initializer = 'fan_in'    # 
        args.use_gpu_replay_mem = False       # Whether to store replay memory in gpu or not to speed up leraning        
    
    if args.asynchronousRL == True:
        args.train_batch_size = 5
        args.max_replay_memory = 100
        args.max_epoch = 20
        args.epoch_step = 500000    # 4,000,000 global steps / 8 threads
        args.train_start = 100        
        args.train_step = 5                 
        args.learning_rate = 0.0007                 # RL learning rate
        args.rms_decay = 0.99                 # rms decay
        args.rms_epsilon = 0.1                 # rms epsilon
        args.choose_max_action = False
        args.multi_thread_no = 8              # Number of multiple threads for Asynchronous RL
        args.minibatch_random = False       # whether to use random indexing or sequential indexing for minibatch
        args.save_step = 4000000            # save result every this training step
        args.prioritized_replay = False
    else:
        args.train_batch_size = 32
        args.max_replay_memory = 1000000
        args.max_epoch = 200
        args.epoch_step = 250000
        args.train_start = 50000        # start training after filling this replay memory size
        args.train_step = 4                 # Train every this screen step
        args.learning_rate = 0.00025                 
        args.rms_decay = 0.95                
        args.rms_epsilon = 0.01                 
        args.choose_max_action = True
        args.multi_thread_no = 0
        args.minibatch_random = True
        args.train_min_epsilon = 0.1    # minimum greedy epsilon value for exloration
        args.update_step = 10000    # copy train network into target network every this train step
        args.optimizer = 'RMSProp'    # 
        args.save_step = 50000            # save result every this training step

        if args.dqn_type == 'dqn':                     # DQN hyper params
            args.double_dqn = False                   # whether to use double dqn
            args.test_epsilon = 0.05                    # greedy epsilon for test
            args.prioritized_replay = False
        elif args.dqn_type == 'double_dqn':    # Double DQN hyper params
            args.double_dqn = True    # 
            args.train_min_epsilon = 0.01    # 
            args.test_epsilon = 0.001    # 
            args.update_step = 30000    # 
        elif args.dqn_type == 'prioritized_rank':    # Prioritized experience replay params for RANK
            args.prioritized_replay = True    # 
            args.learning_rate = 0.00025 / 4    # 
            args.prioritized_mode = 'RANK'    # 
            args.sampling_alpha = 0.7    # 
            args.sampling_beta = 0.5    # 
            args.heap_sort_term = 250000    # 
            args.prioritized_replay = True
            args.use_priority_weight = True    # whether to priority weight
        elif args.dqn_type == 'prioritized_proportion':    # Prioritized experience replay params for PROPORTION
            args.prioritized_replay = True    # 
            args.learning_rate = 0.00025 / 4    # 
            args.prioritized_mode = 'PROPORTION'    # 
            args.sampling_alpha = 0.6    # 
            args.sampling_beta = 0.4    # 
            args.heap_sort_term = 250000    # 
            args.prioritized_replay = True
            args.use_priority_weight = True    # whether to priority weight
    

    data_file = None    
    #data_file = 'snapshot/%s/%s' % (args.game, '20161030_144636/dqn_702052
    #play(args, data_file)
    
    train(args, data_file)
