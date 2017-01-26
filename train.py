# train.py
# Author: Daejoong Kim

import os
import numpy as np
import random
import cv2
import pickle
import sys
import threading
import time
import util
from select import select
from replay_memory import ReplayMemory
from sampling_manager import SamplingManager
from network_model.model_tf_a3c import ModelA3C, ModelRunnerTFA3C
from network_model.model_tf_a3c_lstm import ModelA3CLstm, ModelRunnerTFA3CLstm
from network_model.model_tf_async import ModelRunnerTFAsync, load_global_vars
from network_model.model_tf_ddpg import ModelRunnerTFDdpg
from network_model.model_tf import Model, ModelRunnerTF
from env.arguments import get_args, get_env

class Trainer:
    def __init__(self, args, play_file=None, thread_no=0, global_list=None):
        self.args = args
        self.play_file = play_file
        self.current_state = None
        self.thread_no = thread_no
        self.global_list = global_list

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

        self.blank_screen = np.zeros((self.args.screen_height, self.args.screen_width))
        self.total_step = 0
        self.epoch_done = 0
        self.next_test_thread_no = 0
        self.train_start = time.strftime('%Y%m%d_%H%M%S')
        
        if os.path.exists('output') == False:
            os.makedirs('output')
        if os.path.exists('snapshot') == False:
            os.makedirs('snapshot')

        if self.play_file is None and self.thread_no == 0:
            log_file="output/%s_%s.log" % (args.game, self.train_start)            
            util.Logger(log_file)
            
        if os.path.exists(args.snapshot_folder) == False:
            os.makedirs(args.snapshot_folder)
        
        self.print_env()
        self.initialize_post()
        
    def initialize_post(self):
        """ initialization that should be run on __init__() or after deserialization """
        if self.args.show_screen and self.thread_no == 0:
            display_screen = True
        else:
            display_screen = False

        self.env = get_env(self.args, True, display_screen)
        self.legal_actions = self.env.get_actions()
        self.initialize_model()
        self.initialize_replay_memory()
        
        if self.thread_no == 0:
            self.debug_input = DebugInput(self)
            self.debug_input.start()
        else:
            self.debug_input = None
        
    def initialize_model(self):
        if self.args.backend == 'NEON':
            from network_model.model_neon import ModelRunnerNeon
            self.model_runner = ModelRunnerNeon(
                                    self.args, 
                                    max_action_no = len(self.legal_actions),
                                    batch_dimension = self.batch_dimension
                                    )
        elif self.args.backend == 'TF':
            if self.args.ddpg:
                self.model_runner = ModelRunnerTFDdpg(
                                self.args, 
                                action_group_no = self.env.action_group_no,
                                thread_no = self.thread_no
                                )
            elif self.args.drl == 'a3c':            
                self.model_runner = ModelRunnerTFA3C(
                                self.global_list,
                                self.args, 
                                max_action_no = len(self.legal_actions),
                                thread_no = self.thread_no
                                )
            elif self.args.drl == 'a3c_lstm':            
                self.model_runner = ModelRunnerTFA3CLstm(
                                self.global_list,
                                self.args, 
                                max_action_no = len(self.legal_actions),
                                thread_no = self.thread_no
                                )
            elif self.args.drl == '1q':
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
                                    thread_no = self.thread_no
                                    )
        else:
            raise ValueError('args.backend should be TF or NEON.')

    def initialize_replay_memory(self):
        if self.env.continuous_action:
            action_group_no = self.env.action_group_no
        else:
            action_group_no = 1
        uniform_replay_memory = ReplayMemory(self.args, self.env.state_dtype, self.env.continuous_action, action_group_no)
        if self.args.prioritized_replay == True:
            self.replay_memory = SamplingManager(self.args, uniform_replay_memory)
        else:
            self.replay_memory = uniform_replay_memory

    def set_global_list(self, global_list):
        self.global_list = global_list

    def get_greedy_epsilon(self, mode):
        if mode == 'TEST':
            greedy_epsilon = self.args.test_epsilon
        else:
            min_epsilon = self.args.train_min_epsilon
            if self.total_step < self.args.train_epsilon_start_step:
                return 1.0
            elif self.total_step <= self.args.train_epsilon_end_step:
                greedy_epsilon = ((self.args.train_min_epsilon - 1) * self.total_step + self.args.train_epsilon_end_step - self.args.train_epsilon_start_step *  self.args.train_min_epsilon) / (self.args.train_epsilon_end_step - self.args.train_epsilon_start_step)
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
        if self.env.continuous_action:
            return self.get_action_continuous(mode)
        else:        
            return self.get_action_discrete(mode)
        
    def get_action_discrete(self, mode):
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
            return action_index, 0

    def get_action_continuous(self, mode):
        global debug_print_step

        state = self.replay_memory.history_buffer
        action_values = self.model_runner.predict(state)

        greedy_epsilon = self.get_greedy_epsilon(mode)
        
        if mode == 'TRAIN':
            self.env.apply_action_noise(action_values, greedy_epsilon)
                    
        if debug_print_step and self.thread_no == 0:
            print 'greedy_epsilon: %.3f, action_values: %s' % (greedy_epsilon, action_values)

        return action_values, greedy_epsilon
                                                         
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
            print 'Start time: %s' % self.train_start
            print '[ Running Environment ]'
            for arg in sorted(vars(self.args)):
                print '{} : '.format(arg).ljust(30) + '{}'.format(getattr(self.args, arg))
                
    def print_weights(self):
        self.model_runner.print_weights()
        
    def reset_game(self):
        self.replay_memory.clear_history_buffer()
        self.env.reset_game()
        self.current_state = None
        action_index = 0
        
        if self.args.drl == 'a3c_lstm':
            self.model_runner.reset_lstm_state()

        if self.args.use_random_action_on_reset:
            for _ in range(random.randint(4, 30)):
                self.do_actions(action_index, 'TRAIN')

        first_state = self.resize_screen(self.env.getScreenGrayscale())
        for i in range(self.args.screen_history):
            if i < self.args.screen_history - 1:
                state = self.blank_screen
            else:
                state = first_state
            if self.args.minibatch_random == False:
                self.replay_memory.add(action_index, 0, state, False)
            else:
                self.replay_memory.add_to_history_buffer(state)
    
    def resize_screen(self, state):
        if len(state.shape) < 2:
            return state
        elif state.shape[0] == self.args.screen_height and state.shape[1] == self.args.screen_width:
            return state
        else:
            resized = cv2.resize(state, (self.args.screen_width, self.args.screen_height))
            return resized
        
    def do_actions(self, action_index, mode):
        global debug_display
        global debug_display_sleep
        
        if self.thread_no == 0:
            _debug_display = debug_display
            _debug_display_sleep = debug_display_sleep
        else:
            _debug_display = False
            _debug_display_sleep = 0
        
        if self.env.continuous_action:
            action = action_index
        else:
            action = self.legal_actions[action_index]
            
        reward = 0
        terminal = False 
        lives = self.env.lives()
        frame_repeat = self.args.frame_repeat

        if frame_repeat == 1 or self.args.use_env_frame_skip == True:
            reward += self.env.act(action)
            new_state = self.env.getScreenGrayscale(_debug_display, _debug_display_sleep)
            game_over = self.env.game_over()
            if (self.args.lost_life_terminal == True and self.env.lives() < lives) or game_over:
                terminal = True
                if mode == 'TRAIN' and self.args.lost_life_game_over == True:
                    game_over = True
        else:
            if self.current_state is None:
                self.current_state = self.env.getScreenGrayscale(_debug_display, _debug_display_sleep)                
            for _ in range(frame_repeat):
                prev_state = self.current_state
                reward += self.env.act(action)
                state = self.env.getScreenGrayscale(_debug_display, _debug_display_sleep)
                if state is not None:
                    self.current_state = state 
                game_over = self.env.game_over()
                if (self.args.lost_life_terminal == True and self.env.lives() < lives) or game_over:
                    terminal = True
                    if mode == 'TRAIN' and self.args.lost_life_game_over == True:
                        game_over = True
                    break
            new_state = np.maximum(prev_state, self.current_state)
        if new_state is None:
            new_state = self.blank_screen
        resized = self.resize_screen(new_state)
        return reward, resized, terminal, game_over
    
    def generate_replay_memory(self, count):
        global debug_quit
        
        if self.thread_no == 0:
            print 'Generating %s replay memory' % count
        start_time = time.time()
        self.reset_game()
        for _ in range(count):
            action_index, greedy_epsilon = self.get_action_index('TRAIN')
            reward, state, terminal, game_over = self.do_actions(action_index, 'TRAIN')
            self.replay_memory.add(action_index, reward, state, terminal)
                
            if game_over:
                self.reset_game()

            if debug_quit:
                return
        
        if self.thread_no == 0:
            print 'Generating replay memory took %.0f sec' % (time.time() - start_time)
        
    def check_pause(self):
        global debug_pause
        if debug_pause:
            while debug_pause:
                time.sleep(1.0)
        
    def test(self, epoch):
        global debug_print
        global debug_quit
        
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
            
                if debug_print:
                    print "[ Test  %s ] score: %.2f, avg score: %.2f ep: %d, elapsed: %.0fm. last e: %.3f" % \
                          (epoch, episode_reward, float(total_reward) / episode, episode, 
                           (time.time() - test_start_time) / 60,
                           greedy_epsilon)

                self.reset_game()
                episode_reward = 0
                
            self.check_pause()
            if debug_quit:
                return
        
        episode = max(episode, 1)          
        print "[ Test  %s ] avg score: %.2f elapsed: %.0fm. last e: %.3f" % \
              (epoch, float(total_reward) / episode, 
               (time.time() - test_start_time) / 60,
               greedy_epsilon)
                  
    def train(self, replay_memory_no=None):
        """
        train loop for 'dqn', 'double_dqn' and '1q'
        """
        global global_step_no
        global debug_print_step
        global debug_quit
        
        if replay_memory_no == None:
            replay_memory_no = self.args.train_start
        if replay_memory_no > 0:
            self.generate_replay_memory(replay_memory_no)
        
        max_global_step_no = self.args.max_epoch * self.args.epoch_step * self.args.thread_no
        
        if self.thread_no == 0:
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
                self.total_step += 1
                global_step_no += 1

                self.replay_memory.add(action_index, reward, state, terminal)
                    
                if step_no % self.args.train_step == 0:
                    minibatch = self.replay_memory.get_minibatch()
                    if self.args.drl == '1q':
                        learning_rate = self._anneal_learning_rate(max_global_step_no, global_step_no)
                    else:
                        learning_rate = self.args.learning_rate
                                        
                    self.model_runner.train(minibatch, self.replay_memory, learning_rate, debug_print)
                
                if self.total_step % self.args.save_step == 0 and self.thread_no == 0:
                    file_name = 'dqn_%s' % self.total_step
                    self.save(file_name)
                     
                if game_over:
                    if debug_print:
                        print_step = 1
                    else:
                        print_step = 500
                        
                    if episode % print_step == 0:
                        print "Ep %s, score: %.2f, step: %s, elapsed: %.1fs, avg: %.2f t_step:%s, t_elapsed: %.0fm" % (
                                                                                episode, episode_total_reward,
                                                                                step_no, (time.time() - episode_start_time),
                                                                                float(epoch_total_reward) / episode,
                                                                                self.total_step,
                                                                                (time.time() - start_time) / 60)
                    episode_start_time = time.time()
                    
                    episode += 1
                    episode_total_reward = 0
                    
                    self.reset_game()
                    
                if step_no > 0 and step_no % self.args.update_step == 0:
                    self.model_runner.update_model()
                
                self.check_pause()
                if debug_quit:
                    self.env.finish()
                    return
                
            print "[ Train %s ] avg score: %.2f elapsed: %.0fm. last e: %.3f, t_step:%s, t_elapsed: %.0fm" % \
                  (epoch, float(epoch_total_reward) / episode, 
                   (time.time() - epoch_start_time) / 60,
                   greedy_epsilon, self.total_step, (time.time() - start_time) / 60)
             
            # Test once every epoch
            if args.run_test == True:
                if args.asynchronousRL == False:
                    self.test(epoch)
                else:
                    if self.thread_no == self.next_test_thread_no:
                        self.test(epoch)
                    self.next_test_thread_no = (self.next_test_thread_no + 1) % self.args.thread_no
                    
            self.epoch_done = epoch
                        
        if self.thread_no == 0:
            file_name = 'dqn_%s' % self.total_step
            self.save(file_name)    

        if self.debug_input != None:
            self.debug_input.finish()
                
    def _anneal_learning_rate(self, max_global_step_no, global_step_no):
        learning_rate = self.args.learning_rate * (max_global_step_no - global_step_no) / max_global_step_no
        if learning_rate < 0.0:
            learning_rate = 0.0
        return learning_rate

    def train_async_a3c(self, replay_memory_no=None):
        """
        train loop for 'a3c' and 'a3c_lstm'
        """
        global global_step_no
        global debug_print
        global debug_pause
        global debug_quit

        max_global_step_no = self.args.max_epoch * self.args.epoch_step * self.args.thread_no
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
            
            while step_no <= self.args.epoch_step:
                v_pres = []
    
                if self.args.drl == 'a3c_lstm':
                    lstm_state_value = self.model_runner.get_lstm_state()            
                for i in range(self.args.train_step):
                    action_index, state_value = self.get_action_state_value('TRAIN')
                    reward, state, terminal, game_over = self.do_actions(action_index, 'TRAIN')

                    self.replay_memory.add(action_index, reward, state, terminal)
                    v_pres.append(state_value)
    
                    episode_total_reward += reward
                    epoch_total_reward += reward
                    self.total_step += 1

                    if terminal:
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

                if self.args.drl == 'a3c_lstm':
                    self.model_runner.train(prestates, v_pres, actions, rewards, terminals, v_post, learning_rate, lstm_state_value)
                else:
                    self.model_runner.train(prestates, v_pres, actions, rewards, terminals, v_post, learning_rate)

                if game_over:
                    if debug_print:
                        print_step = 1
                    else:
                        print_step = 500
                        
                    if episode % print_step == 0:
                        print "Ep %s, score: %.2f, step: %s, elapsed: %.1fs, avg: %.2f t_step:%s, t_elapsed: %.0fm" % (
                                                                                episode, episode_total_reward,
                                                                                step_no, (time.time() - episode_start_time),
                                                                                float(epoch_total_reward) / episode,
                                                                                self.total_step,
                                                                                (time.time() - start_time) / 60)
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

                self.check_pause()
                if debug_quit:
                    self.env.finish()
                    return
            
            self.epoch_done = epoch

            print "[ Train %s ] avg score: %.2f elapsed: %.0fm. rl: %.5f" % \
                  (epoch, float(epoch_total_reward) / episode, 
                   (time.time() - epoch_start_time) / 60, learning_rate)
                
            if self.thread_no == 0:
                file_name = 'a3c_%s' % global_step_no
                self.save(file_name)
            """
            elif global_step_no >= self.args.max_global_step_no:
                file_name = 'a3c_%s' % self.args.max_global_step_no
                self.save(file_name)
            """
             
            # Test once every epoch
            if args.run_test == True:
                if args.asynchronousRL == False:
                    self.test(epoch)
                else:
                    if self.thread_no == self.next_test_thread_no:
                        self.test(epoch)
                    self.next_test_thread_no = (self.next_test_thread_no + 1) % self.args.thread_no

        print 'thread %s finished' % self.thread_no

        if self.debug_input != None:
            self.debug_input.finish()        

    def save(self, file_name):
        timesnapshot_folder = self.args.snapshot_folder + '/' + self.train_start
        if os.path.exists(timesnapshot_folder) == False:
            os.makedirs(timesnapshot_folder)
        
        file_name = '%s/%s' % (timesnapshot_folder, file_name)
        with open(file_name + '.pickle', 'wb') as f:
            pickle.dump(self, f)
            self.model_runner.save(file_name + '.weight')
            #print '%s dumped' % file_name
        
    def __getstate__(self):
        self.replay_memory_no = self.replay_memory.count
        d = dict(self.__dict__)
        del d['env']
        del d['replay_memory']
        del d['model_runner']
        if 'global_list' in d:
            del d['global_list']
        if 'debug_input' in d:
            del d['debug_input']
        return d
        
class DebugInput(threading.Thread):
    def __init__(self, player):
        threading.Thread.__init__(self)
        self.player = player
        self.running = True
    
    def run(self):
        global debug_print
        global debug_print_step
        global debug_pause
        global debug_display
        global debug_display_sleep
        global debug_quit
        
        time.sleep(10)
        while (self.running):
            rlist, _, _ = select([sys.stdin], [], [], 1)
            if rlist:
                key_input = sys.stdin.readline().rstrip()
            else:
                continue
            
            if key_input == 'p':
                debug_print = not debug_print
                print 'Debug print : %s' % debug_print
            elif key_input == 'u':
                debug_pause = not debug_pause
                print 'Debug pause : %s' % debug_pause
            elif key_input == 'd' or key_input == 'dd':
                if debug_display == False:
                    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                    debug_display = True
                else:
                    debug_display = False
                    cv2.destroyAllWindows()                    
                print 'Debug display : %s' % debug_display
                if key_input == 'dd':
                    debug_print_step = not debug_print_step
                    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})                    
                    print 'Debug mode'
            elif key_input == '-':
                debug_display_sleep -= 20
                debug_display_sleep = max(1, debug_display_sleep)
                print 'Debug display_sleep : %s' % debug_display_sleep
            elif key_input == '+':
                debug_display_sleep += 20
                debug_display_sleep = min(5000, debug_display_sleep)
                print 'Debug display_sleep : %s' % debug_display_sleep
            elif key_input == 'e':
                self.player.print_env()
            elif key_input == 'w':
                self.player.print_weights()
            elif key_input == 'quit':
                print 'Quiting...'
                debug_quit = True
                debug_pause = False
                break
                
    def finish(self):
        self.running = False
    
debug_print = False
debug_print_step = False
debug_pause = False
debug_display = False
debug_display_sleep = 300
debug_quit = False
global_data = []
global_step_no = 0

if __name__ == '__main__':
    args = get_args()
    save_file = args.snapshot

    if args.asynchronousRL:
        threadList = []
        playerList = []

        env = get_env(args, False, False)
        legal_actions = env.get_actions(args.env)
        
        # initialize global settings
        if args.drl == 'a3c':
            model = ModelA3C(args, 'global', True,  len(legal_actions), thread_no = -1)
        elif args.drl == 'a3c_lstm':     
            model = ModelA3CLstm(args, 'global', True,  len(legal_actions), thread_no = -1)
        elif args.drl == '1q':     
            model = Model(args, 'global', True, len(legal_actions), thread_no = -1)

        global_list = model.prepare_global(args.rms_decay, args.rms_epsilon)
        global_sess = global_list[0]
        global_vars = global_list[1]

        if save_file is not None:        # retrain
            current_time = time.strftime('%Y%m%d_%H%M%S')
            log_file="output/%s_%s.log" % (args.game, current_time)            
            util.Logger(log_file)
            print 'Resume trainig: %s' % save_file

            for i in range(args.thread_no):        
                with open(save_file + '.pickle') as f:
                    player = pickle.load(f)
                    player.train_start = current_time
                    player.thread_no = i
                    if i == 0:
                        player.print_env()
                    player.set_global_list(global_list)
                    player.initialize_post()
                    playerList.append(player)                    

            model.init_global(global_sess)
            
            global_step_no = playerList[0].epoch_done * 4000000

            """
            import tensorflow as tf
            writer = tf.train.SummaryWriter("/tmp/tf_graph", global_list[0].graph_def)
            writer.close()
            print 'tf_graph is written'
            """
            
            # Load global variables
            load_global_vars(global_sess, global_vars, save_file + '.weight')
            
            # copy global variables to local variables
            for i in range(args.thread_no):        
                playerList[i].model_runner.copy_from_global_to_local()
        else:
            for i in range(args.thread_no):        
                print 'creating a thread[%s]' % i
                player = Trainer(args, thread_no= i, global_list=global_list)
                playerList.append(player)

            model.init_global(global_sess)
        
        for player in playerList:
            if args.drl.startswith('a3c'):
                target_func = player.train_async_a3c
            else:
                target_func = player.train
            t = threading.Thread(target=target_func, args=())
            t.start()
            threadList.append(t)
        
        for thread in threadList:
            thread.join()
    else:
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
                player.train()
        else:
            player = Trainer(args)
            player.total_step = 0
            player.train()
            
