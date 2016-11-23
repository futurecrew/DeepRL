import numpy as np
import cv2
from vizdoom import *

class VizDoomEnv():
    def initialize(self, config_file_path, display_screen=False):
        self.game = DoomGame()
        self.game.set_window_visible(display_screen)        

        if config_file_path == None:
            print 'Need to set vizdoom --config-file-path'
            exit()
        
        #config_file_path = "ViZDoom-master/examples/config/simpler_basic.cfg"
        self.game.load_config(config_file_path)
        self.game.set_mode(Mode.PLAYER)
        self.game.set_screen_format(ScreenFormat.GRAY8)
        #self.game.set_screen_resolution(ScreenResolution.RES_640X480)
        self.game.set_screen_resolution(ScreenResolution.RES_160X120)
        self.game.init()
        
        self.actions = self.get_actions()
        print 'actions: %s' % self.actions
        
    def get_actions(self, rom=None):
        actions = [[False, False, False], [True, False, False], [False, True, False], [False, False, True]]
        return actions
        
    def reset_game(self):
        self.game.new_episode()
        
    def lives(self):
        return 1 if self.game.is_player_dead() == False else 0
    
    def getScreenRGB(self):
        return self.game.get_state().screen_buffer
    
    def getScreenGrayscale(self, debug_display=False, debug_display_sleep=0):
        screen = self.game.get_state().screen_buffer
        if screen is not None and debug_display:
            cv2.imshow('image', screen)
            cv2.waitKey(debug_display_sleep)
        return screen
    
    def act(self, action):
        return self.game.make_action(action)
    
    def game_over(self):
        return self.game.is_episode_finished()
        
