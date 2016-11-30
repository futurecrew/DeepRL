import numpy as np
import cv2
import itertools as it
from vizdoom import *

class VizDoomEnv():
    def __init__(self, config, display_screen, use_env_frame_skip, frame_repeat):
        if config == None:
            print 'Need to set vizdoom --config'
            exit()
        self.actions = None
        self.config = config
        self.display_screen = display_screen
        if use_env_frame_skip == True:
            self.frame_repeat = frame_repeat
        else:
            self.frame_repeat = 1

    def initialize(self):
        self.game = DoomGame()
        self.game.set_window_visible(self.display_screen)        
        
        #self.config = "ViZDoom-master/examples/config/simpler_basic.cfg"
        self.game.load_config(self.config)
        self.game.set_mode(Mode.PLAYER)
        self.game.set_screen_format(ScreenFormat.GRAY8)
        #self.game.set_screen_resolution(ScreenResolution.RES_640X480)
        self.game.set_screen_resolution(ScreenResolution.RES_160X120)
        self.game.init()
        
        self.available_buttons_size = self.game.get_available_buttons_size()
        
        self.actions = [list(a) for a in it.product([0, 1], repeat=self.available_buttons_size)]
        print 'actions: %s' % self.actions
        
    def get_actions(self, rom=None):
        if self.actions is None:
            game = DoomGame()
            game.set_window_visible(False)        
            game.load_config(self.config)
            game.set_screen_resolution(ScreenResolution.RES_160X120)
            game.init()
            self.actions = [list(a) for a in it.product([0, 1], repeat=game.get_available_buttons_size())]
        return self.actions
        
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
        return self.game.make_action(action, self.frame_repeat)
    
    def game_over(self):
        return self.game.is_episode_finished()
        
