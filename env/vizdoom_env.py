import numpy as np
import cv2
from vizdoom import *

class VizDoomEnv():
    def initialize(self, display_screen=False):
        self.game = DoomGame()
        self.game.set_window_visible(display_screen)        

        vizdoom_path = '/media/big/download/ViZDoom-master/'
        self.game.set_vizdoom_path(vizdoom_path + "/bin/vizdoom")
        
        # Sets path to iwad resource file which contains the actual doom game. Default is "./doom2.wad".
        self.game.set_doom_game_path(vizdoom_path + "/scenarios/freedoom2.wad")
        # game.set_doom_game_path("../../scenarios/doom2.wad")  # Not provided with environment due to licences.
        
        # Sets path to additional resources wad file which is basically your scenario wad.
        # If not specified default maps will be used and it's pretty much useless... unless you want to play good old Doom.
        self.game.set_doom_scenario_path(vizdoom_path + "/scenarios/basic.wad")
        
        # Sets map to start (scenario .wad files can contain many maps).
        self.game.set_doom_map("map01")
        
        # Sets resolution. Default is 320X240
        #self.game.set_screen_resolution(ScreenResolution.RES_640X480)
        self.game.set_screen_resolution(ScreenResolution.RES_160X120)
        
        # Sets the screen buffer format. Not used here but now you can change it. Defalut is CRCGCB.
        #self.game.set_screen_format(ScreenFormat.RGB24)
        self.game.set_screen_format(ScreenFormat.GRAY8)
        
        # Enables depth buffer.
        #self.game.set_depth_buffer_enabled(True)
        
        # Enables labeling of in game objects labeling.
        self.game.set_labels_buffer_enabled(True)
        
        # Enables buffer with top down map of the current episode/level.
        self.game.set_automap_buffer_enabled(True)
        
        # Sets other rendering options
        self.game.set_render_hud(False)
        self.game.set_render_minimal_hud(False) # If hud is enabled
        self.game.set_render_crosshair(False)
        self.game.set_render_weapon(True)
        self.game.set_render_decals(False)
        self.game.set_render_particles(False)
        self.game.set_render_effects_sprites(False)
        
        # Adds buttons that will be allowed. 
        self.game.add_available_button(Button.MOVE_LEFT)
        self.game.add_available_button(Button.MOVE_RIGHT)
        self.game.add_available_button(Button.ATTACK)
        
        # Adds game variables that will be included in state.
        self.game.add_available_game_variable(GameVariable.AMMO2)
        
        # Causes episodes to finish after 200 tics (actions)
        self.game.set_episode_timeout(300)
        
        # Makes episodes start after 10 tics (~after raising the weapon)
        self.game.set_episode_start_time(10)
        
        # Turns on the sound. (turned off by default)
        #self.game.set_sound_enabled(True)
        
        # Sets the livin reward (for each move) to -1
        self.game.set_living_reward(-1)
        
        # Sets ViZDoom mode (PLAYER, ASYNC_PLAYER, SPECTATOR, ASYNC_SPECTATOR, PLAYER mode is default)
        self.game.set_mode(Mode.PLAYER)
        
        # Initialize the game. Further configuration won't take any effect from now on.
        #self.game.set_console_enabled(True)
        self.game.init()
        
        # Define some actions. Each list entry corresponds to declared buttons:
        # MOVE_LEFT, MOVE_RIGHT, ATTACK
        # 5 more combinations are naturally possible but only 3 are included for transparency when watching.
        
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
    
    def getScreenGrayscale(self, debug_screen, debug_screen_sleep):
        screen = self.game.get_state().screen_buffer
        if screen is not None and debug_screen:
            #print 'getScreenGrayscale() 1'
            cv2.imshow('image', screen)
            #print 'getScreenGrayscale() 2'
            cv2.waitKey(debug_screen_sleep)
            #print 'getScreenGrayscale() 3'
        return screen
    
    def act(self, action):
        return self.game.make_action(action)
    
    def game_over(self):
        return self.game.is_episode_finished()
        
        
"""
import numpy as np
from vizdoom import *
import cv2

class VizDoomEnv():
    def initialize(self, display_screen=False):
        self.game = DoomGame()
        self.game.set_window_visible(display_screen)        

        vizdoom_path = '/media/big/download/ViZDoom-master/'
        self.game.set_vizdoom_path(vizdoom_path + "/bin/vizdoom")
        
        # Sets path to iwad resource file which contains the actual doom game. Default is "./doom2.wad".
        self.game.set_doom_game_path(vizdoom_path + "/scenarios/freedoom2.wad")
        # game.set_doom_game_path("../../scenarios/doom2.wad")  # Not provided with environment due to licences.
        
        # Sets path to additional resources wad file which is basically your scenario wad.
        # If not specified default maps will be used and it's pretty much useless... unless you want to play good old Doom.
        self.game.set_doom_scenario_path(vizdoom_path + "/scenarios/basic.wad")
        
        # Sets map to start (scenario .wad files can contain many maps).
        self.game.set_doom_map("map01")
        
        # Sets resolution. Default is 320X240
        #self.game.set_screen_resolution(ScreenResolution.RES_640X480)
        self.game.set_screen_resolution(ScreenResolution.RES_160X120)
        
        # Sets the screen buffer format. Not used here but now you can change it. Defalut is CRCGCB.
        self.game.set_screen_format(ScreenFormat.RGB24)
        
        # Enables depth buffer.
        #self.game.set_depth_buffer_enabled(True)
        
        # Enables labeling of in game objects labeling.
        self.game.set_labels_buffer_enabled(True)
        
        # Enables buffer with top down map of the current episode/level.
        self.game.set_automap_buffer_enabled(True)
        
        # Sets other rendering options
        self.game.set_render_hud(False)
        self.game.set_render_minimal_hud(False) # If hud is enabled
        self.game.set_render_crosshair(False)
        self.game.set_render_weapon(True)
        self.game.set_render_decals(False)
        self.game.set_render_particles(False)
        self.game.set_render_effects_sprites(False)
        
        # Adds buttons that will be allowed. 
        self.game.add_available_button(Button.MOVE_LEFT)
        self.game.add_available_button(Button.MOVE_RIGHT)
        self.game.add_available_button(Button.ATTACK)
        
        # Adds game variables that will be included in state.
        self.game.add_available_game_variable(GameVariable.AMMO2)
        
        # Causes episodes to finish after 200 tics (actions)
        self.game.set_episode_timeout(300)
        
        # Makes episodes start after 10 tics (~after raising the weapon)
        self.game.set_episode_start_time(10)
        
        # Turns on the sound. (turned off by default)
        #self.game.set_sound_enabled(True)
        
        # Sets the livin reward (for each move) to -1
        self.game.set_living_reward(-1)
        
        # Sets ViZDoom mode (PLAYER, ASYNC_PLAYER, SPECTATOR, ASYNC_SPECTATOR, PLAYER mode is default)
        self.game.set_mode(Mode.PLAYER)
        
        # Initialize the game. Further configuration won't take any effect from now on.
        #self.game.set_console_enabled(True)
        self.game.init()
        
        # Define some actions. Each list entry corresponds to declared buttons:
        # MOVE_LEFT, MOVE_RIGHT, ATTACK
        # 5 more combinations are naturally possible but only 3 are included for transparency when watching.
        
        self.actions = self.get_actions()
        print 'actions: %s' % self.actions

        # DJDJ        
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        
        
    def get_actions(self, rom=None):
        actions = [[False, False, False], [True, False, False], [False, True, False], [False, False, True]]
        return actions
        
    def reset_game(self):
        print 'reset_game()'
        self.game.new_episode()
        
    def lives(self):
        lives =1 if self.game.is_player_dead() == False else 0
        if lives != 1:
            print 'lives : %s' % lives
        return lives
    
    def getScreenRGB(self):
        return self.game.get_state().screen_buffer
    
    def getScreenGrayscale(self):
        rgb = self.getScreenRGB()
        if rgb is not None:
            gray = np.dot(rgb, np.array([.299, .587, .114])).astype(np.uint8)
        
            cv2.imshow('image',gray)
            k = cv2.waitKey(500)
            
            return gray
        else:
            return None
    
    def act(self, action):
        #import time
        #time.sleep(0.2)
        if action[0] == True:
            print 'act() : left'
        elif action[1] == True:
            print 'act() : RIGHT'
        elif action[2] == True:
            print 'act() : *****'
        else:
            print 'act() : nothing'
        return self.game.make_action(action)
    
    def game_over(self):
        game_over = self.game.is_episode_finished()
        if game_over:
            print 'game_over() true'
        return game_over
"""        

