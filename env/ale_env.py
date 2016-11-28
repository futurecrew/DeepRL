import random
from ale_python_interface import ALEInterface

class AleEnv():
    def __init__(self, rom, display_screen, use_env_frame_skip, frame_repeat):
        self.actions = None
        self.rom = rom
        self.display_screen = display_screen
        self.use_env_frame_skip = use_env_frame_skip
        self.frame_repeat = frame_repeat
        
    def initialize(self):
        self.ale = ALEInterface()
        self.ale.setInt("random_seed", random.randint(1, 1000))
        if self.display_screen:
            self.ale.setBool('display_screen', True)

        if self.use_env_frame_skip == True:
            self.ale.setInt('frame_skip', self.frame_repeat)
            self.ale.setBool('color_averaging', True)        
 
        self.ale.setFloat('repeat_action_probability', 0)
        self.ale.loadROM(self.rom)
        self.actions = self.ale.getMinimalActionSet()
        print 'actions: %s' % self.actions
        (self.screen_width,self.screen_height) = self.ale.getScreenDims()
        print("width/height: " +str(self.screen_width) + "/" + str(self.screen_height))
        
        self.initialized = True
        
    def get_actions(self, rom=None):
        if self.actions is None and rom != None:
            ale = ALEInterface()
            ale.loadROM(rom)
            self.actions = ale.getMinimalActionSet()
        return self.actions
        
    def reset_game(self):
        self.ale.reset_game()
        
    def lives(self):
        return self.ale.lives()
    
    def getScreenRGB(self):
        return self.ale.getScreenRGB()
    
    def getScreenGrayscale(self, debug_display=False, debug_display_sleep=0):
        return self.ale.getScreenGrayscale()
    
    def act(self, action):
        return self.ale.act(action)
    
    def game_over(self):
        return self.ale.game_over()
    
    
    