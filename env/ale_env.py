import random
from ale_python_interface import ALEInterface

class AleEnv():
    def __init__(self):
        self.actions = None
        
    def initialize(self, rom, display_screen=False, use_ale_frame_skip=False, frame_repeat=0):
        self.ale = ALEInterface()
        self.ale.setInt("random_seed", random.randint(1, 1000))
        if display_screen:
            self.ale.setBool('display_screen', True)

        if use_ale_frame_skip == True:
            self.ale.setInt('frame_skip', frame_repeat)
            self.ale.setBool('color_averaging', True)        
 
        self.ale.setFloat('repeat_action_probability', 0)
        self.ale.loadROM(rom)
        self.actions = self.get_actions(rom)
        print 'actions: %s' % self.actions
        
        (self.screen_width,self.screen_height) = self.ale.getScreenDims()
        print("width/height: " +str(self.screen_width) + "/" + str(self.screen_height))
        
    def get_actions(self, rom=None):
        if self.actions == None and rom != None:
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
    
    def getScreenGrayscale(self):
        return self.ale.getScreenGrayscale()
    
    def act(self, action):
        return self.ale.act(action)
    
    def game_over(self):
        return self.ale.game_over()
    
    
    