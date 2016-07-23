#!/usr/bin/env python

# ale_python_test_pygame_player.py
# Author: Ben Goodrich
#
# This modified ale_python_test_pygame.py to provide a fully interactive experience allowing the player
# to play. RAM Contents, current action, and reward are also displayed.
# keys are:
# arrow keys -> up/down/left/right
# z -> fire button
import sys
from ale_python_interface import ALEInterface
import numpy as np
import pygame
import matplotlib.pyplot as plt
from scipy.misc import imresize

key_action_tform_table = (
0, #00000 none
2, #00001 up
5, #00010 down
2, #00011 up/down (invalid)
4, #00100 left
7, #00101 up/left
9, #00110 down/left
7, #00111 up/down/left (invalid)
3, #01000 right
6, #01001 up/right
8, #01010 down/right
6, #01011 up/down/right (invalid)
3, #01100 left/right (invalid)
6, #01101 left/right/up (invalid)
8, #01110 left/right/down (invalid)
6, #01111 up/down/left/right (invalid)
1, #10000 fire
10, #10001 fire up
13, #10010 fire down
10, #10011 fire up/down (invalid)
12, #10100 fire left
15, #10101 fire up/left
17, #10110 fire down/left
15, #10111 fire up/down/left (invalid)
11, #11000 fire right
14, #11001 fire up/right
16, #11010 fire down/right
14, #11011 fire up/down/right (invalid)
11, #11100 fire left/right (invalid)
14, #11101 fire left/right/up (invalid)
16, #11110 fire left/right/down (invalid)
14  #11111 fire up/down/left/right (invalid)
)

class DeepRLPlayer:
    def __init__(self, showScreen):
        self.showScreen = showScreen
        self.grayPixels = np.zeros((84, 84), np.float)
        
        self.rgb_palette = [
          0x000000, 0, 0x4a4a4a, 0, 0x6f6f6f, 0, 0x8e8e8e, 0,
          0xaaaaaa, 0, 0xc0c0c0, 0, 0xd6d6d6, 0, 0xececec, 0,
          0x484800, 0, 0x69690f, 0, 0x86861d, 0, 0xa2a22a, 0,
          0xbbbb35, 0, 0xd2d240, 0, 0xe8e84a, 0, 0xfcfc54, 0,
          0x7c2c00, 0, 0x904811, 0, 0xa26221, 0, 0xb47a30, 0,
          0xc3903d, 0, 0xd2a44a, 0, 0xdfb755, 0, 0xecc860, 0,
          0x901c00, 0, 0xa33915, 0, 0xb55328, 0, 0xc66c3a, 0,
          0xd5824a, 0, 0xe39759, 0, 0xf0aa67, 0, 0xfcbc74, 0,
          0x940000, 0, 0xa71a1a, 0, 0xb83232, 0, 0xc84848, 0,
          0xd65c5c, 0, 0xe46f6f, 0, 0xf08080, 0, 0xfc9090, 0,
          0x840064, 0, 0x97197a, 0, 0xa8308f, 0, 0xb846a2, 0,
          0xc659b3, 0, 0xd46cc3, 0, 0xe07cd2, 0, 0xec8ce0, 0,
          0x500084, 0, 0x68199a, 0, 0x7d30ad, 0, 0x9246c0, 0,
          0xa459d0, 0, 0xb56ce0, 0, 0xc57cee, 0, 0xd48cfc, 0,
          0x140090, 0, 0x331aa3, 0, 0x4e32b5, 0, 0x6848c6, 0,
          0x7f5cd5, 0, 0x956fe3, 0, 0xa980f0, 0, 0xbc90fc, 0,
          0x000094, 0, 0x181aa7, 0, 0x2d32b8, 0, 0x4248c8, 0,
          0x545cd6, 0, 0x656fe4, 0, 0x7580f0, 0, 0x8490fc, 0,
          0x001c88, 0, 0x183b9d, 0, 0x2d57b0, 0, 0x4272c2, 0,
          0x548ad2, 0, 0x65a0e1, 0, 0x75b5ef, 0, 0x84c8fc, 0,
          0x003064, 0, 0x185080, 0, 0x2d6d98, 0, 0x4288b0, 0,
          0x54a0c5, 0, 0x65b7d9, 0, 0x75cceb, 0, 0x84e0fc, 0,
          0x004030, 0, 0x18624e, 0, 0x2d8169, 0, 0x429e82, 0,
          0x54b899, 0, 0x65d1ae, 0, 0x75e7c2, 0, 0x84fcd4, 0,
          0x004400, 0, 0x1a661a, 0, 0x328432, 0, 0x48a048, 0,
          0x5cba5c, 0, 0x6fd26f, 0, 0x80e880, 0, 0x90fc90, 0,
          0x143c00, 0, 0x355f18, 0, 0x527e2d, 0, 0x6e9c42, 0,
          0x87b754, 0, 0x9ed065, 0, 0xb4e775, 0, 0xc8fc84, 0,
          0x303800, 0, 0x505916, 0, 0x6d762b, 0, 0x88923e, 0,
          0xa0ab4f, 0, 0xb7c25f, 0, 0xccd86e, 0, 0xe0ec7c, 0,
          0x482c00, 0, 0x694d14, 0, 0x866a26, 0, 0xa28638, 0,
          0xbb9f47, 0, 0xd2b656, 0, 0xe8cc63, 0, 0xfce070, 0
        ]
        

    
    def getActionFromKeys(self):
        #get the keys
        keys = 0
        pressed = pygame.key.get_pressed()
        keys |= pressed[pygame.K_UP]
        keys |= pressed[pygame.K_DOWN]  <<1
        keys |= pressed[pygame.K_LEFT]  <<2
        keys |= pressed[pygame.K_RIGHT] <<3
        keys |= pressed[pygame.K_z] <<4
        a = key_action_tform_table[keys]
        return a, pressed

    def displayInfo(self, screen, ram, a, total_reward):
            font = pygame.font.SysFont("Ubuntu Mono",32)
            text = font.render("RAM: " ,1,(255,208,208))
            screen.blit(text,(330,10))
        
            font = pygame.font.SysFont("Ubuntu Mono",25)
            height = font.get_height()*1.2
        
            line_pos = 40
            ram_pos = 0
            while(ram_pos < 128):
                ram_string = ''.join(["%02X "%ram[x] for x in range(ram_pos,min(ram_pos+16,128))])
                text = font.render(ram_string,1,(255,255,255))
                screen.blit(text,(340,line_pos))
                line_pos += height
                ram_pos +=16
                
            #display current action
            font = pygame.font.SysFont("Ubuntu Mono",32)
            text = font.render("Current Action: " + str(a) ,1,(208,208,255))
            height = font.get_height()*1.2
            screen.blit(text,(330,line_pos))
            line_pos += height
        
            #display reward
            font = pygame.font.SysFont("Ubuntu Mono",30)
            text = font.render("Total Reward: " + str(total_reward) ,1,(208,255,255))
            screen.blit(text,(330,line_pos))

    def display(self, rgb, gray=False):
        if (gray):
            plt.imshow(rgb, cmap='gray')
        else:
            plt.imshow(rgb)
        plt.show()

    def getScreenPixels(self, ale, screen, game_surface, step):
        
        numpy_surface = np.frombuffer(game_surface.get_buffer(),dtype=np.int32)
        ale.getScreenRGB(numpy_surface)
        
        '''
        s1 = np.zeros((33600,), dtype=np.int32)
        ale.getScreenRGB(s1)
        data = s1.view(np.uint8).reshape(s1.shape + (4,))
        rgba = np.reshape(data, (self.screen_height, self.screen_width, 4))
        rgb = rgba[:, :, (2, 1, 0)]        
        cropped = rgb.astype(np.float) / 256
        #self.display(cropped)
        
        
        s2 = np.zeros((210*160,), dtype=np.uint8)
        ale.getScreen(s2)
        rgba2 = np.reshape(s2, (self.screen_height, self.screen_width))
        resized = imresize(rgba2, (110, 84, 3))
        cropped = resized[26:110, :].astype(np.float) / 256        
        
        newRgba2 = np.zeros((210, 160, 3), dtype=np.uint32)
        for y in range(210):
            for x in range(160):
                #data = rgba2[y, x]
                data = rgba2[y, x] & ~0x1
                a = self.rgb_palette[data]
                
                if a != 0:
                    r = (a & 0xFF0000) >> 16;
                    g = (a & 0x00FF00) >> 8;
                    b = a & 0x0000FF; 
                    newRgba2[y, x, 0] = r
                    newRgba2[y, x, 1] = g
                    newRgba2[y, x, 2] = b
        '''
        
        #del numpy_surface
        

        if (self.showScreen):
            screen.fill((0,0,0))
            screen.blit(pygame.transform.scale2x(game_surface),(0,0))
            #screen.blit(game_surface.copy(),(0,0))

        data = numpy_surface.view(np.uint8).reshape(numpy_surface.shape + (4,))
        rgba = np.reshape(data, (self.screen_height, self.screen_width, 4))
        rgb = rgba[:, :, (2, 1, 0)]
        resized = imresize(rgb, (110, 84, 3))
        cropped = resized[26:110, :, :].astype(np.float) / 256
        
        #if step > 200:        
        #    self.display(cropped)
        
        self.grayPixels = np.dot(cropped[...,:3], [0.299, 0.587, 0.114])        
        #self.display(self.grayPixels, gray=True)

        return self.grayPixels
            
    def checkExit(self, pressed): 
        #process pygame event queue
        exit=False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit=True
                break;
        if(pressed[pygame.K_q]):
            exit = True
        return exit
        
    def gogo(self):
        ale = ALEInterface()
        
        max_frames_per_episode = ale.getInt("max_num_frames_per_episode");
        ale.set("random_seed",123)
        
        random_seed = ale.getInt("random_seed")
        print("random_seed: " + str(random_seed))
        
        ale.loadROM(sys.argv[1])
        legal_actions = ale.getMinimalActionSet()
        print legal_actions
        
        (self.screen_width,self.screen_height) = ale.getScreenDims()
        print("width/height: " +str(self.screen_width) + "/" + str(self.screen_height))
        
        (display_width,display_height) = (1024,420)
        
        pygame.init()
        game_surface = pygame.Surface((self.screen_width,self.screen_height))

        if self.showScreen:
            screen = pygame.display.set_mode((display_width,display_height))
            pygame.display.set_caption("Arcade Learning Environment Player Agent Display")
            
            pygame.display.flip()
        else:
            screen = None
        
        clock = pygame.time.Clock()
        
        episode = 0
        total_reward = 0.0 
        step = 0
        while(episode < 10):
            if self.showScreen:
                a, pressed = self.getActionFromKeys()
            else:
                a = 1

            reward = ale.act(a);
            total_reward += reward
        
            pixels = self.getScreenPixels(ale, screen, game_surface, step)
        
            ram_size = ale.getRAMSize()
            ram = np.zeros((ram_size),dtype=np.uint8)
            ale.getRAM(ram)
            
            if self.showScreen:
                self.displayInfo(screen, ram, a, total_reward)
                pygame.display.flip()
        
                if(self.checkExit(pressed)):
                    break
        
            step +=1 
            
            #delay to 60fps
            #clock.tick(60.)
        
            if(ale.game_over()):
                episode_frame_number = ale.getEpisodeFrameNumber()
                frame_number = ale.getFrameNumber()
                print("Frame Number: " + str(frame_number) + " Episode Frame Number: " + str(episode_frame_number))
                print("Episode " + str(episode) + " ended with score: " + str(total_reward))
                ale.reset_game()
                total_reward = 0.0 
                episode = episode + 1
        
    
if __name__ == '__main__':    
    if(len(sys.argv) < 2):
        print("Usage ./ale_python_test_pygame_player.py <ROM_FILE_NAME>")
        sys.exit()
    
    showScreen = True
    #showScreen = False
    
    DeepRLPlayer(showScreen).gogo()
