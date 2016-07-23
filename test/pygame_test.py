import pygame
import os
import time

pygame.init()
screen = pygame.display.set_mode((720, 480))
pygame.display.set_caption('My game')

background = pygame.Surface(screen.get_size())
"""
background = background.convert()
background.fill((250, 250, 250))

charRect = pygame.Rect((0,0),(100, 100))
charImage = pygame.image.load(os.path.abspath("/media/big/download/ic_launcher.png"))
charImage = pygame.transform.scale(charImage, charRect.size)
charImage = charImage.convert()

background.blit(charImage, charRect) #This just makes it in the same location
                                     #and prints it the same size as the image
"""
screen.blit(background,(0,0))
pygame.display.flip()

time.sleep(2)