import pickle
import random

import numpy as np
import pygame
from pygame import event
import matplotlib.pyplot as plt

from main import SCREEN_WIDTH, SCREEN_HEIGHT, drawGrid, FOOD_REWARD, Food, Snake, handleSnakeCollisions, \
    handleFoodEating, SIZE
from p_gradient.Lunar_Lander import getObsGrid
from p_gradient.utils import getObsSmall, get_food_distance

'''31000 episode is best'''
'''20000 is decent too'''
Q_TABLE = 'q_tables/' + 'qtable-31000ep.p'

with open(Q_TABLE, "rb") as f:
    q_table = pickle.load(f)

def display():
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)

    surface = pygame.Surface(screen.get_size())
    surface = surface.convert()
    drawGrid(surface)

    player = Snake(0)
    player.lives = 10
    enemy = Snake(1)
    enemy.positions = [(random.randint(0, SIZE - 1), random.randint(0, SIZE - 1))]
    food = Food([player])

    myfont = pygame.font.SysFont("bahnschrift", 20)
    iterations = 0
    while (player.lives > 0):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    print("SPace")
                    clock.tick(0)
        #print(get_food_distance(player, food))
        #getObsSmall([player, enemy], food)
        clock.tick(1)
        drawGrid(surface)
        enemy.handle_keys()
        obs = (player - food)
        # no random action in display phase
        action_space = np.array(q_table[obs]).copy()

        player.action(action_space, "QL")
        enemy.move()
        handleSnakeCollisions(player,enemy)
        handleFoodEating([player,enemy], food)
        getObsGrid([player,enemy], food, size=7, fullGrid=True)

        player.draw(surface)
        enemy.draw(surface)
        food.draw(surface)
        screen.blit(surface, (0, 0))
        text1 = myfont.render("Score AI {0}".format(player.score), 1, (250, 250, 250))
        text2 = myfont.render("Score Player {0}".format(enemy.score), 1, (250, 250, 250))
        screen.blit(text1, (5, 10))
        screen.blit(text2, (SCREEN_WIDTH - 175, 10))
        pygame.display.update()
    print("Snake Player Final Score:", player.score)
    #print("Snake AI Final Score:", enemy.score)

display()