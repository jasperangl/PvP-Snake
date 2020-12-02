import pickle
import random

import numpy as np
import pygame

from main import SCREEN_WIDTH, SCREEN_HEIGHT, drawGrid, FOOD_REWARD, Food, Snake, handleSnakeCollisions, \
    handleFoodEating, SIZE

'''31000 episode is best'''
'''20000 is decent too'''
Q_TABLE = 'q_tables/' + 'qtable-100000ep-wEnemy-2D.p'

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
    enemy = Snake(2)
    enemy.positions = [(random.randint(0, SIZE - 1), random.randint(0, SIZE - 1))]
    food = Food([player])

    myfont = pygame.font.SysFont("bahnschrift", 20)
    iterations = 0
    while (player.lives > 0):
        clock.tick(3)
        drawGrid(surface)
        enemy.handle_keys()
        enemy.move()
        obs = (player - food, player - enemy)
        # no random action in display phase
        action_space = np.array(q_table[obs]).copy()
        print("Position:", player.get_head_position())
        print(action_space)
        player.action(action_space, "QL")

        handleSnakeCollisions(player,enemy)
        handleFoodEating(player,enemy, food)
        player.draw(surface)
        enemy.draw(surface)
        food.draw(surface)
        screen.blit(surface, (0, 0))
        text1 = myfont.render("Score Player {0}".format(player.score), 1, (250, 250, 250))
        #text2 = myfont.render("Score AI {0}".format(enemy.score), 1, (250, 250, 250))
        screen.blit(text1, (5, 10))
        #screen.blit(text2, (SCREEN_WIDTH - 120, 10))
        pygame.display.update()
    print("Snake Player Final Score:", player.score)
    #print("Snake AI Final Score:", enemy.score)

display()