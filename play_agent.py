import pickle

import numpy as np
import pygame

from main import SCREEN_WIDTH, SCREEN_HEIGHT, drawGrid, ENEMY_PENALTY, FOOD_REWARD, Food, Snake

'''20000 episode is actually decent'''
Q_TABLE = 'qtable-31000ep.p'

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
    #enemy = Snake(1)
    food = Food([player])

    myfont = pygame.font.SysFont("bahnschrift", 20)
    iterations = 0
    while (player.lives > 0):
        iterations += 1
        if iterations >= 100:
            player.reset()
            iterations = 0
        clock.tick(6)
        drawGrid(surface)
        obs = (player - food)
        # no random action in display phase
        action_space = np.array(q_table[obs]).copy()
        print("Position:", player.get_head_position())
        print(action_space)
        player.action(action_space, "QL")

        # if player.get_head_position() == enemy.get_head_position():
        #     player.reward = -ENEMY_PENALTY
        #     player.score = -ENEMY_PENALTY
        #     player.reset()
        if player.get_head_position() == food.position:
            player.length += 1
            player.score += FOOD_REWARD
            player.reward += FOOD_REWARD
            food.randomize_position([player])
        # if enemy.get_head_position() == food.position:
        #     enemy.length += 1
        #     enemy.score += FOOD_REWARD
        #     food.randomize_position([player, enemy])
        player.draw(surface)
        #enemy.draw(surface)
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