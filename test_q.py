import pickle
import random

import numpy as np
import pygame
from pygame import event
import matplotlib.pyplot as plt

from main import SCREEN_WIDTH, SCREEN_HEIGHT, drawGrid, FOOD_REWARD, Food, Snake, handleSnakeCollisions, \
    handleFoodEating, SIZE
from p_gradient.pg_main import getObsGrid
from p_gradient.utils import getObsSmall, get_food_distance, enemyMovement
from qlearning import displayRL

'''31000 episode is best'''
'''20000 is decent too'''
Q_TABLE = 'q_tables/' + 'qtable-38000ep-2D.p'
Q_TABLE_NAME = "QL-Experiment-38000-WE"
MAX_STEPS = 200
SHOW_EVERY = 25
with open(Q_TABLE, "rb") as f:
    q_table = pickle.load(f)


### Displays one run of the Q_Learning agent
def displayQL():
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)

    surface = pygame.Surface(screen.get_size())
    surface = surface.convert()


    player = Snake(0)
    enemy = Snake(1)
    food = Food([player, enemy])

    myfont = pygame.font.SysFont("bahnschrift", 20)
    max_iter = 0
    while player.lives > 0:
        clock.tick(10)
        drawGrid(surface)
        obs = (player-food, player.get_head_position())
        #obs = (player - food)
        #enemy.positions = [(random.randint(0, SIZE - 1), random.randint(0, SIZE - 1))]
        action = np.argmax(q_table[obs])
        action_space = [action]
        player.action(action_space, "QL")
        enemyMovement(food=food, enemy=enemy)
        handleSnakeCollisions(player, enemy)
        handleFoodEating([player, enemy], food)
        player.draw(surface)
        enemy.draw(surface)
        food.draw(surface)
        screen.blit(surface, (0, 0))
        text1 = myfont.render("Score Player {0}".format(player.score), 1, (250, 250, 250))
        # text2 = myfont.render("Score AI {0}".format(enemy.score), 1, (250, 250, 250))
        screen.blit(text1, (5, 10))
        # screen.blit(text2, (SCREEN_WIDTH - 120, 10))
        pygame.display.update()
    print("Snake Player Final Score:", player.score)
    # print("Snake AI Final Score:", enemy.score)


episode_rewards = []
episode_scores = []
for episode in range(200):
    player = Snake(0)
    enemy = Snake(1)
    food = Food([player, enemy])
    show = False
    if episode % SHOW_EVERY == 0:
        show = True
    if show:
        #displayQL()
        print(f"{SHOW_EVERY} ep mean score: {np.mean(episode_scores[-SHOW_EVERY:])}")

    while player.lives > 0:
        obs = (player-food, player.get_head_position())
        #obs = (player - food)
        enemy.positions = [(random.randint(0, SIZE - 1), random.randint(0, SIZE - 1))]
        action = np.argmax(q_table[obs])

        action_space = [action]
        player.action(action_space, "QL")
        enemyMovement(food=food, enemy=enemy)
        handleSnakeCollisions(player, enemy)
        handleFoodEating([player, enemy], food)
        reward = player.score
        episode_score = player.score
    print(episode_scores)
    episode_scores.append(episode_score)
    episode_rewards.append(reward)

moving_avg2 = np.convolve(episode_scores, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')

print(f"{200} ep mean score: {np.mean(episode_scores)}")

plt.plot([i for i in range(len(moving_avg2))], moving_avg2)
plt.ylabel(f"Score {SHOW_EVERY}ma")
plt.xlabel("episode #")

plt.title(f"{Q_TABLE_NAME} Avg Score: {np.mean(episode_scores)}")
plt.savefig('p_gradient/tests/' + f"{Q_TABLE_NAME}-Scores.png")
plt.show()
