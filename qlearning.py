import datetime

import numpy as np
import matplotlib.pyplot as plt
import pickle

import pygame
from matplotlib import style
import time
from main import Snake, Food, SIZE, SCREEN_WIDTH, SCREEN_HEIGHT, drawGrid, FOOD_REWARD, ENEMY_PENALTY, DEATH_PENALTY, \
    MOVE_PENALTY

style.use("ggplot")


# Q - Learning Variables
HM_EPISODES = 31000
N_STEPS = 100
MAX_STEPS = 200 # the maximum amount of steps we take for each episode

epsilon = 0.9
EPS_DECAY = 0.9993 # Every episode will be epsilon*EPS_DECAY
SHOW_EVERY = 10000 # how often to play through env visually.
start_q_table = None # if we have a pickled Q table, we'll put the filename of it here.
LEARNING_RATE = 0.2
DISCOUNT = 0.8

Q_TABLE_NAME = f"qtable-{HM_EPISODES}ep.p"
print(Q_TABLE_NAME)

PLAYER_N = 2
FOOD_N = 2

'''Q-Table with distance to food and distance to enemy'''
# if start_q_table is None:
#     # initialize the q-table#
#     # action space currently equals the distance from one agent snake to the food
#     # (x1,y1)=relative distance of player to the food (x2,y2)=relative distance of player to the enemy
#     q_table = {}
#     for x1 in range(-SIZE+1, SIZE):
#         for y1 in range(-SIZE+1, SIZE):
#             for x2 in range(-SIZE+1, SIZE):
#                     for y2 in range(-SIZE+1, SIZE):
#                         # each space has four random values for four different actions
#                         # for these two positions, what are the Q-Values of the 4 actions to take
#                         q_table[((x1, y1), (x2, y2))] = [np.random.uniform(-5, 0) for i in range(4)]
'''Simplified Q-Table with only distance to food'''
if start_q_table is None:
    q_table = {}
    for x1 in range(-SIZE+1, SIZE):
        for y1 in range(-SIZE+1, SIZE):
            q_table[(x1, y1)] = [np.random.uniform(-5, 0) for i in range(4)]

if start_q_table is not None:
    print("Loaded q-table")
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

episode_rewards = []
episode_scores = []

### Displays one run of the Q_Learning agent
def displayRL(player,food):
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)

    surface = pygame.Surface(screen.get_size())
    surface = surface.convert()
    drawGrid(surface)

    myfont = pygame.font.SysFont("bahnschrift", 20)
    max_iter = 0
    while (player.lives > 4):
        max_iter += 1
        if max_iter == 50:
            break
        clock.tick(10)
        drawGrid(surface)
        #obs = (player - food, player - enemy)
        obs = (player - food)
        # no random action in display phase
        action_space = np.array(q_table[obs]).copy()
        player.action(action_space, "QL")
        # if player.get_head_position() == enemy.get_head_position():
        #     player.reward = -ENEMY_PENALTY
        #     player.score = -ENEMY_PENALTY
        #     break
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
        # enemy.draw(surface)
        food.draw(surface)
        screen.blit(surface, (0, 0))
        text1 = myfont.render("Score Player {0}".format(player.score), 1, (250, 250, 250))
        # text2 = myfont.render("Score AI {0}".format(enemy.score), 1, (250, 250, 250))
        screen.blit(text1, (5, 10))
        #screen.blit(text2, (SCREEN_WIDTH - 120, 10))
        pygame.display.update()
    print("Snake Player Final Score:", player.score)
    # print("Snake AI Final Score:", enemy.score)

for episode in range(HM_EPISODES):
    player = Snake(0)
    #enemy = Snake(1)
    food = Food([player])
    if episode % SHOW_EVERY == 0:
        print(f"on #{episode}, epsilon is {epsilon}")
        print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
        print(f"{SHOW_EVERY} ep mean score: {np.mean(episode_scores[-SHOW_EVERY:])}")
        print(episode_rewards)
        show = True
    else:
        show = False

    episode_reward = 0
    episode_score = 0
    for i in range(MAX_STEPS):
        #obs = (player-food,player-enemy)
        obs = (player-food)
        if np.random.random() > epsilon:
            # GET THE ACTION
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0, 4)
        # Take the action!
        '''Training Rewards handled inside snake class(kinda inefficient)'''
        '''We want to now the second best option as well, in case the first
            is running into it's own body'''
        action_space = np.array(q_table[obs]).copy()
        player.action(action_space, "QL")
        #### MAYBE ###
        # this could potentially harm the training
        # enemy.move()
        # food.move() # what if the food could move? would that be interesting?
        ##############
        # If snake catches food it receives food reward (RL Reward)
        # Snake should need to know the snake object
        # player.reward -= 4 * manhattan_distance(player.x,player.y,food.x,food.y)
        # if player.get_head_position() == enemy.get_head_position():
        #     player.reward = -DEATH_PENALTY
        if food.position == player.get_head_position():
            player.reward = FOOD_REWARD
            player.score = FOOD_REWARD
        reward = player.reward
        # first we need to obs immediately after the move.
        #new_obs = (player - food, player - enemy)
        new_obs = (player-food)
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]

        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        q_table[obs][action] = new_q


        episode_reward = player.reward
        episode_score = player.score
        if reward == -ENEMY_PENALTY:
            break

    if show:
        displayRL(player, food)
    episode_rewards.append(episode_reward)
    episode_scores.append(episode_score)
    epsilon *= EPS_DECAY

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')
moving_avg2 = np.convolve(episode_scores, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')


print(f"on #{HM_EPISODES}, epsilon is {epsilon}")
print(f"{SHOW_EVERY} ep mean reward: {np.mean(episode_rewards[-SHOW_EVERY:])}")
print(f"{SHOW_EVERY} ep mean score: {np.mean(episode_scores[-SHOW_EVERY:])}")

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.show()

plt.plot([i for i in range(len(moving_avg2))], moving_avg2)
plt.ylabel(f"Score {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.title(f"Rewards:M:{MOVE_PENALTY}F:{FOOD_REWARD}D:{DEATH_PENALTY}; a:{LEARNING_RATE} d:{DISCOUNT}ep:{HM_EPISODES}")
plt.show()
plt.savefig('graphs/' + f"QL-{HM_EPISODES}.p")

with open(Q_TABLE_NAME, "wb") as f:
    pickle.dump("q_tables/" + q_table, f)

