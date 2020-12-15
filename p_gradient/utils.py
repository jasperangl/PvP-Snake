import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pygame

from main import Food, SIZE, SCREEN_WIDTH, drawGrid, Snake, handleFoodEating, SCREEN_HEIGHT, handleSnakeCollisions

# For now represent the grid as Numpy 2D array using only 4 numbers for every cells state
# 0: Empty
# 1: Agent Snake
# 2: Food
# 3: Enemy Snake
# Example Grid:
#     +---+---+---+---+---+
#     | 2 |   | 3 | 3 |   |
#     +---+---+---+---+---+
#     |   |   |   | 3 | 3 |
#     +---+---+---+---+---+
#     |   |   | 1 |   |   |
#     +---+---+---+---+---+
#     |   |   | 1 | 1 |   |
#     +---+---+---+---+---+
#     |   |   |   | 1 |   |
#     +---+---+---+---+---+

'''For now make border and enemy have the same meaning'''
def getObsGrid(snakes: list, food: Food, size: int, fullGrid: bool):
    player = snakes[0]
    snakeBody = []
    enemyBody = []
    foodList = []
    if fullGrid:
        grid = np.empty(shape=(SIZE, SIZE), dtype=np.int8)
        # Resets grid before reassignment
        grid.fill(0)
        for x in range(SIZE):
            for y in range(SIZE):
                curr_posn = (x, y)
                if curr_posn in player.positions:
                    snakeBody.append(curr_posn)
                elif curr_posn == food.position:
                    foodList.append(curr_posn)
                elif len(snakes) > 1:
                    enemy = snakes[1]
                    if curr_posn in enemy.positions:
                        enemyBody.append(curr_posn)
                if curr_posn[0] >= SIZE or curr_posn[0] < 0 or curr_posn[1] >= SIZE or curr_posn[1] < 0:
                    enemyBody.append(curr_posn)

        # Now relativize positions so that they are applicable for numpy array4
        for posn in snakeBody:
            grid[posn[1]][posn[0]] = 1
        for posn in foodList:
            grid[posn[1]][posn[0]] = 2
        for posn in enemyBody:
            grid[posn[1]][posn[0]] = 3
        grid = grid.flatten()


    else:
        grid = np.empty(shape=(size, size), dtype=np.int8)
        # Resets grid before reassignment
        grid.fill(0)
        vectFactor = math.floor(size/2)
        if len(snakes) > 1:
            enemy = snakes[1]
        center = player.get_head_position()
        spanV1 = (center[0] - vectFactor, center[1] - vectFactor)
        spanV2 = (center[0] + vectFactor, center[1] + vectFactor)
        #print("Center:", center)
        #print("Span 1:", spanV1)
        #print("Span 2:", spanV2)
        for x in range(spanV1[0],spanV2[0] + 1):
            for y in range(spanV1[1],spanV2[1] + 1):
                curr_posn = (x,y)
                if curr_posn in player.positions:
                    snakeBody.append(curr_posn)
                elif curr_posn == food.position:
                    foodList.append(curr_posn)
                elif len(snakes) > 1:
                    enemy = snakes[1]
                    if curr_posn in enemy.positions:
                        enemyBody.append(curr_posn)
                if curr_posn[0] >= SIZE or curr_posn[0] < 0 or curr_posn[1] >= SIZE or curr_posn[1] < 0:
                    enemyBody.append(curr_posn)
        #print("Snake Body:", snakeBody)

        # Now relativize positions so that they are applicable for numpy array4
        for posn in snakeBody:
            rel_x, rel_y = sub_posn(posn, spanV1)
            grid[rel_x][rel_y] = 1
        for posn in foodList:
            rel_x, rel_y = sub_posn(posn, spanV1)
            grid[rel_x][rel_y] = 2
        for posn in enemyBody:
            rel_x, rel_y = sub_posn(posn, spanV1)
            grid[rel_x][rel_y] = 3
        grid = grid.flatten()
    return grid

def sub_posn(posn1,posn2):
    return posn1[0] - posn2[0], posn1[1] - posn2[1]


# This observation space contains only a limited amount of information:
# 1. Whether an obstacle is to the left right or in front of the snake
# 2. Normalized angle between movement direction and food position
# 3. Normalized angle between movement direction and enemy position (assuming only enemies head position)
def getObsSmall(snakes: list, food: Food):
    player = snakes[0]
    enemy = snakes[1]
    # list only contains numbers from -1 to 1
    obs_space = []
    # Step 1
    barrier_left = is_direction_blocked(player, enemy, turn_vector_to_the_left(player.direction))
    barrier_front = is_direction_blocked(player, enemy, player.direction)
    barrier_right = is_direction_blocked(player, enemy, turn_vector_to_the_right(player.direction))
    # Step 2
    snake_direction = np.array(player.direction)
    food_direction = get_food_direction_vector(player, food)
    enemy_direction = get_enemy_direction_vector(player, enemy)
    food_distance = get_food_distance(player, food)
    enemy_distance = get_enemy_distance(player, enemy)
    # Leave it out for now
    #enemy_center_direction = get_food_direction_vector(snake, self.get_snake_center(enemy))

    food_angle = get_angle(snake_direction, food_direction)
    enemy_angle = get_angle(snake_direction, enemy_direction)
    # Knowing about enemies center position
    #enemy_cog_angle = get_angle(snake_direction, enemy_center_direction)
    # Knowing about own center position
    #player_cog_distance = get_food_distance(snake, self.get_snake_center(enemy))
    #print(f"Front:{int(barrier_front)}, Left:{int(barrier_left)}, Right:{int(barrier_right)}, Food:{round(food_angle, 1)}, Enemy:{round(enemy_angle, 1)}")

    return np.array([int(barrier_front),
                     int(barrier_left),
                     int(barrier_right),
                     food_angle,
                     food_distance,
                     enemy_distance
                     # enemy_cog_angle,
                     # player_cog_distance
                     ])

def get_angle(a, b):
    a = normalize_vector(a)
    b = normalize_vector(b)
    return math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]) / math.pi

def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / np.linalg.norm(vector)

def get_enemy_distance(snake, enemy):
    return np.linalg.norm(get_enemy_direction_vector(snake, enemy))

def get_food_distance(snake, food):
    return np.linalg.norm(get_food_direction_vector(snake, food))

# returns true if future direction is blocked either by wall, itself or enemy
def is_direction_blocked(snake: Snake, enemy: Snake, direction):
    point = np.array(snake.get_head_position()) + np.array(direction)
    #print(point)
    return point.tolist() in snake.positions[:-1] \
           or point.tolist() in enemy.positions \
           or point[0] == 0 or point[1] == 0 \
           or point[0] == SIZE or point[1] == SIZE

def get_food_direction_vector(snake: Snake, food: Food):
    return np.array(food.position) - np.array(snake.get_head_position())

def get_enemy_direction_vector(snake: Snake, enemy: Snake):
    return np.array(enemy.get_head_position()) - np.array(snake.get_head_position())

def turn_vector_to_the_right(vector):
    return np.array([-vector[1], vector[0]])

def turn_vector_to_the_left(vector):
    return np.array([vector[1], -vector[0]])

def enemyMovement(food: Food, enemy: Snake):
    enemy_epsilon = 0.5
    direction = 0
    if np.random.random() <= enemy_epsilon:
        direction = random.randint(0, 3)
    else:
        if food.position[0] < enemy.get_head_position()[0]:
            direction = 2
        elif food.position[0] > enemy.get_head_position()[0]:
            direction = 3
        elif food.position[1] < enemy.get_head_position()[1]:
            direction = 0
        elif food.position[1] > enemy.get_head_position()[1]:
            direction = 1
        else:
            direction = random.randint(0, 3)
    enemy.action([direction], "Enemy")

def render(agent, obs_type: str, board_size: int):
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)

    surface = pygame.Surface(screen.get_size())
    surface = surface.convert()
    drawGrid(surface)

    player = Snake(0)
    player.lives = 5
    enemy = Snake(1)
    enemy.lives = 5
    enemy.positions = [[5,5]]
    food = Food([player])

    myfont = pygame.font.SysFont("bahnschrift", 20)
    iterations = 0
    while (player.lives > 4 and enemy.lives > 4):
        clock.tick(4)
        drawGrid(surface)
        enemyMovement(food=food,enemy=enemy)
        # no random action in display phase
        if obs_type == "Grid":
            obs = getObsGrid([player], food, size=board_size, fullGrid=False)
        if obs_type == 'Small':
            obs = getObsSmall([player, enemy], food)
        action = agent.choose_action(obs)
        action_space = [action]
        player.action(action_space, "PG")

        handleSnakeCollisions(player,enemy)
        handleFoodEating([player, enemy], food)
        player.draw(surface)
        enemy.draw(surface)

        food.draw(surface)
        screen.blit(surface, (0, 0))
        text1 = myfont.render("Score AI {0}".format(player.score), 1, (250, 250, 250))
        #text2 = myfont.render("Score Player {0}".format(enemy.score), 1, (250, 250, 250))
        screen.blit(text1, (5, 10))
        #screen.blit(text2, (SCREEN_WIDTH - 175, 10))
        pygame.display.update()
    print("Snake Player Final Score:", player.score)
    #print("Snake AI Final Score:", enemy.score)



def plotLearning(scores, filename, title, x=None, window=5, ylabel= str):
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])
    if x is None:
        x = [i for i in range(N)]
    plt.title = title
    plt.ylabel(ylabel)
    plt.xlabel('Games')
    plt.plot(x, running_avg)
    plt.savefig(filename)
    plt.show()