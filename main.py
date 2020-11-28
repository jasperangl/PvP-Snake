import pygame
import sys
import random
import math
import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt
from matplotlib import style
import time



pygame.font.init()



# Reward variables
MOVE_PENALTY = 0.5
FOOD_REWARD = 25
ENEMY_PENALTY = 100
DEATH_PENALTY = 100

# Game variables
SIZE = 10
SNAKE_LIVES = 5
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
GRIDSIZE = SCREEN_WIDTH / SIZE

UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1,0)
RIGHT = (1,0)
MOVES = {0:UP, 1:DOWN, 2:LEFT, 3:RIGHT}

SQUARE_COLOR = (80,80,80)
SNAKE_COLOR = ((154,205,50), (50,50,250))
FOOD_COLOR = (255,69,0)

class Snake():
    # snake_id 0 for snake 1, 1 for snake 2
    def __init__(self, snake_id):
        self.lives = SNAKE_LIVES
        self.length = 1
        self.snake_id = snake_id
        # spawn snake 1 middle left and snake 2 middle right
        self.positions = [(math.ceil((SIZE/4)+snake_id*(SIZE/2)), math.ceil(SIZE/2))]
        self.direction = random.choice([UP, DOWN, LEFT, RIGHT])
        self.color = SNAKE_COLOR[snake_id]
        self.score = 0
        self.reward = 0 # used for RL Training(probably should be the same as score)
        self.x = self.positions[0][0]
        self.y = self.positions[0][1]

    # relative distance to other object with x,y properties
    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)

    def get_head_position(self):
        return self.positions[0]


    def turn(self, point):
        if self.length > 1 and (point[0] * -1, point[1] * -1) == self.direction:
            print("Snake wants to run into body")
            return
        else:
            self.direction = point

    '''Need to add if snake runs into other snakes head, both snakes die and if snake runs into other snakes 
        tail, only this snake dies'''
    def move(self):
        current = self.get_head_position()
        if self.snake_id == 0:
            x,y = self.direction
        else:
            x, y = random.choice([UP, DOWN, LEFT, RIGHT])
        newPosition =  ((current[0]+x), (current[1]+y))


        # if snake catches its own tail game is ended
        if len(self.positions) > 2 and newPosition in self.positions[2:]:
            self.reset()

        # if snake runs into border game is ended
        elif newPosition[0] >= SIZE or newPosition[0] < 0 or newPosition[1] >= SIZE or newPosition[1] < 0 :
            self.reset()
        # else we add new head position and pop the last one
        else:
            self.reward -= MOVE_PENALTY
            self.positions.insert(0, newPosition)
            if len(self.positions) > self.length:
                self.positions.pop()
            self.x = self.positions[0][0]
            self.y = self.positions[0][1]

    '''In the future respawn can only happen if other snake is not on this spot'''
    def reset(self):
        self.lives -= 1
        self.length = 1
        self.positions = [((SIZE // 2), (SIZE // 2))]
        self.direction = random.choice([UP, DOWN, LEFT, RIGHT])
        self.score = -DEATH_PENALTY
        self.reward = -DEATH_PENALTY
        self.x = self.positions[0][0]
        self.y = self.positions[0][1]

    def draw(self, surface):
        for p in self.positions:
            r = pygame.Rect((p[0]*GRIDSIZE, p[1]*GRIDSIZE), (GRIDSIZE, GRIDSIZE))
            pygame.draw.rect(surface, self.color, r)
            pygame.draw.rect(surface, SQUARE_COLOR, r, 1)


    # handles key input in case someone plays the game themselves
    def handle_keys(self):
        for event in pygame.event.get():
            '''In the future no agent will quit the game'''
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN and self.snake_id == 0:
                if event.key == pygame.K_UP:
                    self.turn(UP)
                elif event.key == pygame.K_DOWN:
                    self.turn(DOWN)
                elif event.key == pygame.K_LEFT:
                    self.turn(LEFT)
                elif event.key == pygame.K_RIGHT:
                    self.turn(RIGHT)
            elif event.type == pygame.KEYDOWN and self.snake_id == 1:
                if event.key == pygame.K_w:
                    self.turn(UP)
                elif event.key == pygame.K_s:
                    self.turn(DOWN)
                elif event.key == pygame.K_a:
                    self.turn(LEFT)
                elif event.key == pygame.K_d:
                    self.turn(RIGHT)

    '''Necessary parts for RL'''

    # This is used for the RL Methods so that the snake takes a move bases on inputs 1,2,3 or 4
    def action(self, action_space, algorithm):
        choice = action_space[0]
        if algorithm == "QL":
            action_space_indecies = action_space.argsort()[-3:][::-1]
            choice = action_space_indecies[0]
            choice_direction = MOVES.get(choice)
            #print("First Choice:", choice)
            if self.length > 1 and (choice_direction[0] * -1, choice_direction[1] * -1) == self.direction:
                choice = action_space_indecies[1]
                #print("Second Choice:", choice)

        # Choice is either 0,1,2,3 uses dictionary MOVES
        self.turn(MOVES.get(choice))
        self.move()

class Food():
    # Snakes is a list of all snakes
    def __init__(self, Snakes):
        self.position = (0,0)
        self.color = FOOD_COLOR
        self.snakes = Snakes
        self.randomize_position(self.snakes)
        self.x = self.position[0]
        self.y = self.position[1]


    # ensures food doesn't spawn on the snakes
    def randomize_position(self, Snakes):
        posn = (random.randint(0, SIZE - 1), random.randint(0, SIZE - 1))
        for snake in Snakes:
            if posn not in snake.positions[1:]:
                self.position = posn
            else:
                self.randomize_position(Snakes)
        self.x = self.position[0]
        self.y = self.position[1]


    def draw(self, surface):
        r = pygame.Rect((self.position[0]*GRIDSIZE, self.position[1]*GRIDSIZE), (GRIDSIZE, GRIDSIZE))
        pygame.draw.rect(surface, self.color, r)
        pygame.draw.rect(surface, SQUARE_COLOR, r, 1)


def drawGrid(surface):
    for y in range(0, int(SIZE)):
        for x in range(0, int(SIZE)):
            r = pygame.Rect((x* GRIDSIZE, y* GRIDSIZE), (GRIDSIZE,GRIDSIZE))
            pygame.draw.rect(surface, SQUARE_COLOR, r)

def manhattan_distance(x1,y1,x2,y2):
    return abs(x1 - x2) + abs(y1 - y2)

def main():
    pygame.init()

    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)

    surface = pygame.Surface(screen.get_size())
    surface = surface.convert()
    drawGrid(surface)

    snake = Snake(0)
    snake2 = Snake(1)
    food = Food([snake,snake2])

    myfont = pygame.font.SysFont("bahnschrift",20)

    while (snake.lives > 0 and snake2.lives > 0):
        clock.tick(5)
        snake.handle_keys()
        snake2.handle_keys()
        drawGrid(surface)
        snake.move()
        snake2.move()
        if snake.get_head_position() == food.position:
            snake.length += 1
            snake.score += FOOD_REWARD
            food.randomize_position([snake,snake2])
        if snake2.get_head_position() == food.position:
            snake2.length += 1
            snake2.score += FOOD_REWARD
        snake.draw(surface)
        snake2.draw(surface)
        food.draw(surface)
        screen.blit(surface, (0,0))
        text1 = myfont.render("Score Player {0}".format(snake.score), 1, (250,250,250))
        text2 = myfont.render("Score AI {0}".format(snake2.score), 1, (250, 250, 250))
        screen.blit(text1, (5,10))
        screen.blit(text2, (SCREEN_WIDTH-120, 10))
        pygame.display.update()

    print("Snake Player Final Score:", snake.score)
    print("Snake AI Final Score:", snake2.score)
