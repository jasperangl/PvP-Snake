import pygame
import sys
import random
import math
import numpy as np
import pickle



pygame.font.init()



# Reward variables
MOVE_PENALTY = 0.0
FOOD_REWARD = 5
DEATH_PENALTY = 10

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
SNAKE_COLOR = ((138,184,45), (0,171,239), (230,230,30))
SNAKE_HEAD_COLOR = ((154,205,50), (0,191,255), (255,255,51))
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
        self.color_head = SNAKE_HEAD_COLOR[snake_id]
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
            #print("Snake wants to run into body")
            return
        else:
            self.direction = point


    def move(self):
        current = self.get_head_position()
        if self.snake_id == 0 or self.snake_id == 1:
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
            self.reward = -MOVE_PENALTY
            self.positions.insert(0, newPosition)
            if len(self.positions) > self.length:
                self.positions.pop()
            self.x = self.positions[0][0]
            self.y = self.positions[0][1]

    '''In the future respawn can only happen if other snake is not on this spot'''
    def reset(self):
        self.lives -= 1
        self.length = 1
        self.positions = [(math.ceil((SIZE/4)+self.snake_id*(SIZE/2)), math.ceil(SIZE/2))]
        self.direction = random.choice([UP, DOWN, LEFT, RIGHT])
        self.score -= DEATH_PENALTY
        self.reward = -DEATH_PENALTY
        self.x = self.positions[0][0]
        self.y = self.positions[0][1]

    def draw(self, surface):
        for p in self.positions:
            r = pygame.Rect((p[0] * GRIDSIZE, p[1] * GRIDSIZE), (GRIDSIZE, GRIDSIZE))
            if p == self.get_head_position():
                pygame.draw.rect(surface, self.color_head, r)
            else:
                pygame.draw.rect(surface, self.color, r)
            pygame.draw.rect(surface, SQUARE_COLOR, r, 1)


    # handles key input in case someone plays the game themselves
    def handle_keys(self):
        for event in pygame.event.get():
            '''In the future no agent will quit the game'''
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN and self.snake_id == 1:
                if event.key == pygame.K_UP:
                    self.turn(UP)
                elif event.key == pygame.K_DOWN:
                    self.turn(DOWN)
                elif event.key == pygame.K_LEFT:
                    self.turn(LEFT)
                elif event.key == pygame.K_RIGHT:
                    self.turn(RIGHT)
            elif event.type == pygame.KEYDOWN and self.snake_id == 0:
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
            if len(action_space) > 1:
                action_space_indecies = action_space.argsort()[-3:][::-1]
                choice = action_space_indecies[0]
                choice_direction = MOVES.get(choice)
                #print("First Choice:", choice)
                if self.length > 1 and (choice_direction[0] * -1, choice_direction[1] * -1) == self.direction:
                    choice = action_space_indecies[1]

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

def drawSnakeVision(surface, snake:Snake):
    center = snake.get_head_position()
    vectFactor = 2
    spanV1 = (center[0] - vectFactor, center[1] - vectFactor)
    spanV2 = (center[0] + vectFactor, center[1] + vectFactor)
    # print("Center:", center)
    # print("Span 1:", spanV1)
    # print("Span 2:", spanV2)
    for x in range(spanV1[0], spanV2[0] + 1):
        for y in range(spanV1[1], spanV2[1] + 1):
            r = pygame.Rect((x* GRIDSIZE, y* GRIDSIZE), (GRIDSIZE,GRIDSIZE))
            pygame.draw.rect(surface, colorBrighten(SQUARE_COLOR), r)

def colorBrighten(color):
    return (color[0]+25,color[1]+25,color[2]+25)
###############################################################
#####################   UTILS   ###############################

# determines whether snake1 ran into snake2 and outputs True or False
def ranIntoSnake(snake1: Snake, snake2: Snake):
    return snake1.get_head_position() in snake2.positions

# resets snakes according to collisions
def handleSnakeCollisions(snake1: Snake, snake2: Snake):
    if snake1.get_head_position() == snake2.get_head_position():
        snake1.reset()
        snake2.reset()
    if snake1.get_head_position() in snake2.positions:
        snake1.reset()
    if snake2.get_head_position() in snake1.positions:
        snake2.reset()

# determines whether given snake ran into food
def ranIntoFood(snake1: Snake, food: Food):
    return snake1.get_head_position() == food.position

# handles food position and snake behavior if Food is eaten
def handleFoodEating(snakes: list, food: Food):
    for snake in snakes:
        if snake.get_head_position() == food.position:
            snake.length += 1
            # food score relative and food reward absolute
            snake.score += FOOD_REWARD
            snake.reward += FOOD_REWARD
            food.randomize_position(snakes)

def manhattan_distance(x1,y1,x2,y2):
    return abs(x1 - x2) + abs(y1 - y2)

###############################################################

def test_env():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
    surface = pygame.Surface(screen.get_size())
    surface = surface.convert()
    drawGrid(surface)

    snake = Snake(0)
    snake2 = Snake(1)
    food = Food([snake, snake2])
    snake.positions = [(3,7),(3,6),(3,5),(4,5),(5,5)]
    snake2.positions = [(7,7),(8,7),(9,7)]
    food.position = (5, 9)
    drawSnakeVision(surface, snake)


    myfont = pygame.font.SysFont("bahnschrift", 20)
    while (snake.lives > 0 and snake2.lives > 0):
        snake.draw(surface)
        snake2.draw(surface)
        food.draw(surface)
        screen.blit(surface, (0, 0))
        text1 = myfont.render("Score Player {}".format(10), 1, (250, 250, 250))
        text2 = myfont.render("Score AI {0}".format(-5), 1, (250, 250, 250))
        screen.blit(text1, (5, 10))
        screen.blit(text2, (SCREEN_WIDTH - 120, 10))
        pygame.display.update()

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
        handleSnakeCollisions(snake, snake2)
        handleFoodEating(snake,snake2,food)
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
