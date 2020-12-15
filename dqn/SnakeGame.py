import math

import pygame
import numpy as np
from main import Snake, SCREEN_WIDTH, SCREEN_HEIGHT, drawGrid, GRIDSIZE
from random import randint

FOOD_REWARD = 5
DEATH_PENALTY = 10
MOVE_PENALTY = 0.1
LIVES = 5

SQUARE_COLOR = (80,80,80)
SNAKE_HEAD_COLOR = ((0,51,0), (0,0,153), (102,0,102))
SNAKE_COLOR = ((154,205,50), (50,50,250), (50,0,250))
FOOD_COLOR = (255,69,0)

class SnakeGame:
    def __init__(self, board_width = 10, board_height = 10, gui = False, enemy_epsilon=0.1):
        self.score = 0
        self.board = {'width': board_width, 'height': board_height}
        self.gui = gui
        self.lives = LIVES
        self.player = []
        self.enemy = []
        self.enemy_epsilon = enemy_epsilon
        self.food = []

    def start(self):
        '''
        :return: [lives, score, player, enemy, food]
        '''
        self.player_init(LIVES)
        self.enemy_init()
        self.generate_food()
        if self.gui: self.render_init()
        return self.generate_observations()

    def player_init(self, lives=LIVES):
        x = randint(3, math.ceil(self.board["width"] / 2) - 1)
        y = randint(3, self.board["height"] - 3)
        self.player = []
        vertical = randint(0, 1) == 0
        for i in range(3):
            point = [x + i, y] if vertical else [x, y + i]
            self.player.insert(0, point)
        self.lives = lives

    def enemy_init(self):
        x = randint(math.ceil(self.board["width"] / 2), self.board["width"] - 3)
        y = randint(3, self.board["height"] - 3)
        self.enemy = []
        vertical = randint(0, 1) == 0
        for i in range(3):
            point = [x + i, y] if vertical else [x, y + i]
            self.enemy.insert(0, point)

        if self.enemy[0] in self.player[1:-1]:
            self.enemy_init() # retry

    def generate_food(self):
        food = []
        while not food:
            food = [randint(1, self.board["width"]), randint(1, self.board["height"])]
            if food in self.enemy: food = []
            elif food in self.player: food = []
        self.food = food

    def get_enemy_movement(self):
        '''
        0 - UP, (-1, 0)
        1 - RIGHT, (
        2 - DOWN,
        3 - LEFT
        '''
        if np.random.random() <= self.enemy_epsilon:
            return randint(0, 3)

        if self.food[0] > self.enemy[0][0]:
            return 2
        elif self.food[0] < self.enemy[0][0]:
            return 0
        elif self.food[1] > self.enemy[0][1]:
            return 1
        elif self.food[1] < self.enemy[0][1]:
            return 3

        return randint(0, 3)

    def step(self, key):
        '''
        0 - UP,
        1 - RIGHT,
        2 - DOWN,
        3 - LEFT

        :param key:
        :return: [lives, score, player, enemy, food]
        '''

        if self.is_done() :
            self.end_game()

        if not self.food:
            self.generate_food()

        self.create_new_point(self.player, key)
        self.create_new_point(self.enemy, self.get_enemy_movement())

        player_ate = False
        if self.food_eaten(self.player):
            self.score += FOOD_REWARD
            self.generate_food()
            player_ate = True
        else:
            self.remove_last_point(self.player)
            self.score -= MOVE_PENALTY

        if (not player_ate) and self.food_eaten(self.enemy):
            self.generate_food()
        else:
            self.remove_last_point(self.enemy)

        self.check_collisions()

        if not self.food:
            self.generate_food()

        return self.generate_observations()

    def create_new_point(self, snake, key):
        new_point = [snake[0][0], snake[0][1]]
        if key == 0: # UP
            new_point[0] -= 1
        elif key == 1: # RIGHT
            new_point[1] += 1
        elif key == 2: # DOWN
            new_point[0] += 1
        elif key == 3: # LEFT
            new_point[1] -= 1
        snake.insert(0, new_point)

    def food_eaten(self, snake):
        return self.food in snake

    def remove_last_point(self, snake):
        snake.pop()


    def check_collisions(self):

        state = 0
        # 0 -> no collision,
        # 1 -> player collision,
        # 2 -> enemy collision

        player_collided = False
        enemy_collided = False

        if (self.player[0][0] == 0 or
                self.player[0][0] == self.board["width"] or
                self.player[0][1] == 0 or
                self.player[0][1] == self.board["height"] or
                self.player[0] in self.player[1:-1] or
                self.player[0] in self.enemy):
            player_collided = True

        if (self.enemy[0][0] == 0 or
                self.enemy[0][0] == self.board["width"] or
                self.enemy[0][1] == 0 or
                self.enemy[0][1] == self.board["height"] or
                self.enemy[0] in self.player or
                self.enemy[0] in self.enemy[1:-1]):
            enemy_collided = True

        if player_collided:
            self.lives -= 1
            if not self.is_done():
                self.player_init(self.lives)

        if enemy_collided:
            self.enemy_init() # enemy moves randomly but has infinite lives

    def generate_observations(self):
        '''
        :return: [lives, score, player, enemy, food]
        '''
        return self.lives, self.score, self.player, self.enemy, self.food

    '''Methods for Rendering the game'''

    def render_init(self):
        pygame.init()
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)

        self.surface = pygame.Surface(self.screen.get_size())
        self.surface = self.surface.convert()
        drawGrid(self.surface)
        self.myfont = pygame.font.SysFont("bahnschrift", 20)

    def step_render(self, key):
        '''
        :return: [lives, score, player, enemy, food]
        '''
        self.clock.tick(3)
        drawGrid(self.surface)

        if not self.food:
            self.generate_food()

        _lives, _score, _player, _enemy, _food = self.step(key)

        self.draw_snake(self.player, self.surface, SNAKE_COLOR[0], SNAKE_HEAD_COLOR[0])
        self.draw_snake(self.enemy, self.surface, SNAKE_COLOR[1], SNAKE_HEAD_COLOR[1])

        if not self.food:
            self.generate_food()

        self.draw_food(self.surface, FOOD_COLOR)

        self.screen.blit(self.surface, (0, 0))
        text1 = self.myfont.render("Score: {0} Lives: {1}".format(round(self.score, 2), self.lives), True, (250, 250, 250))
        # text2 = myfont.render("Score AI {0}".format(enemy.score), 1, (250, 250, 250))
        self.screen.blit(text1, (5, 10))
        # screen.blit(text2, (SCREEN_WIDTH - 120, 10))
        pygame.display.update()
        return _lives, _score, _player, _enemy, _food

    def draw_snake(self, snake, surface, color, head_color):
        drew_head = False
        for p in snake:
            curr_color = color
            if not drew_head:
                curr_color = head_color
                drew_head = True

            r = pygame.Rect((p[0]*GRIDSIZE, p[1]*GRIDSIZE), (GRIDSIZE, GRIDSIZE))
            pygame.draw.rect(surface, curr_color, r)
            pygame.draw.rect(surface, SQUARE_COLOR, r, 1)

    def draw_food(self, surface, color):
        r = pygame.Rect((self.food[0] * GRIDSIZE, self.food[1] * GRIDSIZE), (GRIDSIZE, GRIDSIZE))
        pygame.draw.rect(surface, color, r)
        pygame.draw.rect(surface, SQUARE_COLOR, r, 1)

    def is_done(self):
        return self.lives <= 0

    def render_destroy(self):
        print("Snake Player Final Score:", self.score)

    def end_game(self):
        if self.gui: self.render_destroy()
        raise Exception("Game over")


