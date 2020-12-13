# from snake_game import SnakeGame
import sys
from random import randint
import numpy as np
import pygame
import tflearn
import math
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean
from collections import Counter

from main import Snake, Food, LEFT, DOWN, RIGHT, UP, handleSnakeCollisions, handleFoodEating, SCREEN_WIDTH, \
    SCREEN_HEIGHT, drawGrid, DEATH_PENALTY, SIZE

EPS_DECAY = 0.9997
OUTPUT_DIR = 'tensorflow_out/'
FILENAME = 'snake_dqn_2_6nodes_low_reward.tflearn'
INPUT_LAYER = 6

class SnakeDQN:
    def __init__(self,epsilon = 0.9, initial_games = 10000, test_games = 10000, goal_steps = 2000, lr = 1e-2,
                 filename = OUTPUT_DIR + FILENAME):

        self.epsilon = epsilon
        self.initial_games = initial_games
        self.test_games = test_games
        self.goal_steps = goal_steps
        self.lr = lr
        self.filename = filename
        self.vectors_and_keys = [
                [UP, 0],
                [DOWN, 1],
                [LEFT, 2],
                [RIGHT, 3]
        ]

    def initial_population(self):
        training_data = []
        for _ in range(self.initial_games):
            player = Snake(0)
            player.init_positions()

            prev_lives, prev_score = player.generate_observations()

            enemy = Snake(1)
            enemy.init_positions()

            food = Food([player, enemy])

            # print(player.positions)
            # print(enemy.positions)

            # game = SnakeGame()
            # _, prev_score, player, food = game.start()
            # prev_action =
            prev_observation = self.generate_observation(player, food, enemy)
            prev_food_distance = self.get_food_distance(player, food)

            for _ in range(self.goal_steps):
                action, game_action = self.generate_action(player, food)
                done, score, player, enemy, food = self.game_step(player, enemy, food, game_action)
                if done:
                    training_data.append([self.add_action_to_observation(prev_observation, action), -1])
                    break
                else:
                    food_distance = self.get_food_distance(player, food)
                    if score > prev_score or food_distance < prev_food_distance:
                        training_data.append([self.add_action_to_observation(prev_observation, action), 1])
                    else:
                        training_data.append([self.add_action_to_observation(prev_observation, action), 0])
                    prev_observation = self.generate_observation(player, food, enemy)
                    prev_food_distance = food_distance
        return training_data

    def game_step(self, player, enemy, food, action_key, enemy_ctrl=False):

        if np.random.random() > self.epsilon:
            # GET THE ACTION
            action = self.vectors_and_keys[action_key][1]

        else:
            action = np.random.randint(0, 4)

        if enemy_ctrl:
            enemy.handle_keys()
            enemy.move()
        else:
            enemy.positions = [(randint(0, SIZE - 1), randint(0, SIZE - 1))]
            #### MAYBE ###
            # this could potentially harm the training
            # enemy.move()
            # food.move() # what if the food could move? would that be interesting?
            ##############

        # Take the action!
        _lives = player.lives
        player.action([action], "DQN")

        handleSnakeCollisions(player, enemy)
        handleFoodEating(player, enemy, food)

        if(player.lives < _lives):
            player.lives = _lives - 1

        if len(player.positions) < 2:
            player.init_positions()

        if len(enemy.positions) < 2:
            enemy.init_positions()

        lives, score = player.generate_observations()
        done = lives < 1

        self.epsilon *= EPS_DECAY

        return done, score, player, enemy, food

    def generate_action(self, snake, food):
        action = randint(0,2) - 1

        game_action = self.get_game_action(snake, action)

        if snake.get_head_position()[1] == food.position[1]:
            if(snake.get_head_position()[0] - 1 == food.position[0]):
                action = 0
                game_action = 3 # RIGHT
            elif(snake.get_head_position()[0] + 1 == food.position[0]):
                action = 0
                game_action = 2 # LEFT
        elif snake.get_head_position()[0] == food.position[0]:
            if (snake.get_head_position()[1] - 1 == food.position[1]):
                action = 0
                game_action = 0 # UP
            elif (snake.get_head_position()[1] + 1 == food.position[1]):
                action = 0
                game_action = 1 # DOWN

        return action, game_action

    def get_game_action(self, snake, action):
        snake_direction = self.get_snake_direction_vector(snake)
        new_direction = snake_direction
        if action == -1:
            new_direction = self.turn_vector_to_the_left(snake_direction)
        elif action == 1:
            new_direction = self.turn_vector_to_the_right(snake_direction)

        game_action = 0
        for pair in self.vectors_and_keys:
            new_pair = [new_direction[0], new_direction[1]]
            if pair[0] == new_pair:
                game_action = pair[1]
        return game_action

    def generate_observation(self, snake, food, enemy):
        snake_direction = self.get_snake_direction_vector(snake)
        food_direction = self.get_food_direction_vector(snake, food)
        enemy_direction = self.get_snake_direction_vector(enemy)

        # print("dirs: {0}, {1}".format(snake_direction, food_direction))

        barrier_left = self.is_direction_blocked(snake, enemy, self.turn_vector_to_the_left(snake_direction))
        barrier_front = self.is_direction_blocked(snake, enemy, snake_direction)
        barrier_right = self.is_direction_blocked(snake, enemy, self.turn_vector_to_the_right(snake_direction))

        food_angle = self.get_angle(snake_direction, food_direction)
        enemy_angle = self.get_angle(snake_direction, enemy_direction)
        return np.array([int(barrier_left), int(barrier_front), int(barrier_right), food_angle, enemy_angle])

    def add_action_to_observation(self, observation, action):
        return np.append([action], observation)

    def get_snake_direction_vector(self, snake):
        return np.array(snake.positions[0]) - np.array(snake.positions[1])

    def get_food_direction_vector(self, snake, food):
        return np.array(food.position) - np.array(snake.positions[0])

    def normalize_vector(self, vector):
        denom = np.linalg.norm(vector)

        if denom == 0:
            return [0,0]

        return vector / denom

    def get_food_distance(self, snake, food):
        return np.linalg.norm(self.get_food_direction_vector(snake, food))

    def is_direction_blocked(self, snake, enemy, direction):
        pointList = np.array(snake.get_head_position()) + np.array(direction)
        point = (pointList[0], pointList[1])

        return point in snake.positions[:-1] or \
               point[0] < 0 or \
               point[1] < 0 or \
               point[0] >= SIZE or \
               point[1] >= SIZE

    def turn_vector_to_the_left(self, vector):
        return np.array([-vector[1], vector[0]])

    def turn_vector_to_the_right(self, vector):
        return np.array([vector[1], -vector[0]])

    def get_angle(self, a, b):
        a = self.normalize_vector(a)
        b = self.normalize_vector(b)
        return math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]) / math.pi

    def model(self):
        network = input_data(shape=[None, INPUT_LAYER, 1], name='input')
        network = fully_connected(network, INPUT_LAYER * INPUT_LAYER, activation='relu') # hidden layer
        network = fully_connected(network, 1, activation='linear')
        network = regression(network, optimizer='adam', learning_rate=self.lr, loss='mean_square', name='target')
        model = tflearn.DNN(network, tensorboard_dir='../log')
        return model

    def train_model(self, training_data, model):
        X = np.array([i[0] for i in training_data]).reshape(-1, INPUT_LAYER, 1)
        y = np.array([i[1] for i in training_data]).reshape(-1, 1)
        model.fit(X,y, n_epoch = 3, shuffle = True, run_id = self.filename)
        model.save(self.filename)
        return model

    def test_model(self, model):
        steps_arr = []
        scores_arr = []
        for _ in range(self.test_games):
            steps = 0

            game_memory = []

            player = Snake(0)
            player.init_positions()

            _, score = player.generate_observations()
            enemy = Snake(1)
            enemy.init_positions()
            food = Food([player, enemy])

            prev_observation = self.generate_observation(player, food, enemy)
            for _ in range(self.goal_steps):
                predictions = []
                for action in range(-1, 2):
                   predictions.append(model.predict(self.add_action_to_observation(prev_observation, action).reshape(-1, INPUT_LAYER, 1)))
                action = np.argmax(np.array(predictions))

                game_action = self.get_game_action(player, action - 1)
                if player.get_head_position()[1] == food.position[1]:
                    if (player.get_head_position()[0] - 1 == food.position[0]):
                        print("FORCE RIGHT")
                        game_action = 3  # RIGHT
                    elif (player.get_head_position()[0] + 1 == food.position[0]):
                        print("FORCE LEFT")
                        game_action = 2  # LEFT
                elif player.get_head_position()[0] == food.position[0]:
                    if (player.get_head_position()[1] - 1 == food.position[1]):
                        print("FORCE UP")
                        game_action = 0  # UP
                    elif (player.get_head_position()[1] + 1 == food.position[1]):
                        print("FORCE DOWN")
                        game_action = 1  # DOWN

                done, score, player, enemy, food = self.game_step(player, enemy, food, game_action)
                game_memory.append([prev_observation, action])
                if done:
                    print('-----')
                    print("Steps: {0}".format(steps))
                    print("Player: {0}".format(player.positions))
                    print("Enemy: {0}".format(enemy.positions))
                    print("Food: {0}, angle={1}".format(food.position, prev_observation[4]))
                    print("Obs: {0}".format(prev_observation))
                    print("Pred: {0}".format(predictions))
                    break
                else:
                    prev_observation = self.generate_observation(player, food, enemy)
                    steps += 1
            steps_arr.append(steps)
            scores_arr.append(score)
        print('Average steps:',mean(steps_arr))
        print(Counter(steps_arr))
        print('Average score:',mean(scores_arr))
        print(Counter(scores_arr))

    def visualise_game(self, model):

        pygame.init()
        clock = pygame.time.Clock()
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
        surface = pygame.Surface(screen.get_size())
        surface = surface.convert()
        drawGrid(surface)

        player = Snake(0)
        player.init_positions()
        enemy = Snake(1)

        position0 = (randint(0, SIZE - 1), randint(0, SIZE - 2))
        position1 = (position0[0], position0[0] + 1)
        enemy.positions = [position0, position1]

        food = Food([player, enemy])

        myfont = pygame.font.SysFont("bahnschrift", 20)

        # _, _, snake, food = game.start()
        prev_observation = self.generate_observation(player, food, enemy)
        while player.lives > 0:
            clock.tick(3)
            drawGrid(surface)

            precictions = []
            for action in range(-1, 2):
                precictions.append(
                    model.predict(self.add_action_to_observation(prev_observation, action).reshape(-1, INPUT_LAYER, 1)))
            action = np.argmax(np.array(precictions))

            game_action = self.get_game_action(player, action - 1)
            if player.get_head_position()[1] == food.position[1]:
                if (player.get_head_position()[0] - 1 == food.position[0]):
                    print("FORCE RIGHT")
                    game_action = 3  # RIGHT
                elif (player.get_head_position()[0] + 1 == food.position[0]):
                    print("FORCE LEFT")
                    game_action = 2  # LEFT
            elif player.get_head_position()[0] == food.position[0]:
                if (player.get_head_position()[1] - 1 == food.position[1]):
                    print("FORCE UP")
                    game_action = 0  # UP
                elif (player.get_head_position()[1] + 1 == food.position[1]):
                    print("FORCE DOWN")
                    game_action = 1  # DOWN

            done, score, player, enemy, food = self.game_step(player, enemy, food, game_action, enemy_ctrl=True)

            player.draw(surface)
            enemy.draw(surface)
            food.draw(surface)
            screen.blit(surface, (0, 0))
            text1 = myfont.render("Score: {0} Lives: {1}".format(player.score, player.lives), 1, (250, 250, 250))
            # text2 = myfont.render("Score AI {0}".format(enemy.score), 1, (250, 250, 250))
            screen.blit(text1, (5, 10))
            # screen.blit(text2, (SCREEN_WIDTH - 120, 10))
            pygame.display.update()

            if done:
                break
            else:
                prev_observation = self.generate_observation(player, food, enemy)
        return player.score

    def train(self):
        training_data = self.initial_population()
        nn_model = self.model()
        nn_model = self.train_model(training_data, nn_model)
        self.test_model(nn_model)

    def visualise(self):
        nn_model = self.model()
        nn_model.load(self.filename)
        score = self.visualise_game(nn_model)
        print("Snake Player Final Score:", score)

    def test(self):
        nn_model = self.model()
        nn_model.load(self.filename)
        self.test_model(nn_model)

if __name__ == "__main__":
    if len(sys.argv) <= 0:
        SnakeDQN().visualise()
    elif sys.argv[1] == "-train":
        SnakeDQN().train()
    elif sys.argv[1] == "-test":
        SnakeDQN().test()
    elif sys.argv[1] == "-vis":
        SnakeDQN().visualise()
