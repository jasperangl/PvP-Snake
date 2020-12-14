import sys

from dqn.SnakeGame import SnakeGame, MOVE_PENALTY, FOOD_REWARD, DEATH_PENALTY
from random import randint
import numpy as np
import tflearn
import math
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean
from collections import Counter
import matplotlib.pyplot as plt

BASIC_SEARCH_EPSILON = 0.5

OUT_DIR = 'tfdata/'
FILENAME = 'dqn-5k-games-5k-ep(4)'
FILETYPE = '.tflearn'

N_INPUTS = 7


class SnakeDQN:
    def __init__(self, initial_games = 5000, test_games = 200, goal_steps = 2000, lr = 1e-2,
                 filename = OUT_DIR + FILENAME + FILETYPE):
        self.initial_games = initial_games
        self.test_games = test_games
        self.goal_steps = goal_steps
        self.lr = lr
        self.filename = filename
        self.vectors_and_keys = [
                [[-1, 0], 0],
                [[0, 1], 1],
                [[1, 0], 2],
                [[0, -1], 3]
                ]

    def initial_population(self):
        training_data = []
        for _ in range(self.initial_games):
            game = SnakeGame(enemy_epsilon=BASIC_SEARCH_EPSILON)
            _, prev_score, player, enemy, food = game.start()
            prev_observation = self.generate_observation(player, enemy, food)
            prev_food_distance = self.get_food_distance(player, food)
            prev_enemy_distance = self.get_food_distance(player, self.get_snake_center(enemy))
            for _ in range(self.goal_steps):
                action, game_action = self.generate_action(player)
                lives, score, player, enemy, food = game.step(game_action)
                if game.is_done():
                    training_data.append([self.add_action_to_observation(prev_observation, action), -1])
                    break
                else:
                    food_distance = self.get_food_distance(player, food)
                    enemy_distance = self.get_food_distance(player, enemy[0])
                    if score > prev_score \
                            or food_distance < prev_food_distance \
                            or enemy_distance > prev_enemy_distance:
                        training_data.append([self.add_action_to_observation(prev_observation, action), 1])
                    else:
                        training_data.append([self.add_action_to_observation(prev_observation, action), 0])
                    prev_observation = self.generate_observation(player, enemy, food)
                    prev_food_distance = food_distance
                    prev_enemy_distance = enemy_distance
        return training_data

    def generate_action(self, snake):
        action = randint(0,2) - 1
        return action, self.get_game_action(snake, action)

    def get_game_action(self, snake, action):
        snake_direction = self.get_snake_direction_vector(snake)
        new_direction = snake_direction
        if action == -1:
            new_direction = self.turn_vector_to_the_left(snake_direction)
        elif action == 1:
            new_direction = self.turn_vector_to_the_right(snake_direction)
        for pair in self.vectors_and_keys:
            if pair[0] == new_direction.tolist():
                game_action = pair[1]
        return game_action


    def get_snake_center(self, snake):
        xs = [p[0] for p in snake]
        ys = [p[1] for p in snake]
        center_y = np.sum(ys)/len(snake)
        center_x = np.sum(xs)/len(snake)
        return (center_x, center_y)

    def generate_observation(self, snake, enemy, food):
        '''
        These observations have 6 elements.
        :return: [barrier_left, barrier_front, barrier_right, apple_angle, enemy_head_angle, enemy_center_angle]
        '''
        snake_direction = self.get_snake_direction_vector(snake)
        food_direction = self.get_food_direction_vector(snake, food)
        enemy_direction = self.get_food_direction_vector(snake, enemy[0])
        enemy_center_direction = self.get_food_direction_vector(snake, self.get_snake_center(enemy))

        barrier_left = self.is_direction_blocked(snake, self.turn_vector_to_the_left(snake_direction))
        barrier_front = self.is_direction_blocked(snake, snake_direction)
        barrier_right = self.is_direction_blocked(snake, self.turn_vector_to_the_right(snake_direction))

        food_angle = self.get_angle(snake_direction, food_direction)
        enemy_angle = self.get_angle(snake_direction, enemy_direction)
        enemy_cog_angle = self.get_angle(snake_direction, enemy_center_direction)
        enemy_cog_distance = self.get_food_distance(snake, self.get_snake_center(enemy))

        return np.array([int(barrier_left),
                         int(barrier_front),
                         int(barrier_right),
                         food_angle,
                         enemy_angle,
                         # enemy_cog_angle,
                         enemy_cog_distance
                         ])

    def add_action_to_observation(self, observation, action):
        return np.append([action], observation)

    def get_snake_direction_vector(self, snake):
        return np.array(snake[0]) - np.array(snake[1])

    def get_food_direction_vector(self, snake, food):
        return np.array(food) - np.array(snake[0])

    def normalize_vector(self, vector):
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector

        return vector / np.linalg.norm(vector)

    def get_food_distance(self, snake, food):
        return np.linalg.norm(self.get_food_direction_vector(snake, food))

    def is_direction_blocked(self, snake, direction):
        point = np.array(snake[0]) + np.array(direction)
        return point.tolist() in snake[:-1] or point[0] == 0 or point[1] == 0 or point[0] == 21 or point[1] == 21

    def turn_vector_to_the_left(self, vector):
        return np.array([-vector[1], vector[0]])

    def turn_vector_to_the_right(self, vector):
        return np.array([vector[1], -vector[0]])

    def get_angle(self, a, b):
        a = self.normalize_vector(a)
        b = self.normalize_vector(b)
        return math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]) / math.pi

    def model(self):
        network = input_data(shape=[None, N_INPUTS, 1], name='input')
        network = fully_connected(network, N_INPUTS * N_INPUTS, activation='relu')
        network = fully_connected(network, 1, activation='linear')
        network = regression(network, optimizer='adam', learning_rate=self.lr, loss='mean_square', name='target')
        model = tflearn.DNN(network, tensorboard_dir='../log')
        return model

    def train_model(self, training_data, model):
        X = np.array([i[0] for i in training_data]).reshape(-1, N_INPUTS, 1)
        y = np.array([i[1] for i in training_data]).reshape(-1, 1)
        model.fit(X,y, n_epoch = 3, shuffle = True, run_id = self.filename)
        model.save(self.filename)
        return model

    def test_model(self, model, generate_graph=False):
        steps_arr = []
        scores_arr = []
        player_len_arr = []
        enemy_len_arr = []
        for ngame in range(self.test_games):
            steps = 0
            game_memory = []
            game = SnakeGame(enemy_epsilon=BASIC_SEARCH_EPSILON)
            _, score, player, enemy, food = game.start()
            prev_observation = self.generate_observation(player, enemy, food)
            for _ in range(self.goal_steps):
                predictions = []
                for action in range(-1, 2):
                   predictions.append(
                       model.predict(self.add_action_to_observation(prev_observation, action).reshape(-1, N_INPUTS, 1)))
                action = np.argmax(np.array(predictions))
                game_action = self.get_game_action(player, action - 1)
                lives, score, player, enemy, food = game.step(game_action)
                game_memory.append([prev_observation, score, action]) # [4] -> score, [5] -> action
                if game.is_done():
                    print('--- {0} / {1} ---'.format(ngame, self.test_games))
                    print('{0} in {1} steps'.format(round(score, 2), steps))
                    print(player)
                    print(enemy)
                    print(food)
                    print(prev_observation)
                    print(predictions)
                    break
                else:
                    prev_observation = self.generate_observation(player, enemy, food)
                    steps += 1
            steps_arr.append(steps)
            scores_arr.append(score)
            player_len_arr.append(len(player))
            enemy_len_arr.append(len(enemy))
        print('Average steps:',mean(steps_arr))
        print(Counter(steps_arr))
        print('Average score:',mean(scores_arr))
        print(Counter(scores_arr))

        if generate_graph:
            self.generate_graph([scores_arr, player_len_arr, enemy_len_arr])

    def visualise_game(self, model):
        game = SnakeGame(enemy_epsilon=BASIC_SEARCH_EPSILON, gui = True)
        _, _, player, enemy, food = game.start()
        prev_observation = self.generate_observation(player, enemy, food)
        for _ in range(self.goal_steps):
            precictions = []
            for action in range(-1, 2):
               precictions.append(
                   model.predict(self.add_action_to_observation(prev_observation, action).reshape(-1, N_INPUTS, 1)))
            action = np.argmax(np.array(precictions))
            game_action = self.get_game_action(player, action - 1)
            lives, _, player, enemy, food  = game.step_render(game_action)
            if game.is_done():
                break
            else:
                prev_observation = self.generate_observation(player, enemy, food)

    def train(self, generate_graph=False):
        training_data = self.initial_population()
        nn_model = self.model()
        nn_model = self.train_model(training_data, nn_model)
        self.test_model(nn_model, generate_graph=generate_graph)

    def generate_graph(self, test_results):
        SHOW_EVERY = 1  # how often to play through env visually.

        test_scores = test_results[0]
        test_len = test_results[1]
        test_enemy = test_results[2]

        print("\n==========")
        mean_score = np.mean(test_scores[-SHOW_EVERY:])
        max_score = np.max(test_scores[-SHOW_EVERY:])
        moving_avg = np.convolve(test_scores, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')
        print(f"SCORES (per {SHOW_EVERY}):, mean: {mean_score}, max: {max_score}")

        mean_len = np.mean(test_len[-SHOW_EVERY:])
        max_len = np.max(test_len[-SHOW_EVERY:])
        moving_avg2 = np.convolve(test_len, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')
        print(f"LENGTH (per {SHOW_EVERY}):, mean: {mean_len}, max: {max_len}")

        mean_elen = np.mean(test_enemy[-SHOW_EVERY:])
        max_elen = np.max(test_enemy[-SHOW_EVERY:])
        moving_avg3 = np.convolve(test_enemy, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')
        print(f"ENEMY LENGTH (m.avg per {SHOW_EVERY}):, mean: {mean_elen}, max: {max_elen}")

        plt.plot([i for i in range(len(moving_avg))], moving_avg, 'b--',
                 [j for j in range(len(moving_avg2))], moving_avg2, 'g--',
                 [k for k in range(len(moving_avg3))], moving_avg3, 'r--'
                 )
        plt.ylabel(f"Score (m.avg per {SHOW_EVERY})")
        plt.xlabel(f"episode (per {SHOW_EVERY})")
        plt.title(
            f"Player Training({self.test_games}) avg:{round(mean_score,1)} max:{round(max_score,1)}:\n"
            f" [Rewards: food:+{FOOD_REWARD} death:-{DEATH_PENALTY}];\n"
            f" [enemy_e:{BASIC_SEARCH_EPSILON}, inputs:{N_INPUTS}, lr:{self.lr}]")
        plt.savefig('graphs/for-paper(2)-' + f"{FILENAME}-scores.png")
        plt.show()

    def visualise(self):
        nn_model = self.model()
        nn_model.load(self.filename)
        self.visualise_game(nn_model)

    def test(self,generate_graph=False):
        nn_model = self.model()
        nn_model.load(self.filename)
        self.test_model(nn_model,generate_graph=generate_graph)

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        MOVE_PENALTY = 0
        SnakeDQN().visualise()
    elif sys.argv[1] == "-train":
        MOVE_PENALTY = 0.1
        SnakeDQN().train(generate_graph=True)
    elif sys.argv[1] == "-test":
        MOVE_PENALTY = 0
        SnakeDQN().test(generate_graph=True)
    elif sys.argv[1] == "-vis":
        SnakeDQN().visualise()
