import random

import gym
import numpy as np
from tqdm import tqdm
import torch as T

from p_gradient.actor_critic import ActorCriticAgent
from p_gradient.reinforce import PolicyGradientAgent
from p_gradient.utils import plotLearning, getObsGrid, render, getObsSmall, enemyMovement
from main import Snake, Food, handleSnakeCollisions, handleFoodEating, DEATH_PENALTY, SIZE

OBS_GRID_SIZE = 5
num_episodes = 200
SHOW_EVERY = 50
agent_type = "AC"
obs_type = "Small"
modelname = f"{agent_type}-{obs_type}-{num_episodes}-New"


# following gym environment guidelines
def step(snakes: list, food: Food, action):
        # First snake must be the player7
        player = snakes[0]
        player.action([action], "AC")
        done = False
        obs = "Wrong Input"
        # currently overall reward increases every time food is eaten
        handleFoodEating(snakes, food)
        # the current state after move has been done
        # could be the whole fucking grid or maybe just part of it around snakes head
        # for now let me try making it a 5x5 grid around its head
        if obs_type == "Grid":
            obs = getObsGrid(snakes, food, OBS_GRID_SIZE, fullGrid=False)
        if obs_type == "Small":
            obs = getObsSmall(snakes, food)
        if player.reward == -DEATH_PENALTY:
            done = True
        info = ""

        return obs, player.reward, done, info


if __name__ == '__main__':
    if agent_type == "PG":
        agent = PolicyGradientAgent(lr=0.0005, input_dims=6, GAMMA=0.99,
                                n_actions=4, layer1_size=128, layer2_size=128)
        agent.policy = T.load("model/PG-Small-2001")
    if agent_type == "AC":
        agent = ActorCriticAgent(alpha=0.001, input_dims=6, gamma=0.99,
                              layer1_size=128, layer2_size=64, n_actions=4)
        agent.actor_critic = T.load("model/AC-Small-2001")

    score_history = []
    score = 0
    n_steps_history = []
    for i in tqdm(range(num_episodes)):
        player = Snake(0)
        enemy = Snake(1)
        food = Food([player, enemy])
        done = False
        score = 0
        # returns a numpy array of the state we care about
        if obs_type == "Grid":
            observation = getObsGrid(snakes=[player], food=food, size=OBS_GRID_SIZE, fullGrid=False)
        if obs_type == 'Small':
            observation = getObsSmall([player, enemy], food)
        n_steps = 0
        while player.lives > 0 and enemy.lives > 0 and n_steps < 1000:
            n_steps += 1
            # action needs to be either 0,1,2 or 3
            action = agent.choose_action(observation)
            observation_, reward, done, info = step(snakes=[player, enemy], food=food, action=action)
            enemyMovement(enemy=enemy,food=food)
            if agent_type == "AC":
                agent.learn(observation, reward, observation_, done) # For Actor-Critic
            if agent_type == "PG":
                agent.store_rewards(reward) # For REINFORCE
            observation = observation_
            score += reward
        score_history.append(score)
        n_steps_history.append(n_steps)
        if i % SHOW_EVERY == 0:
            print(f"\n {i} ep mean: {np.mean(score_history[-SHOW_EVERY:])}")
            print(f"{i} steps mean: {np.mean(n_steps_history[-SHOW_EVERY:])}")
        if i % (SHOW_EVERY) == 0:
            render(agent, obs_type, 5)

    print("Max Score:", np.max(score_history))
    print("Median Score:", np.median(score_history))
    print("Mean Score:", np.mean(score_history))
    filename = modelname + '-scores.png'
    filename2 = modelname + '-steps.png'
    plotLearning(score_history, filename="tests/" + filename, window=50, ylabel="Scores")
    plotLearning(n_steps_history, filename="tests/" + filename2, window=50, ylabel="# of Steps")







