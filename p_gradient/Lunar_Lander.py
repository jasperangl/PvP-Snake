import random

import gym
import numpy as np
from tqdm import tqdm
import torch as T

from p_gradient.actor_critic import ActorCriticAgent
from p_gradient.reinforce import PolicyGradientAgent
from p_gradient.utils import plotLearning, getObsGrid, render, getObsSmall
from main import Snake, Food, handleSnakeCollisions, handleFoodEating, DEATH_PENALTY, SIZE

OBS_GRID_SIZE = 5
num_episodes = 5000
SHOW_EVERY = 2000
obs_type = "Grid"
modelname = f"AC-{obs_type}-{num_episodes}"

if __name__ == '__main__':
    #agent = PolicyGradientAgent(lr=0.001, input_dims=OBS_GRID_SIZE**2, GAMMA=0.99,
                               # n_actions=4, layer1_size=512, layer2_size=512)
    agentAC = ActorCriticAgent(alpha=0.001, input_dims=OBS_GRID_SIZE**2, gamma=0.99, layer1_size=128, layer2_size=64, n_actions=4)

    # following gym environment guidelines
    def step(snakes: list, food: Food, action, obs_type: str):
        # First snake must be the player7
        player = snakes[0]
        player.action([action], "PG")
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


    score_history = []
    score = 0
    n_steps_history = []
    for i in tqdm(range(num_episodes)):
        player = Snake(0)
        enemy = Snake(1)
        enemy.positions = [(random.randint(0, SIZE - 1), random.randint(0, SIZE - 1))]
        food = Food([player])
        done = False
        score = 0
        # returns a numpy array of the state we care about
        observation = getObsGrid(snakes=[player], food=food, size=OBS_GRID_SIZE, fullGrid=False)
        #observation = getObsSmall([player, enemy], food)
        n_steps = 0
        while not done and n_steps < 100:
            n_steps += 1
            # action needs to be either 0,1,2 or 3
            action = agentAC.choose_action(observation)
            observation_, reward, done, info = step(snakes=[player, enemy], food=food, action=action, obs_type=obs_type)
            agentAC.learn(observation, reward, observation_, done) # For Actor-Critic
            #agent.store_rewards(reward) # For REINFORCE
            observation = observation_
            score += reward
        score_history.append(score)
        n_steps_history.append(n_steps)
        if i % SHOW_EVERY == 0:
            #print(f"on #{i}, epsilon is {lr}")
            print(f"\n {i} ep mean: {np.mean(score_history[-SHOW_EVERY:])}")
            print(f"{i} steps mean: {np.mean(n_steps_history[-SHOW_EVERY:])}")
        if i % (SHOW_EVERY) == 0:
            render(agentAC, obs_type, board_size=OBS_GRID_SIZE)

    T.save(agentAC.actor_critic, "model/" + modelname)

    print("Max Score:", np.max(score_history))
    print("Median Score:", np.median(score_history))
    print("Mean Score:", np.mean(score_history))
    filename = modelname + '-scores.png'
    filename2 = modelname + '-steps.png'
    plotLearning(score_history, filename="graphs/" + filename, window=50, ylabel="Scores")
    plotLearning(n_steps_history, filename="graphs/" + filename2, window=50, ylabel="# of Steps")







