#Snake PvP




#### Running ε-Greedy Q-Learning Agent

*All parameters need to be specified at the top of each file*

+ `python qlearning.py ` Trains Q-Learning algorithm 
+ `python play_agent.py ` Allows you to play against a Q-Learning Algorithm using the arrow keys
+ `python test_q.py ` Runs the test environment on Q-Learning of 200 episodes with 5 lives

#### Running Advantage Actor Critic Agent

Need to navigate to the `p_gradient` directory first<br/>
*All parameters need to be specified at the top of each file*

+ `python pg_main.py ` Trains actor critic algorithm
+ `python test.py ` Runs the test environment on the actor critic algorithm<br/>

Supporting files:<br/>
`actor_critic.py ` Holds the actor critic algorithm and agent<br/>
`reinforce.py ` Holds the reinforce algorithm and agent (not mentioned due to irrelevant results)<br/>
`utils.py ` Holds helper methods for the PG algorithms<br/>
`graphs` Contains all PG training graphs<br/>
`model` Contains all saved models <br/>
`tests` Contains all graphic results of the testing environment<br/>



#### Running DQN Agent (7-input neural network)
###### Tested w. Python 3.8


Move to the `dqn` folder

Set `FILENAME` in **deep_qlearining.py** to the tensorflow model that you want to use or create (must be in the **tfdata/** dir)


Set any other constants in **SnakeGame.py** or line 24 of **deep_qlearning.py**


+ `python deep_qlearning.py -vis` runs one game with the set model against the simple search agent and shows the game as it is played
+ `python deep_qlearning.py -train` trains a new model and creates the model files (will override model files of the same name)
+ `python deep_qlearning.py -test` tests the given model
