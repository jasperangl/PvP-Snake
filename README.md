### Running DQN Agent (7-input neural network)
###### Tested w. Python 3.8

Move to the `dqn` folder

Set `FILENAME` in **deep_qlearining.py** to the tensorflow model that you want to use or create (must be in the **tfdata/** dir)


Set any other constants in **SnakeGame.py** or line 24 of **deep_qlearning.py**


+ `python deep_qlearning.py -vis` runs one game with the set model against the simple search agent and shows the game as it is played
+ `python deep_qlearning.py -train` trains a new model and creates the model files (will override model files of the same name)
+ `python deep_qlearning.py -test` tests the given model
