GPUS = 1

# Experience replay:
MEMORY_CAPACITY = 75000
MEMORIES_PER_UPDATE = 40 # Must be divisible by 4 atm due to experience replay
REPLAY_AFTER_X_STEPS = 0

# Q-learning
USE_POLICY_NETWORK = False
USE_TARGET = True # Otherwise td-error is used in value network
EXP_REPLAY_ENABLED = True
GRID_VIEW_ENABLED = True
TARGET_NETWORK_STEPS = 1000
TARGET_NETWORK_MAX_STEPS = 5000
DISCOUNT = 0.9 # 0.9 seems better so far than 0.995

Exploration = True
EPSILON = 0.05 if Exploration else 0 # Exploration rate. 0 == No Exploration
EPSILON_DECREASE_RATE = 1
FRAME_SKIP_RATE = 2
GRID_SQUARES_PER_FOV = 12
NUM_OF_GRIDS = 3

#ANN
NEURON_TYPE = "MLP" #"LSTM" lstm does not work yet
ALPHA = 0.00025 #Learning rate
OPTIMIZER = "Adam"
ACTIVATION_FUNC_HIDDEN = 'sigmoid' #'tanh' seems to perform worse than sigmoid
ACTIVATION_FUNC_OUTPUT = 'linear'

#Layer neurons
STATE_REPR_LEN = GRID_SQUARES_PER_FOV * GRID_SQUARES_PER_FOV * NUM_OF_GRIDS + 2
HIDDEN_LAYER_1 = 100 # Best so far is 200 neurons in first two layers
HIDDEN_LAYER_2 = 0
HIDDEN_LAYER_3 = 0
