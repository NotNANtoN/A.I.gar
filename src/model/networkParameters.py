
MAX_UPDATES = 3000000
GPUS = 1

# Experience replay:
MEMORY_CAPACITY = 50000
MEMORIES_PER_UPDATE = 40 # Must be divisible by 4 atm due to experience replay

# Q-learning
EXP_REPLAY_ENABLED = True
GRID_VIEW_ENABLED = True
TARGET_NETWORK_STEPS = 1000
TARGET_NETWORK_MAX_STEPS = 1000
DISCOUNT = 0.97
Exploration = True

EPSILON = 0.1 if Exploration else 0 # Exploration rate. 0 == No Exploration
FRAME_SKIP_RATE = 2
GRID_SQUARES_PER_FOV = 12 # is modified by the user later on anyways
NUM_OF_GRIDS = 5

#ANN
ALPHA = 0.005 #Learning rate
OPTIMIZER = "SGD"
ACTIVATION_FUNC_HIDDEN = 'sigmoid'
ACTIVATION_FUNC_OUTPUT = 'linear'

#Layer neurons
STATE_REPR_LEN = GRID_SQUARES_PER_FOV * GRID_SQUARES_PER_FOV * NUM_OF_GRIDS
HIDDEN_LAYER_1 = 200
HIDDEN_LAYER_2 = 100
HIDDEN_LAYER_3 = 50
