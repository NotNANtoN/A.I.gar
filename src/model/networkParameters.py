GPUS = 1
ALGORITHM = "None"

# Game
PELLET_SPAWN = True

# Experience replay:
MEMORY_CAPACITY = 75000
MEMORY_BATCH_LEN = 40 # Must be divisible by 4 atm due to experience replay
REPLAY_AFTER_X_STEPS = 0

# General RL:
ENABLE_SPLIT = False #TODO: these two do not have an effect yet, implement that they
ENABLE_EJECT = False #TODO: en/disable ejection and splitting for both continuous and discrete algorithms


# Q-learning
USE_POLICY_NETWORK = False
USE_TARGET = True # Otherwise td-error is used in value network. Using td-error is not horrible, but worse than target
EXP_REPLAY_ENABLED = False
GRID_VIEW_ENABLED = True
TARGET_NETWORK_STEPS = 10000
TARGET_NETWORK_MAX_STEPS = 10000 # 2000 performs worse than 5000. 20000 was a bit better than 5000. 20k was worse than 10k
DISCOUNT = 0.9 # 0.9 seems best so far. Better than 0.995 and 0.9999 . 0.5 and below performs much worse. 0.925 performs worse than 0.9

# Higher discount seems to lead to much more stable learning, less variance
TD = 0
Exploration = True
EPSILON = 0.1 if Exploration else 0 # Exploration rate. 0 == No Exploration
# epsilon set to 0 performs best so far... (keep in mind that it declines from 1 to 0 throughout the non-gui training
FRAME_SKIP_RATE = 9 # Frame skipping of around 5-10 leads to good performance. 15 and 30 lead to worse performance.
GRID_SQUARES_PER_FOV = 9 #11 is pretty good so far.
NUM_OF_GRIDS = 5

# Actor-critic:
MINIMUM_NOISE = 0.01
STEPS_TO_MIN_NOISE = 500000
ALPHA_POLICY = 0.0005
OPTIMIZER_POLICY = "Adam"
ACTIVATION_FUNC_HIDDEN_POLICY = "relu"
HIDDEN_LAYER_1_POLICY = 75
HIDDEN_LAYER_2_POLICY = 0
HIDDEN_LAYER_3_POLICY = 0

#LSTM
NEURON_TYPE = "LSTM"
ACTIVATION_FUNC_LSTM = "tanh"
UPDATE_LSTM_MOVE_NETWORK = 1
TRACE_MIN = 7 # The minimum amount of traces that are not trained on, as they have insufficient hidden state info
MEMORY_TRACE_LEN = 10 # The length of memory traces retrieved via exp replay


#ANN
DROPOUT_RATE= 0 #TODO: not yet implemented at all, but might be interesting
ALPHA = 0.00025 #Learning rate. Marco recommended to try lower learning rates too, decrease by factor of 10 or 100
OPTIMIZER = "Adam" #SGD has much worse performance
ACTIVATION_FUNC_HIDDEN = 'elu' #'relu' is better than sigmoid, but gives more variable results. we should try elu
ACTIVATION_FUNC_OUTPUT = 'linear'

#Layer neurons
STATE_REPR_LEN = GRID_SQUARES_PER_FOV * GRID_SQUARES_PER_FOV * NUM_OF_GRIDS + 2
HIDDEN_LAYER_1 = 500
HIDDEN_LAYER_2 = 500
HIDDEN_LAYER_3 = 500
# More hidden layers lead to improved performance. Best so far three hidden layers with 100 neurons each and relu activation
