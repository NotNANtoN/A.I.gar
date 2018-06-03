GPUS = 1
ALGORITHM = "None"

Default = False

# Game
PELLET_SPAWN = True
NUM_GREEDY_BOTS = 0
NUM_NN_BOTS = 1

# Experience replay:
MEMORY_CAPACITY = 75000 #10000 is worse
MEMORY_BATCH_LEN = 32

# General RL:
TRAINING_WAIT_TIME = 1 # Only train after the wait time is over to maximize gpu effectiveness. 1 == train every step
ENABLE_SPLIT = False
ENABLE_EJECT = False
NEURON_TYPE = "MLP"
FRAME_SKIP_RATE = 12 # Frame skipping of around 5-10 leads to good performance. 15 and 30 lead to worse performance.
GRID_SQUARES_PER_FOV = 11 #11 is pretty good so far.
NUM_OF_GRIDS = 5
MAX_TRAINING_STEPS = 100000
MAX_SIMULATION_STEPS = MAX_TRAINING_STEPS * (FRAME_SKIP_RATE + 1)
NOISE_AT_HALF_TRAINING = 0.01
NOISE_DECAY = NOISE_AT_HALF_TRAINING ** (1 / (MAX_TRAINING_STEPS / 2))
INITIALIZER = "Default" # "glorot_uniform" or "glorot_normal"

# Q-learning
ALPHA = 0.0001
NUM_ACTIONS = 8 # That number plus 1 (for standing still)
OPTIMIZER = "Adam" #SGD has much worse performance
ACTIVATION_FUNC_HIDDEN = 'elu' 
ELU_ALPHA = 1 # TODO: only works for Q-learning so far. Test if it is useful, if so implement for others too
ACTIVATION_FUNC_OUTPUT = 'linear'
EXP_REPLAY_ENABLED = True
GRID_VIEW_ENABLED = True
TARGET_NETWORK_STEPS = 1500
DISCOUNT = 0.85 # Higher discount seems to lead to much more stable learning, less variance
USE_ACTION_AS_INPUT = False
TD = 0
Exploration = True
EPSILON = 1 if Exploration else 0 # Exploration rate. 0 == No Exploration
EXPLORATION_STRATEGY = "e-Greedy" # "Boltzmann" or "e-Greedy"
TEMPERATURE = 5
TEMPERATURE_AT_END_TRAINING = 0.005
TEMPERATURE_DECAY = TEMPERATURE_AT_END_TRAINING ** (1 / MAX_TRAINING_STEPS)

# Actor-critic:
ACTOR_CRITIC_TYPE = "CACLA" # "Standard"/"CACLA". Standard multiplies gradient by tdE, CACLA only updates once for positive tdE
CACLA_UPDATE_ON_NEGATIVE_TD = False
POLICY_OUTPUT_ACTIVATION_FUNC = "sigmoid" # "relu_max" or "sigmoid"
ACTOR_REPLAY_ENABLED = True
GAUSSIAN_NOISE = 1 # Initial noise
ALPHA_POLICY = 0.00005
OPTIMIZER_POLICY = "Adam"
ACTIVATION_FUNC_HIDDEN_POLICY = "relu"
HIDDEN_ALL_POLICY = 100
HIDDEN_LAYER_1_POLICY = HIDDEN_ALL_POLICY if HIDDEN_ALL_POLICY else 100
HIDDEN_LAYER_2_POLICY = HIDDEN_ALL_POLICY if HIDDEN_ALL_POLICY else 100
HIDDEN_LAYER_3_POLICY = HIDDEN_ALL_POLICY if HIDDEN_ALL_POLICY else 100

# LSTM
ACTIVATION_FUNC_LSTM = "sigmoid"
UPDATE_LSTM_MOVE_NETWORK = 1
TRACE_MIN = 1 # The minimum amount of traces that are not trained on, as they have insufficient hidden state info
MEMORY_TRACE_LEN = 15 # The length of memory traces retrieved via exp replay

# CNN
CNN_REPRESENTATION = False
CNN_PIXEL_REPRESENTATION = False
CNN_USE_LAYER_1 = False
CNN_LAYER_1 = 5
CNN_LAYER_1_STRIDE = 2
CNN_LAYER_1_FILTER_NUM = 10
CNN_SIZE_OF_INPUT_DIM_1 = 200

CNN_USE_LAYER_2 = True
CNN_LAYER_2 = 8
CNN_LAYER_2_STRIDE = 4
CNN_LAYER_2_FILTER_NUM = 16
CNN_SIZE_OF_INPUT_DIM_2 = 84

CNN_LAYER_3 = 4
CNN_LAYER_3_STRIDE = 2
CNN_LAYER_3_FILTER_NUM = 32
CNN_SIZE_OF_INPUT_DIM_3 = 42

CNN_POOLING_TYPE = "Max"



#Layer neurons
STATE_REPR_LEN = GRID_SQUARES_PER_FOV * GRID_SQUARES_PER_FOV * NUM_OF_GRIDS + 2
HIDDEN_ALL = 500
HIDDEN_LAYER_1 = HIDDEN_ALL if HIDDEN_ALL else 500
HIDDEN_LAYER_2 = HIDDEN_ALL if HIDDEN_ALL else 500
HIDDEN_LAYER_3 = HIDDEN_ALL if HIDDEN_ALL else 500
# More hidden layers lead to improved performance. Best so far three hidden layers with 100 neurons each and relu activation
