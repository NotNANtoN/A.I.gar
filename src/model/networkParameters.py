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
FRAME_SKIP_RATE = 12 # Frame skipping of around 5-10 leads to good performance. 15 and 30 lead to worse performance.
MAX_TRAINING_STEPS = 125000
MAX_SIMULATION_STEPS = MAX_TRAINING_STEPS * (FRAME_SKIP_RATE + 1)
TRAINING_WAIT_TIME = 1 # Only train after the wait time is over to maximize gpu effectiveness. 1 == train every step
ENABLE_SPLIT = False
ENABLE_EJECT = False
NOISE_AT_HALF_TRAINING = 0.02
NOISE_DECAY = NOISE_AT_HALF_TRAINING ** (1 / (MAX_TRAINING_STEPS / 2))
#Reward function:
REWARD_SCALE = 2
DEATH_TERM = 0
DEATH_FACTOR = 1

# State representation parameters:
NORMALIZE_GRID_BY_MAX_MASS = False
ENABLE_PELLET_GRID = True
ENABLE_SELF_GRID = True
ENABLE_WALL_GRID = True if NUM_GREEDY_BOTS + NUM_NN_BOTS > 1 else False
ENABLE_VIRUS_GRID = False
ENABLE_ENEMY_GRID = True if NUM_GREEDY_BOTS + NUM_NN_BOTS > 1 else False
ENABLE_SIZE_GRID = False
USE_FOVSIZE = True
USE_TOTALMASS = True
GRID_SQUARES_PER_FOV = 11 #11 is pretty good so far.
NUM_OF_GRIDS = ENABLE_PELLET_GRID + ENABLE_SELF_GRID + ENABLE_WALL_GRID + ENABLE_VIRUS_GRID + ENABLE_ENEMY_GRID + ENABLE_SIZE_GRID
STATE_REPR_LEN = GRID_SQUARES_PER_FOV * GRID_SQUARES_PER_FOV * NUM_OF_GRIDS + USE_FOVSIZE + USE_TOTALMASS

INITIALIZER = "glorot_uniform" #"Default" or "glorot_uniform" or "glorot_normal"
NEURON_TYPE = "MLP"
HIDDEN_ALL = 256
HIDDEN_LAYER_1 = HIDDEN_ALL if HIDDEN_ALL else 256
HIDDEN_LAYER_2 = HIDDEN_ALL if HIDDEN_ALL else 256
HIDDEN_LAYER_3 = HIDDEN_ALL if HIDDEN_ALL else 256


# Q-learning
ALPHA = 0.0001
SQUARE_ACTIONS = True
NUM_ACTIONS = 25
OPTIMIZER = "Adam" #SGD has much worse performance
ACTIVATION_FUNC_HIDDEN = 'relu'
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
TEMPERATURE = 7
TEMPERATURE_AT_END_TRAINING = 0.0025
TEMPERATURE_DECAY = TEMPERATURE_AT_END_TRAINING ** (1 / MAX_TRAINING_STEPS)

# Actor-critic:
SOFT_TARGET_UPDATES = False
CACLA_CRITIC_LAYERS = (250, 250, 250)
CACLA_CRITIC_ALPHA  = 0.0001
CACLA_ACTOR_LAYERS  = (100, 100, 100)
CACLA_ACTOR_ALPHA   = 0.00005
ACTOR_CRITIC_TYPE = "CACLA" # "Standard"/"CACLA" / "DPG". Standard multiplies gradient by tdE, CACLA only updates once for positive tdE
CACLA_UPDATE_ON_NEGATIVE_TD = False
POLICY_OUTPUT_ACTIVATION_FUNC = "sigmoid" # "relu_max" or "sigmoid"
ACTOR_REPLAY_ENABLED = True
GAUSSIAN_NOISE = 1 # Initial noise
ALPHA_POLICY = 0.00005
OPTIMIZER_POLICY = "Adam"
ACTIVATION_FUNC_HIDDEN_POLICY = "relu"

# Deterministic Policy Gradient (DPG):
DPG_TAU                    = 0.05 # How quickly the weights of the target networks are updated
DPG_CRITIC_LAYERS          = (250, 250, 250)
DPG_CRITIC_ALPHA           = 0.0002
DPG_CRITIC_FUNC            = "relu"
DPG_CRITIC_WEIGHT_DECAY    = 0 #0.001 L2 weight decay parameter. Set to 0 to disable
DPG_ACTOR_LAYERS           = (100, 100, 100)
DPG_ACTOR_ALPHA            = 0.00001
DPG_ACTOR_FUNC             = "relu"
DPG_Q_VAL_INCREASE         = 1
DPG_FEED_ACTION_IN_LAYER   = 1
DPG_USE_CACLA              = False
DPG_USE_DPG_ACTOR_TRAINING = True
DPG_USE_TARGET_MODELS      = True

# LSTM
ACTIVATION_FUNC_LSTM = "sigmoid"
UPDATE_LSTM_MOVE_NETWORK = 1
TRACE_MIN = 1 # The minimum amount of traces that are not trained on, as they have insufficient hidden state info
MEMORY_TRACE_LEN = 15 # The length of memory traces retrieved via exp replay

# CNN
CNN_REPRESENTATION = False
# Handcraft representation
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

# Pixel representation
CNN_PIXEL_REPRESENTATION = False
CNN_PIXEL_INCEPTION = False
# CNN_PIXEL_USE_LAYER_1 = False
# CNN_PIXEL_LAYER_1_FILTER_NUM = 10
# CNN_PIXEL_SIZE_OF_INPUT_DIM_1 = 200
#
# CNN_PIXEL_USE_LAYER_2 = True
# CNN_PIXEL_LAYER_2_FILTER_NUM = 8
# CNN_PIXEL_SIZE_OF_INPUT_DIM_2 = 84
#
# CNN_PIXEL_LAYER_3_FILTER_NUM = 16
# CNN_PIXEL_SIZE_OF_INPUT_DIM_3 = 84
