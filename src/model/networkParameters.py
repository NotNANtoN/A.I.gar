GPUS = 1
ALGORITHM = "None"

Default = False

# Game
PELLET_SPAWN = True
VIRUS_SPAWN = False
RESET_LIMIT = 10000
EXPORT_POINT_AVERAGING = 500
NUM_GREEDY_BOTS = 0
NUM_NN_BOTS = 1

# Experience replay:
EXP_REPLAY_ENABLED = True
PRIORITIZED_EXP_REPLAY_ENABLED = True
MEMORY_CAPACITY = 75000
MEMORY_BATCH_LEN = 32
MEMORY_ALPHA = 0.6
MEMORY_BETA = 0.4

# General RL:
FRAME_SKIP_RATE_BOTS = 8
FRAME_SKIP_RATE = 12 if NUM_GREEDY_BOTS + NUM_NN_BOTS == 1 else FRAME_SKIP_RATE_BOTS
MAX_TRAINING_STEPS = 300000
MAX_SIMULATION_STEPS = MAX_TRAINING_STEPS * (FRAME_SKIP_RATE + 1)
TRAINING_WAIT_TIME = 1 # Only train after the wait time is over to maximize gpu effectiveness. 1 == train every step
ENABLE_SPLIT = False
ENABLE_EJECT = False
# Noise and Exploration:
NOISE_TYPE = "Gaussian"  # "Gaussian" / "Orn-Uhl"
GAUSSIAN_NOISE = 1 # Initial noise
NOISE_AT_HALF_TRAINING = 0.02
NOISE_DECAY = NOISE_AT_HALF_TRAINING ** (1 / (MAX_TRAINING_STEPS / 2))
ORN_UHL_THETA = 0.15
ORN_UHL_DT = 0.01
ORN_UHL_MU = 0
Exploration = True
EPSILON = 1 if Exploration else 0 # Exploration rate. 0 == No Exploration
EXPLORATION_STRATEGY = "e-Greedy" # "Boltzmann" or "e-Greedy"
TEMPERATURE = 7
TEMPERATURE_AT_END_TRAINING = 0.0025
TEMPERATURE_DECAY = TEMPERATURE_AT_END_TRAINING ** (1 / MAX_TRAINING_STEPS)
#Reward function:
REWARD_SCALE = 2
DEATH_TERM = -50
DEATH_FACTOR = 2

# State representation parameters:
MULTIPLE_BOTS_PRESENT = True if NUM_GREEDY_BOTS + NUM_NN_BOTS > 1 else False
NORMALIZE_GRID_BY_MAX_MASS = False
PELLET_GRID = True
SELF_GRID = MULTIPLE_BOTS_PRESENT
SELF_GRID_LF = MULTIPLE_BOTS_PRESENT
SELF_GRID_SLF = MULTIPLE_BOTS_PRESENT
WALL_GRID = MULTIPLE_BOTS_PRESENT
VIRUS_GRID = False
ENEMY_GRID =MULTIPLE_BOTS_PRESENT
ENEMY_GRID_LF = MULTIPLE_BOTS_PRESENT
ENEMY_GRID_SLF = MULTIPLE_BOTS_PRESENT
SIZE_GRID = False
USE_FOVSIZE = True
USE_TOTALMASS = True
USE_LAST_ACTION = MULTIPLE_BOTS_PRESENT
USE_SECOND_LAST_ACTION = MULTIPLE_BOTS_PRESENT
GRID_SQUARES_PER_FOV = 11 #11 is pretty good so far.
NUM_OF_GRIDS = PELLET_GRID + SELF_GRID + WALL_GRID + VIRUS_GRID + ENEMY_GRID \
               + SIZE_GRID + SELF_GRID_LF + SELF_GRID_SLF \
               + ENEMY_GRID_LF + ENEMY_GRID_SLF
STATE_REPR_LEN = GRID_SQUARES_PER_FOV * GRID_SQUARES_PER_FOV * NUM_OF_GRIDS + USE_FOVSIZE + USE_TOTALMASS \
                 + USE_LAST_ACTION * 4 + USE_SECOND_LAST_ACTION * 4

INITIALIZER = "glorot_uniform" #"Default" or "glorot_uniform" or "glorot_normal"


# Q-learning
NEURON_TYPE = "MLP"
Q_LAYERS = (256, 256, 256)
ALPHA = 0.0001
SQUARE_ACTIONS = True
NUM_ACTIONS = 25
OPTIMIZER = "Adam" #SGD has much worse performance
ACTIVATION_FUNC_HIDDEN = 'relu'
ELU_ALPHA = 1 # TODO: only works for Q-learning so far. Test if it is useful, if so implement for others too
ACTIVATION_FUNC_OUTPUT = 'linear'
GRID_VIEW_ENABLED = True
TARGET_NETWORK_STEPS = 1500
DISCOUNT = 0.85 # Higher discount seems to lead to much more stable learning, less variance
USE_ACTION_AS_INPUT = False
TD = 0

# Actor-critic:
SOFT_TARGET_UPDATES = True
CACLA_CRITIC_LAYERS = (250, 250, 250)
CACLA_CRITIC_ALPHA  = 0.000075
CACLA_ACTOR_LAYERS  = (100, 100, 100)
CACLA_ACTOR_ALPHA   = 0.00015
ACTOR_CRITIC_TYPE = "CACLA" # "Standard"/"CACLA" / "DPG". Standard multiplies gradient by tdE, CACLA only updates once for positive tdE
CACLA_UPDATE_ON_NEGATIVE_TD = False
POLICY_OUTPUT_ACTIVATION_FUNC = "sigmoid"
ACTOR_REPLAY_ENABLED = True
OPTIMIZER_POLICY = "Adam"
ACTIVATION_FUNC_HIDDEN_POLICY = "relu"
# Deterministic Policy Gradient (DPG):
DPG_TAU                    = 0.001 # How quickly the weights of the target networks are updated
DPG_CRITIC_LAYERS          = (250, 250, 250)
DPG_CRITIC_ALPHA           = 0.0005
DPG_CRITIC_FUNC            = "relu"
DPG_CRITIC_WEIGHT_DECAY    = 0.001 #0.001 L2 weight decay parameter. Set to 0 to disable
DPG_ACTOR_LAYERS           = (100, 100, 100)
DPG_ACTOR_ALPHA            = 0.00001
DPG_ACTOR_FUNC             = "relu"
DPG_Q_VAL_INCREASE         = 2
DPG_FEED_ACTION_IN_LAYER   = 1
DPG_USE_DPG_ACTOR_TRAINING = True
DPG_USE_TARGET_MODELS      = True
DPG_USE_CACLA              = False
DPG_CACLA_ALTERNATION      = 0 #fraction of training time in which cacla is used instead of dpg
DPG_CACLA_STEPS            = DPG_CACLA_ALTERNATION * MAX_TRAINING_STEPS

# LSTM
ACTIVATION_FUNC_LSTM = "sigmoid"
UPDATE_LSTM_MOVE_NETWORK = 1
TRACE_MIN = 1 # The minimum amount of traces that are not trained on, as they have insufficient hidden state info
MEMORY_TRACE_LEN = 15 # The length of memory traces retrieved via exp replay

# CNN
CNN_REPRESENTATION = False
CNN_TOWER = False
# Handcraft representation
CNN_USE_LAYER_1 = True
CNN_LAYER_1 = (8, 4, 32)
CNN_SIZE_OF_INPUT_DIM_1 = 84

CNN_USE_LAYER_2 = True
CNN_LAYER_2 = (4, 2, 64)
CNN_SIZE_OF_INPUT_DIM_2 = 84

CNN_USE_LAYER_3 = True
CNN_LAYER_3 = (3, 1, 64)
CNN_SIZE_OF_INPUT_DIM_3 = 42

# Pixel representation
CNN_PIXEL_REPRESENTATION = False
CNN_PIXEL_RGB = False
CNN_PIXEL_INCEPTION = False
CNN_USE_LAST_GRID = True
