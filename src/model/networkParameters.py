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
ENABLE_SPLIT = False #TODO: these two only work  actor critic yet.
ENABLE_EJECT = False #TODO: en/disable ejection and splitting for both continuous and discrete algorithms
NEURON_TYPE = "MLP"
FRAME_SKIP_RATE = 9 # Frame skipping of around 5-10 leads to good performance. 15 and 30 lead to worse performance.
GRID_SQUARES_PER_FOV = 11 #11 is pretty good so far.
NUM_OF_GRIDS = 5
MAX_TRAINING_STEPS = 125000
MAX_SIMULATION_STEPS = MAX_TRAINING_STEPS * (FRAME_SKIP_RATE + 1)
NOISE_AT_HALF_TRAINING = 0.1
NOISE_DECAY = NOISE_AT_HALF_TRAINING ** (1 / (MAX_TRAINING_STEPS / 2)) #0.99995 # Noise decay. with start noise of 1 and decay of 0.999995 it decays slowly to 0 over 1M steps for FSR of 0. For FSR of 9: 0.99995. For FSR of 4: 0.99998

# Q-learning
ALPHA = 0.0001 #Learning rate. Marco recommended to try lower learning rates too, decrease by factor of 10 or 100
OPTIMIZER = "Adam" #SGD has much worse performance
ACTIVATION_FUNC_HIDDEN = 'relu' #'relu' is better than sigmoid, but gives more variable results. we should try elu
ACTIVATION_FUNC_OUTPUT = 'linear'
EXP_REPLAY_ENABLED = True
GRID_VIEW_ENABLED = True
TARGET_NETWORK_STEPS = 1000
DISCOUNT = 0.9 # 0.9 seems best so far. Better than 0.995 and 0.9999 . 0.5 and below performs much worse. 0.925 performs worse than 0.9
# Higher discount seems to lead to much more stable learning, less variance
TD = 0
Exploration = True
EPSILON = 1 if Exploration else 0 # Exploration rate. 0 == No Exploration
EXPLORATION_STRATEGY = "e-Greedy" # "Boltzmann" or "e-Greedy"
TEMPERATURE = 20
TEMPERATURE_AT_END_TRAINING = 0.025
TEMPERATURE_DECAY = TEMPERATURE_AT_END_TRAINING ** (1 / MAX_TRAINING_STEPS)

# Actor-critic:
ACTOR_CRITIC_TYPE = "CACLA" # "Standard"/"CACLA". Standard multiplies gradient by tdE, CACLA only updates once for positive tdE
ACTOR_REPLAY_ENABLED = True
GAUSSIAN_NOISE = 1 # Initial noise
ALPHA_POLICY = 0.00025
OPTIMIZER_POLICY = "Adam"
ACTIVATION_FUNC_HIDDEN_POLICY = "elu"
HIDDEN_ALL_POLICY = 100
HIDDEN_LAYER_1_POLICY = HIDDEN_ALL_POLICY if HIDDEN_ALL_POLICY else 100
HIDDEN_LAYER_2_POLICY = HIDDEN_ALL_POLICY if HIDDEN_ALL_POLICY else 100
HIDDEN_LAYER_3_POLICY = HIDDEN_ALL_POLICY if HIDDEN_ALL_POLICY else 100

# LSTM
ACTIVATION_FUNC_LSTM = "tanh"
UPDATE_LSTM_MOVE_NETWORK = 1
TRACE_MIN = 4 # The minimum amount of traces that are not trained on, as they have insufficient hidden state info
MEMORY_TRACE_LEN = 15 # The length of memory traces retrieved via exp replay

# CNN (Default screen size is 900*900
CNN_REPRESENTATION = False
CNN_PIXEL_REPRESENTATION = False
CNN_USE_LAYER_1 = False
CNN_LAYER_1 = 9
CNN_LAYER_1_STRIDE = 3
CNN_LAYER_1_FILTER_NUM = 10
CNN_POOL_1 = 2
CNN_SIZE_OF_INPUT_DIM_1 = 900

CNN_USE_LAYER_2 = False
CNN_LAYER_2 = 5
CNN_LAYER_2_STRIDE = 2
CNN_LAYER_2_FILTER_NUM = 10
CNN_POOL_2 = 2
CNN_SIZE_OF_INPUT_DIM_2 = 200

CNN_LAYER_3 = 5
CNN_LAYER_3_STRIDE = 2
CNN_LAYER_3_FILTER_NUM = 10
CNN_POOL_3 = 2
CNN_SIZE_OF_INPUT_DIM_3 = 40

CNN_POOLING_TYPE = "Max"



#Layer neurons
STATE_REPR_LEN = GRID_SQUARES_PER_FOV * GRID_SQUARES_PER_FOV * NUM_OF_GRIDS + 2
HIDDEN_ALL = 500
HIDDEN_LAYER_1 = HIDDEN_ALL if HIDDEN_ALL else 500
HIDDEN_LAYER_2 = HIDDEN_ALL if HIDDEN_ALL else 500
HIDDEN_LAYER_3 = HIDDEN_ALL if HIDDEN_ALL else 500
# More hidden layers lead to improved performance. Best so far three hidden layers with 100 neurons each and relu activation
