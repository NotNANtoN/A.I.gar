import math
from .networkParameters import *

# General Parameters
SCREEN_WIDTH = 900
SCREEN_HEIGHT = 900
MAXHUMANPLAYERS = 3

# Simulation Parameters
FPS = 30
GAME_SPEED = 1  # 1sec/1sec
SPEED_MODIFIER = GAME_SPEED / FPS

# Field Parameters
HASH_BUCKET_SIZE = 20
SIZE_INCREASE_PER_PLAYER = 75
START_MASS = 10
START_RADIUS = math.sqrt(START_MASS / math.pi)
# per unit area
MAX_COLLECTIBLE_DENSITY = 0.015 if PELLET_SPAWN else 0
MAX_VIRUS_DENSITY = 0.00005
VIRUS_BASE_SIZE = 100
VIRUS_EAT_FACTOR = 0.5
VIRUS_BASE_RADIUS = math.sqrt(VIRUS_BASE_SIZE / math.pi)
VIRUS_EXPLOSION_BASE_MASS = 15
VIRUS_EXPLOSION_CELL_MASS_PROPORTION = 0.6
EJECTEDBLOB_BASE_MASS = 18

# Cell Parameters
MAX_MASS_SINGLE_CELL = 22500
BASE_MERGE_TIME = 25
MERGE_TIME_MASS_FACTOR = 0.0233
MERGE_TIME_VIRUS_FACTOR = 0.85
CELL_MOVE_SPEED = 90 * SPEED_MODIFIER #units/sec
CELL_MASS_DECAY_RATE = 1 - (0.01 * SPEED_MODIFIER) #default: 1- (0.01 * SPEED_MODIFIER)

# Player Parameters:
#MOMENTUM_PROPORTION_TO_MASS = 0.003
#MOMENTUM_BASE = 6
