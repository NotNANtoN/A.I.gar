import numpy
from .parameters import *
from .spatialHashTable import spatialHashTable
from .replay_buffer import PrioritizedReplayBuffer, ReplayBuffer
import pickle as pkl


class ExpReplay:
    # TODO: extend with prioritized replay based on td_error. Make new specialized functions for this
    def __init__(self, parameters):
        self.memories = []
        self.max = parameters.MEMORY_CAPACITY
        self.batch_size = parameters.MEMORY_BATCH_LEN
        self.parameters = parameters

    def remember(self, new_exp):
        if len(self.memories) >= self.max:
            del self.memories[0]
        self.memories.append(new_exp)

    def canReplay(self):
        return len(self.memories) >= self.batch_size

    def generateTrace(self, experience):
        pass

    def sample(self):
        if self.parameters.NEURON_TYPE == "MLP":
            randIdxs = numpy.random.randint(0, len(self.memories), self.batch_size)
            return [self.memories[idx] for idx in randIdxs]
        elif self.parameters.NEURON_TYPE == "LSTM":
            trace_len = self.parameters.MEMORY_TRACE_LEN
            sampled_start_points = numpy.random.randint(0, len(self.memories) - trace_len, self.batch_size)
            sampled_traces = []
            for start_point in sampled_start_points:
                # Check that the trace does not contain any states in which the bot is dead and no states which are directly before a reset
                valid_start = False
                while not valid_start:
                    valid_start = True
                    for candidate in range(start_point, start_point + trace_len - 1):
                        if self.memories[candidate][-2] is None or self.memories[candidate][-1] == True:
                            valid_start = False
                    if not valid_start:
                        start_point = numpy.random.randint(0, len(self.memories) - trace_len)
                sampled_traces.append(self.memories[start_point:start_point + trace_len])
            return sampled_traces

    def getMemories(self):
        return self.memories


def isCellData(cell):
    return [cell.getX(), cell.getY(), cell.getRadius()]


def checkNan(value):
    if math.isnan(value):
        print("ERROR: predicted reward is nan")
        quit()


def getRelativeCellPos(cell, left, top, size):
    if cell is not None:
        return [round((cell.getX() - left) / size, 5), round((cell.getY() - top) / size, 5)]
    else:
        return [0, 0]


class Bot(object):
    _greedyId = 0
    _nnId = 0
    _randomId = 0
    num_NNbots = 0
    num_Greedybots = 0

    @classmethod
    def init_exp_replayer(cls, parameters, path):
        cls.expReplayer = None
        if parameters.JOB_TRAINING_STEPS != 0 and parameters.JOB_STEP_START > 0:
            with open(path + 'replay_buffer.pkl', 'rb') as input:
                cls.expReplayer = pkl.load(input)

        elif parameters.PRIORITIZED_EXP_REPLAY_ENABLED:
            cls.expReplayer = PrioritizedReplayBuffer(parameters.MEMORY_CAPACITY, parameters.MEMORY_ALPHA,
                                                      parameters.MEMORY_BETA)
        else:
            cls.expReplayer = ReplayBuffer(parameters.MEMORY_CAPACITY)




    @property
    def randomId(self):
        return type(self)._randomId

    @randomId.setter
    def randomId(self, val):
        type(self)._randomId = val

    @property
    def greedyId(self):
        return type(self)._greedyId

    @greedyId.setter
    def greedyId(self, val):
        type(self)._greedyId = val

    @property
    def nnId(self):
        return type(self)._nnId

    @nnId.setter
    def nnId(self, val):
        type(self)._nnId = val

    def __repr__(self):
        return self.type + str(self.id)

    def __init__(self, player, field, bot_type, trainMode, learningAlg, parameters, rgbGenerator=None):
        if bot_type == "Greedy":
            self.id = self.greedyId
            self.greedyId += 1
        elif bot_type == "NN":
            self.id = self.nnId
            self.nnId += 1
        elif bot_type == "Random":
            self.id = self.randomId
            self.randomId += 1
        self.rgbGenerator = rgbGenerator
        self.trainMode = trainMode
        self.parameters = parameters
        self.learningAlg = None
        self.lastMass = None
        self.lastReward = None
        self.lastAction = None
        self.fovSize = None
        self.lastFovSize = None
        self.currentAction = None
        if learningAlg is not None:
            self.learningAlg = learningAlg
            # If Actor-Critic we use continuous actions
            self.trainMode = trainMode
            if not self.trainMode:
                self.learningAlg.setNoise(0)
                self.learningAlg.setTemperature(0)

        self.type = bot_type
        self.player = player
        self.field = field
        self.time = 0
        if self.type == "Greedy":
            self.splitLikelihood = numpy.random.randint(9950, 10000)
            self.ejectLikelihood = 100000  # numpy.random.randint(9990,10000)
        self.totalMasses = []
        self.memories = []
        # If using lstm the memories have to be ordered correctly in time for this bot.
        # if type == "NN" and self.parameters.NEURON_TYPE == "LSTM":
        #     #self.expReplayer = ExpReplay(parameters)
        #     if parameters.PRIORITIZED_EXP_REPLAY_ENABLED:
        #         self.expReplayer = PrioritizedReplayBuffer(parameters.MEMORY_CAPACITY, parameters.MEMORY_ALPHA,
        #                                                    parameters.MEMORY_BETA)
        #     else:
        #         self.expReplayer = ReplayBuffer(parameters.MEMORY_CAPACITY)

        self.secondLastSelfGrid = None
        self.lastSelfGrid = None
        self.secondLastEnemyGrid = None
        self.lastEnemyGrid = None
        self.lastAllPlayerGrid = None
        self.lastPixelGrid = None
        if self.parameters.CNN_REPR:
            if self.parameters.CNN_USE_L1:
                self.gridSquaresPerFov = self.parameters.CNN_INPUT_DIM_1
            elif self.parameters.CNN_USE_L2:
                self.gridSquaresPerFov = self.parameters.CNN_INPUT_DIM_2
            else:
                self.gridSquaresPerFov = self.parameters.CNN_INPUT_DIM_3
        else:
            self.gridSquaresPerFov = self.parameters.GRID_SQUARES_PER_FOV

        self.reset()


    def saveInitialModels(self, path):
        if self.learningAlg is not None:
            self.learningAlg.save(path, "init_")

    def saveModel(self, path):
        self.learningAlg.save(path)

    def resetMassList(self):
        self.totalMasses = []

    def reset(self):
        if self.learningAlg is not None:
            self.learningAlg.reset()
        self.lastMass = None
        self.oldState = None
        # self.stateHistory = []
        self.lastMemory = None
        self.skipFrames = 0
        self.cumulativeReward = 0
        self.lastReward = 0
        # self.rewardHistory = []
        self.rewardAvgOfEpisode = 0
        self.rewardLenOfEpisode = 0
        self.currentlySkipping = False
        if self.type == "NN":
            self.currentActionIdx = None
            self.currentAction = None
            # Mark in the memory that here the episode ended
            if len(self.memories) > 0:
                self.memories[-1][-1] = True
            # self.actionIdxHistory = []
            # self.actionHistory =[]
            if self.parameters.CNN_REPR:
                if self.parameters.CNN_USE_L1:
                    gridSquaresPerFov = self.parameters.CNN_INPUT_DIM_1
                elif self.parameters.CNN_USE_L2:
                    gridSquaresPerFov = self.parameters.CNN_INPUT_DIM_2
                else:
                    gridSquaresPerFov = self.parameters.CNN_INPUT_DIM_3
            else:
                gridSquaresPerFov = self.parameters.GRID_SQUARES_PER_FOV

            self.secondLastSelfGrid = numpy.zeros((gridSquaresPerFov, gridSquaresPerFov))
            self.lastSelfGrid = numpy.zeros((gridSquaresPerFov, gridSquaresPerFov))
            self.secondLastEnemyGrid = numpy.zeros((gridSquaresPerFov, gridSquaresPerFov))
            self.lastEnemyGrid = numpy.zeros((gridSquaresPerFov, gridSquaresPerFov))
            self.fovSize = 0
            self.lastFovSize = 0
        elif self.type == "Greedy" or self.type == "Random":
            self.currentAction = [0, 0, 0, 0]

    def updateRewards(self):
        self.cumulativeReward += self.getReward() if self.lastMass else 0
        self.lastReward = self.cumulativeReward

    # Returns true if we skip this frame
    def updateFrameSkip(self):
        # Do not train if we are skipping this frame
        if self.skipFrames > 0:
            self.skipFrames -= 1
            self.latestTDerror = None
            if self.player.getIsAlive():
                return True
        return False

    def updateValues(self, extraInfo, newAction, newState, newLastMemory=None):
        if newLastMemory is not None:
            self.lastMemory = newLastMemory
        # Reset frame skipping variables
        self.cumulativeReward = 0
        self.skipFrames = self.parameters.FRAME_SKIP_RATE
        self.oldState = newState
        self.lastAction = self.currentAction
        self.currentAction = newAction
        if str(self.learningAlg) == "Q-learning":
            self.currentActionIdx = extraInfo
        else:
            self.currentRawAction = extraInfo

    def learn_and_move_NN(self):
        self.currentlySkipping = False
        if self.currentAction is not None:
            self.updateRewards()
            self.currentlySkipping = self.updateFrameSkip()

        if not self.currentlySkipping:
            newState = self.getStateRepresentation()
            
            # Learn
            if self.trainMode and self.oldState is not None:
                self.time += 1
                action = self.currentActionIdx if self.learningAlg.discrete else self.currentAction

                if self.parameters.EXP_REPLAY_ENABLED:
                    if str(self.learningAlg) != "Q-learning":
                        if self.parameters.ACTOR_CRITIC_TYPE == "CACLA":
                            # Save raw action without noise in "done" field to later determine difference in policies
                            self.expReplayer.add(self.oldState, action, self.lastReward, newState, self.currentRawAction)
                        else:
                            # Save None for SPG in "done" field. This will be later updated with the best sampled action
                            self.expReplayer.add(self.oldState, action, self.lastReward, newState, None)
                    else:
                        self.expReplayer.add(self.oldState, action, self.lastReward, newState, newState is None)

                self.learningAlg.updateNoise()
                self.lastMemory = ([self.oldState], [action], [self.lastReward], [newState], [newState is not None])

                if self.player.getSelected():
                    print("Reward: ", self.cumulativeReward)

            # Move
            if self.player.getIsAlive():
                extraInfo, new_action = self.learningAlg.decideMove(newState, self)

                self.updateValues(extraInfo, new_action, newState)

            if self.player.getIsAlive():
                self.lastMass = self.player.getTotalMass()
            else:
                self.reset()


    def setExploring(self, val):
        self.player.setExploring(val)


    def setMassesOverTime(self, array):
        self.totalMasses = array

    def make_random_bot_move(self):
        if self.time % self.parameters.FRAME_SKIP_RATE == 0:
            self.currentAction[0] = numpy.random.random()
            self.currentAction[1] = numpy.random.random()
            self.currentAction[2] = numpy.random.random() if self.parameters.ENABLE_SPLIT else False
            self.currentAction[3] = numpy.random.random() if self.parameters.ENABLE_EJECT else False
        self.time += 1


    def makeMove(self):
        self.totalMasses.append(self.player.getTotalMass())
        if self.type == "NN":
            self.learn_and_move_NN()

        if not self.player.getIsAlive():
            return

        if self.type == "Greedy":
            self.make_greedy_bot_move()

        if self.type == "Random":
            self.make_random_bot_move()

        action_taken = list(self.currentAction)
        if self.currentlySkipping:
            action_taken[2:] = [0, 0]
        self.set_command_point(action_taken)


    def getStateRepresentation(self):
        stateRepr = None
        if self.player.getIsAlive():
            if self.parameters.GRID_VIEW_ENABLED:
                if self.parameters.CNN_REPR:
                    if self.parameters.CNN_P_REPR:
                        gridView = self.rgbGenerator.get_cnn_inputRGB(self.player)
                        if self.parameters.CNN_LAST_GRID:
                            gridView = numpy.concatenate((gridView, self.lastPixelGrid), axis=2)
                            self.lastPixelGrid = gridView
                        if self.parameters.COORDCONV:
                            coordConv_row, coordConv_col = self.getCoorConvGrids()
                            coordConv = numpy.array([coordConv_row,coordConv_col])
                            coordConv = coordConv.reshape((self.gridSquaresPerFov, self.gridSquaresPerFov, 2))
                            gridView = numpy.concatenate((gridView, coordConv), axis=2)
                    else:
                        gridView = self.getGridStateRepresentation()

                    if self.parameters.EXTRA_INPUT:
                        additionalFeatures = self.getAdditionalFeatures()
                        stateRepr = [gridView, additionalFeatures]
                    else:
                        stateRepr = gridView

                else:
                    gridView = self.getGridStateRepresentation()
                    gridView = gridView.flatten()
                    if self.parameters.EXTRA_INPUT:
                        additionalFeatures = self.getAdditionalFeatures()
                        stateRepr = numpy.concatenate((gridView, additionalFeatures))
                    else:
                        stateRepr = gridView
                    shape = [1]
                    shape.extend(numpy.shape(stateRepr))
                    stateRepr = stateRepr.reshape(shape)
            else:
                stateRepr = self.getSimpleStateRepresentation()

        return stateRepr


    def getAdditionalFeatures(self):
        additionalFeatures = []
        if self.parameters.USE_LAST_FOVSIZE:
            self.lastFovSize = self.fovSize
            additionalFeatures.append(self.lastFovSize)
        if self.parameters.USE_FOVSIZE:
            self.fovSize = self.player.getFovSize()
            additionalFeatures.append(self.fovSize)
        if self.parameters.USE_TOTALMASS:
            mass = self.player.getTotalMass()
            additionalFeatures.append(mass)
        if self.parameters.USE_LAST_ACTION:
            last_action = self.currentAction if self.currentAction is not None else [0, 0, 0, 0]
            additionalFeatures.extend(last_action)
            if len(last_action) < 4:
                additionalFeatures.extend(numpy.zeros(4 - len(last_action)))
        if self.parameters.USE_SECOND_LAST_ACTION:
            second_last_action = self.lastAction if self.lastAction is not None else [0, 0, 0, 0]
            additionalFeatures.extend(second_last_action)
            if len(second_last_action) < 4:
                additionalFeatures.extend(numpy.zeros(4 - len(second_last_action)))
        return additionalFeatures


    def getGridStateRepresentation(self):
        # Get Fov infomation
        fieldSize = self.field.getWidth()
        fovSize = self.player.getFovSize()
        fovPos = self.player.getFovPos()
        x = fovPos[0]
        y = fovPos[1]
        left = x - fovSize / 2
        top = y - fovSize / 2
        # Initialize spatial hash tables:
        gridSquaresPerFov = self.gridSquaresPerFov
        gsSize = fovSize / gridSquaresPerFov  # (gs = grid square)

        pelletSHT = spatialHashTable(fovSize, gsSize, left, top)  # SHT = spatial hash table
        totalPellets = self.field.getPelletsInFov(fovPos, fovSize)
        pelletSHT.insertAllFloatingPointObjects(totalPellets)

        enemyCells = self.field.getEnemyPlayerCellsInFov(self.player)
        playerCells = self.field.getPortionOfCellsInFov(self.player.getCells(), fovPos, fovSize)
        allPlayerSHT = None
        playerSHT = None
        enemySHT = None
        if self.parameters.ALL_PLAYER_GRID:
            allPlayerSHT = spatialHashTable(fovSize, gsSize, left, top)
            allPlayerSHT.insertAllFloatingPointObjects(enemyCells)
            allPlayerSHT.insertAllFloatingPointObjects(playerCells)
        else:
            playerSHT = spatialHashTable(fovSize, gsSize, left, top)
            playerSHT.insertAllFloatingPointObjects(playerCells)
            enemySHT = spatialHashTable(fovSize, gsSize, left, top)
            enemySHT.insertAllFloatingPointObjects(enemyCells)

        virusSHT = None
        if self.field.getVirusEnabled():
            virusSHT = spatialHashTable(fovSize, gsSize, left, top)
            virusCells = self.field.getVirusesInFov(fovPos, fovSize)
            virusSHT.insertAllFloatingPointObjects(virusCells)

        # Calculate mass of biggest cell:
        if self.parameters.NORMALIZE_GRID_BY_MAX_MASS:
            allCells = numpy.concatenate((playerCells, enemyCells))
            biggestCellMass = max(allCells, key = lambda cell: cell.getMass()).getMass()

        # Initialize grid squares with zeros:
        gsPelletMass = numpy.zeros((gridSquaresPerFov, gridSquaresPerFov))
        gsWalls = numpy.zeros((gridSquaresPerFov, gridSquaresPerFov))
        gsBiggestAllCellMass = None
        gsBiggestEnemyCellMass = None
        gsBiggestOwnCellMass = None
        if self.parameters.ALL_PLAYER_GRID:
            gsBiggestAllCellMass = numpy.zeros((gridSquaresPerFov, gridSquaresPerFov))
        else:
            gsBiggestEnemyCellMass = numpy.zeros((gridSquaresPerFov, gridSquaresPerFov))
            gsBiggestOwnCellMass = numpy.zeros((gridSquaresPerFov, gridSquaresPerFov))
        gsVirus = None
        if self.field.getVirusEnabled():
            gsVirus = numpy.zeros((gridSquaresPerFov, gridSquaresPerFov))
        coordConv_row = None
        coordConv_col = None
        # Create row and column grids
        if self.parameters.COORDCONV:
            coordConv_row, coordConv_col = self.getCoorConvGrids()

        gridView = numpy.zeros((self.parameters.NUM_OF_GRIDS, gridSquaresPerFov, gridSquaresPerFov))
        # gsMidPoint is adjusted in the loops
        gsMidPoint = [left + gsSize / 2, top + gsSize / 2]
        for c in range(gridSquaresPerFov):
            for r in range(gridSquaresPerFov):
                count = r + c * gridSquaresPerFov

                # Only check for cells if the grid square fov is within the playing field
                if not (gsMidPoint[0] + gsSize / 2 < 0 or gsMidPoint[0] - gsSize / 2 > fieldSize or
                        gsMidPoint[1] + gsSize / 2 < 0 or gsMidPoint[1] - gsSize / 2 > fieldSize):
                    # Create pellet representation
                    # Make the visionGrid's pellet count a percentage so that the network doesn't have to
                    # work on interpreting the number of pellets relative to the size (and Fov) of the player
                    pelletMassSum = 0
                    pelletsInGS = pelletSHT.getBucketContent(count)
                    if pelletsInGS:
                        for pellet in pelletsInGS:
                            pelletMassSum += pellet.getMass()
                            if NORMALIZE_GRID_BY_MAX_MASS:
                                pelletMassSum /= biggestCellMass
                        gsPelletMass[c][r] = pelletMassSum


                    if self.parameters.ALL_PLAYER_GRID:
                        # Create all player mass representation
                        allPlayersInGS = allPlayerSHT.getBucketContent(count)
                        if allPlayersInGS:
                            biggestCellInGSMass = max(allPlayersInGS, key=lambda cell: cell.getMass()).getMass()
                            if self.parameters.NORMALIZE_GRID_BY_MAX_MASS:
                                biggestCellInGSMass /= biggestCellMass
                            gsBiggestAllCellMass[c][r] = biggestCellInGSMass
                    else:
                        # Create Enemy Cell mass representation
                        # Make the visionGrid's enemy cell representation a percentage. The player's cell mass
                        # in proportion to the biggest enemy cell's mass in each grid square.
                        enemiesInGS = enemySHT.getBucketContent(count)
                        if enemiesInGS:
                            biggestEnemyInGSMass = max(enemiesInGS, key=lambda cell: cell.getMass()).getMass()
                            if self.parameters.NORMALIZE_GRID_BY_MAX_MASS:
                                biggestEnemyInGSMass /= biggestCellMass
                            gsBiggestEnemyCellMass[c][r] = biggestEnemyInGSMass

                        # Create Own Cell mass representation
                        playerCellsInGS = playerSHT.getBucketContent(count)
                        if playerCellsInGS:
                            biggestFriendInGSMass = max(playerCellsInGS, key=lambda cell: cell.getMass()).getMass()
                            if self.parameters.NORMALIZE_GRID_BY_MAX_MASS:
                                biggestFriendInGSMass /= biggestCellMass
                            gsBiggestOwnCellMass[c][r] = biggestFriendInGSMass

                    # Create Virus Cell representation
                    if self.field.getVirusEnabled():
                        virusesInGS = virusSHT.getBucketContent(count)
                        if virusesInGS:
                            biggestVirus = max(virusesInGS, key=lambda virus: virus.getRadius()).getMass()
                            if NORMALIZE_GRID_BY_MAX_MASS:
                                biggestVirus /= biggestCellMass
                            gsVirus[c][r] = biggestVirus

                # Create Wall representation
                # Calculate how much of the grid square is covered by walls
                leftBorder = min(max(gsMidPoint[0] - gsSize / 2, 0), fieldSize)
                topBorder = min(max(gsMidPoint[1] - gsSize / 2, 0), fieldSize)
                rightBorder = max(min(gsMidPoint[0] + gsSize / 2, fieldSize), 0)
                bottomBorder = max(min(gsMidPoint[1] + gsSize / 2, fieldSize), 0)
                freeArea = (rightBorder - leftBorder) * (bottomBorder - topBorder)
                gsWalls[c][r] = round(1 - (freeArea / (gsSize ** 2)), 3)

                # Increment grid square position horizontally
                gsMidPoint[0] += gsSize
            # Reset horizontal grid square, increment grid square position
            gsMidPoint[0] = left + gsSize / 2
            gsMidPoint[1] += gsSize


        count = 0
        if self.parameters.PELLET_GRID:
            gridView[count] = gsPelletMass
            count += 1
        if self.parameters.SELF_GRID:
            gridView[count] = gsBiggestOwnCellMass
            count += 1
        if self.parameters.WALL_GRID:
            gridView[count] = gsWalls
            count += 1
        if self.parameters.ENEMY_GRID:
            gridView[count] = gsBiggestEnemyCellMass
            count += 1
        if self.parameters.ALL_PLAYER_GRID:
            gridView[count] = gsBiggestAllCellMass
            count += 1
        if self.parameters.VIRUS_GRID:
            gridView[count] = gsVirus
            count += 1

        # CoordConv
        if self.parameters.COORDCONV:
            gridView[count] = coordConv_row
            gridView[count+1] = coordConv_col
            count += 2

        # Add grids about own cells and enemy cells from previous frames:
        if self.parameters.SELF_GRID_SLF:
            gridView[count] = self.secondLastSelfGrid
            self.secondLastSelfGrid = self.lastSelfGrid
            count += 1
        if self.parameters.SELF_GRID_LF:
            gridView[count] = self.lastSelfGrid
            self.lastSelfGrid = gsBiggestOwnCellMass
            count += 1
        if self.parameters.ENEMY_GRID_SLF:
            gridView[count] = self.secondLastEnemyGrid
            self.secondLastEnemyGrid = self.lastEnemyGrid
            count += 1
        if self.parameters.ENEMY_GRID_LF:
            gridView[count] = self.lastEnemyGrid
            self.lastEnemyGrid = gsBiggestEnemyCellMass
            count += 1

        return gridView


    def getCoorConvGrids(self):
        dims = (self.gridSquaresPerFov, self.gridSquaresPerFov)
        coordConv_row = numpy.zeros(dims)
        coordConv_col = numpy.zeros(dims)
        for c in range(dims[0]):
            for r in range(dims[1]):
                coordConv_row[c][r] = c
                coordConv_col[c][r] = r
        return coordConv_row, coordConv_col


    def getSimpleStateRepresentation(self):
        # Get data about the field of view of the player
        size = self.player.getFovSize()
        midPoint = self.player.getFovPos()
        x = int(midPoint[0])
        y = int(midPoint[1])
        left = x - int(size / 2)
        top = y - int(size / 2)
        size = int(size)
        # At the moment we only care about the first cell of the current player, to be extended once we get this working
        firstPlayerCell = self.player.getCells()[0]

        # Adding all the state data to totalInfo
        totalInfo = []
        # Add data about player cells
        cellInfos = self.getCellDataOwnPlayer(left, top, size)
        for info in cellInfos:
            totalInfo += info
        # Add data about the closest enemy cell
        playerCellsInFov = self.field.getEnemyPlayerCellsInFov(self.player)
        closestEnemyCell = min(playerCellsInFov,
                               key=lambda p: p.squaredDistance(firstPlayerCell)) if playerCellsInFov else None
        totalInfo += self.isRelativeCellData(closestEnemyCell, left, top, size)
        # Add data about the closest pellet
        pelletsInFov = self.field.getPelletsInFov(midPoint, size)
        closestPellet = min(pelletsInFov, key=lambda p: p.squaredDistance(firstPlayerCell)) if pelletsInFov else None
        closestPelletPos = getRelativeCellPos(closestPellet, left, top, size)
        totalInfo += closestPelletPos
        # Add data about distances to the visible edges of the field
        width = self.field.getWidth()
        height = self.field.getHeight()
        distLeft = x / size if left <= 0 else 1
        distRight = (width - x) / size if left + size >= width else 1
        distTop = y / size if top <= 0 else 1
        distBottom = (height - y) / size if top + size >= height else 1
        totalInfo += [distLeft, distRight, distTop, distBottom]
        return totalInfo


    def set_command_point(self, action):
        midPoint = self.player.getFovPos()
        size = self.player.getFovSize()
        x = int(midPoint[0])
        y = int(midPoint[1])
        left = x - int(size / 2)
        top = y - int(size / 2)
        size = int(size)
        xChoice = left + action[0] * size
        yChoice = top + action[1] * size
        splitChoice = None
        ejectChoice = None
        if len(action) > 2:
            if len(action) == 3:
                if self.parameters.ENABLE_SPLIT:
                    ejectChoice = False
                    splitChoice = True if action[2] > 0.5 else False
                elif self.parameters.ENABLE_EJECT:
                    ejectChoice = True if [3] > 0.5 else False
                    splitChoice = False
            else:
                splitChoice = True if action[2] > 0.5 else False
                ejectChoice = True if action[3] > 0.5 else False
        else:
            splitChoice = False
            ejectChoice = False

        self.player.setCommands(xChoice, yChoice, splitChoice, ejectChoice)

    def make_greedy_bot_move(self):
        midPoint = self.player.getFovPos()
        size = self.player.getFovSize()
        x = int(midPoint[0])
        y = int(midPoint[1])
        left = x - int(size / 2)
        top = y - int(size / 2)

        cellsInFov = self.field.getPelletsInFov(midPoint, size)
        playerCells = self.player.getCells()
        biggestPlayerCell = max(playerCells, key=lambda p: p.getMass())
        # If the bot is split, use its biggest cell as reference
        if len(playerCells) > 1:
            biggestCellMass = biggestPlayerCell.getMass()
            for playerCell in playerCells:
                playerCellMass = playerCell.getMass()
                if biggestCellMass < playerCellMass:
                    biggestPlayerCell = playerCell
                    biggestCellMass = playerCellMass

        playerCellsInFov = self.field.getEnemyPlayerCellsInFov(self.player)
        for opponentCell in playerCellsInFov:
            # If the single celled bot can eat the opponent cell add it to list
            if biggestPlayerCell.getMass() > 1.25 * opponentCell.getMass():
                cellsInFov.append(opponentCell)

        if self.field.getVirusEnabled():
            virusCellsInFov = self.field.getVirusesInFov(midPoint, size)
            for virus in virusCellsInFov:
                # If the single celled bot can eat the opponent cell add it to list
                if biggestPlayerCell.getMass() > 1.25 * virus.getMass():
                    cellsInFov.append(virus)

        if cellsInFov:
            bestCell = max(cellsInFov, key=lambda p: p.getMass() / (
                p.squaredDistance(biggestPlayerCell) if p.squaredDistance(biggestPlayerCell) != 0 else 1))
            bestCellPos = getRelativeCellPos(bestCell, left, top, size)
            self.currentAction[0] = bestCellPos[0]
            self.currentAction[1] = bestCellPos[1]
        else:
            size = int(size / 2)
            self.currentAction[0] = numpy.random.random()
            self.currentAction[1] = numpy.random.random()
        self.currentAction[2] = False
        self.currentAction[3] = False
        if self.parameters.ENABLE_GREEDY_SPLIT:
            randNumSplit = numpy.random.randint(0, 10000)
            randNumEject = numpy.random.randint(0, 10000)
            if randNumSplit > self.splitLikelihood:
                self.currentAction[2] = True
            if randNumEject > self.ejectLikelihood:
                self.currentAction[3] = True
        else:
            self.currentAction[2] = False
            self.currentAction[3] = False

    def isRelativeCellData(self, cell, left, top, size):
        return getRelativeCellPos(cell, left, top, size) + \
               ([round(cell.getRadius() / size if cell.getRadius() <= size else 1, 5)] if cell is not None else [0])

    def getMassOverTime(self):
        return self.totalMasses

    def getAvgReward(self):
        return self.rewardAvgOfEpisode

    def getLastReward(self):
        return self.lastReward

    def getCellDataOwnPlayer(self, left, top, size):
        cells = self.player.getCells()
        totalCells = len(cells)
        return [self.isRelativeCellData(cells[idx], left, top, size) if idx < totalCells else [0, 0, 0]
                for idx in range(1)]

    def getReward(self):
        if self.parameters.MASS_AS_REWARD:
            if self.player.getIsAlive():
                return self.player.getTotalMass() - self.parameters.REWARD_TERM
            else:
                return self.parameters.DEATH_TERM - self.parameters.REWARD_TERM
        if self.lastMass is None:
            return None
        if not self.player.getIsAlive():
            reward = -1 * self.lastMass * self.parameters.DEATH_FACTOR + self.parameters.DEATH_TERM
        else:
            currentMass = self.player.getTotalMass()
            reward = currentMass - self.lastMass
        return reward * self.parameters.REWARD_SCALE - self.parameters.REWARD_TERM

    def getFrameSkipRate(self):
        return self.parameters.FRAME_SKIP_RATE

    def getType(self):
        return self.type

    def getPlayer(self):
        return self.player

    def getTrainMode(self):
        return self.trainMode

    def getLearningAlg(self):
        return self.learningAlg

    def getLastState(self):
        return self.oldState

    def getCurrentActionIdx(self):
        return self.currentActionIdx

    def getCurrentAction(self):
        return self.currentAction

    def getCumulativeReward(self):
        return self.cumulativeReward

    def getLastMemory(self):
        return self.lastMemory

    def getExpRepEnabled(self):
        return self.parameters.EXP_REPLAY_ENABLED

    def getGridSquaresPerFov(self):
        return self.getGridSquaresPerFov()

    def getExpReplayer(self):
        return self.expReplayer
