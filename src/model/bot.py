import heapq
import keras
import numpy
import tensorflow as tf
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.utils.training_utils import multi_gpu_model
from .parameters import *
from .networkParameters import *
from .spatialHashTable import spatialHashTable

class Bot(object):
    """docstring for Bot"""
    # Create all possible discrete actions
    actions = [[x, y, split, eject] for x in [0, 0.5, 1] for y in [0, 0.5, 1] for split in [0, 1] for
               eject in [0, 1]]
    # Filter out actions that do a split and eject at the same time
    for action in actions[:]:
        #if action[2] and action[3]:
        if action[2] or action[3]:
            actions.remove(action)

    num_actions = len(actions)
    stateReprLen = STATE_REPR_LEN
    actionLen = 4

    gpus = GPUS

    # Experience replay:
    memoryCapacity = MEMORY_CAPACITY
    memoriesPerUpdate = MEMORIES_PER_UPDATE # Must be divisible by 4 atm due to experience replay
    memories = []

    num_NNbots = 0
    num_Greedybots = 0

    # Q-learning
    targetNetworkSteps = TARGET_NETWORK_STEPS
    targetNetworkMaxSteps = TARGET_NETWORK_MAX_STEPS
    discount = DISCOUNT
    epsilon = EPSILON
    frameSkipRate = FRAME_SKIP_RATE
    gridSquaresPerFov = GRID_SQUARES_PER_FOV # is modified by the user later on anyways

    #ANN
    learningRate = ALPHA
    optimizer = OPTIMIZER
    activationFuncHidden = ACTIVATION_FUNC_HIDDEN
    activationFuncOutput = ACTIVATION_FUNC_OUTPUT

    hiddenLayer1 = HIDDEN_LAYER_1
    hiddenLayer2 = HIDDEN_LAYER_2
    hiddenLayer3 = HIDDEN_LAYER_3

    loadedModelName = None

    @classmethod
    def initializeNNs(cls):
        weight_initializer_range = math.sqrt(6 / (cls.stateReprLen + cls.num_actions))
        initializer = keras.initializers.RandomUniform(minval=-weight_initializer_range,
                                                       maxval=weight_initializer_range, seed=None)
        if cls.gpus > 1:
            with tf.device("/cpu:0"):
                cls.valueNetwork = Sequential()
                cls.valueNetwork.add(Dense(cls.hiddenLayer1, input_dim=cls.stateReprLen, activation=cls.activationFuncHidden,
                                           bias_initializer=initializer, kernel_initializer=initializer))
                if cls.hiddenLayer2 > 0:
                    cls.valueNetwork.add(Dense(cls.hiddenLayer2, activation=cls.activationFuncHidden, bias_initializer=initializer
                              , kernel_initializer=initializer))
                if cls.hiddenLayer3 > 0:
                    cls.valueNetwork.add(Dense(cls.hiddenLayer3, activation=cls.activationFuncHidden, bias_initializer=initializer
                                               , kernel_initializer=initializer))
                cls.valueNetwork.add(Dense(cls.num_actions, activation=cls.activationFuncOutput, bias_initializer=initializer
                                       , kernel_initializer=initializer))
                cls.valueNetwork = multi_gpu_model(cls.valueNetwork, gpus=cls.gpus)
        else:
            cls.valueNetwork = Sequential()
            cls.valueNetwork.add(Dense(cls.hiddenLayer1, input_dim=cls.stateReprLen, activation=cls.activationFuncHidden,
                                       bias_initializer=initializer
                                       , kernel_initializer=initializer))
            #cls.valueNetwork.add(Dropout(0.5))
            if cls.hiddenLayer2 > 0:
                cls.valueNetwork.add(
                    Dense(cls.hiddenLayer2, activation=cls.activationFuncHidden, bias_initializer=initializer
                          , kernel_initializer=initializer))
                #cls.valueNetwork.add(Dropout(0.5))

            if cls.hiddenLayer3 > 0:
                cls.valueNetwork.add(
                    Dense(cls.hiddenLayer3, activation=cls.activationFuncHidden, bias_initializer=initializer
                          , kernel_initializer=initializer))
                #cls.valueNetwork.add(Dropout(0.5))

            cls.valueNetwork.add(
                Dense(cls.num_actions, activation=cls.activationFuncOutput, bias_initializer=initializer
                      , kernel_initializer=initializer))

        cls.targetNetwork = keras.models.clone_model(cls.valueNetwork)
        cls.targetNetwork.set_weights(cls.valueNetwork.get_weights())

        if cls.optimizer == "Adam":
            cls.valueNetwork.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=cls.learningRate))
            cls.targetNetwork.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=cls.learningRate))
        elif cls.optimizer =="SGD":
            cls.valueNetwork.compile(loss='mse', optimizer=keras.optimizers.SGD(lr=cls.learningRate))
            cls.targetNetwork.compile(loss='mse', optimizer=keras.optimizers.SGD(lr=cls.learningRate))


    def __init__(self, player, field, type, trainMode):
        self.expRepEnabled = EXP_REPLAY_ENABLED
        self.gridViewEnabled = GRID_VIEW_ENABLED
        self.trainMode = trainMode
        self.type = type
        self.player = player
        self.field = field
        self.oldState = None
        self.currentAction = None
        self.currentActionIdx = None


        if self.type == "NN":
            self.lastMass = None
            self.reward = None
            self.latestTDerror = None
            self.cumulativeReward = 0
            self.skipFrames = 0
        elif self.type == "Greedy":
            self.splitLikelihood = numpy.random.randint(9950,10000)
            self.ejectLikelihood = numpy.random.randint(9990,10000)
            self.currentAction = [0, 0, 0, 0]

    def update(self):
        if self.type == "NN":
            if self.trainMode:
                self.qLearn()
            else:
                self.testNetwork()
        elif self.type == "Greedy":
            if not self.player.getIsAlive():
                return
            midPoint = self.player.getFovPos()
            size = self.player.getFovSize()
            x = int(midPoint[0])
            y = int(midPoint[1])
            left = x - int(size / 2)
            top = y - int(size / 2)

            cellsInFov = self.field.getPelletsInFov(midPoint, size)

            playerCellsInFov = self.field.getEnemyPlayerCellsInFov(self.player)
            firstPlayerCell = self.player.getCells()[0]
            for opponentCell in playerCellsInFov:
                # If the single celled bot can eat the opponent cell add it to list
                if firstPlayerCell.getMass() > 1.25 * opponentCell.getMass():
                    cellsInFov.append(opponentCell)
            if cellsInFov:
                bestCell = max(cellsInFov, key = lambda p: p.getMass() / (p.squaredDistance(firstPlayerCell) if p.squaredDistance(firstPlayerCell) != 0 else 1))
                bestCellPos = self.getRelativeCellPos(bestCell, left, top, size)
                self.currentAction[0] = bestCellPos[0]
                self.currentAction[1] = bestCellPos[1]
            else:
                size = int(size / 2)
                self.currentAction[0] = numpy.random.random()
                self.currentAction[1] = numpy.random.random()
            randNumSplit = numpy.random.randint(0,10000)
            randNumEject = numpy.random.randint(0,10000)
            self.currentAction[2] = False
            self.currentAction[3] = False
            if randNumSplit > self.splitLikelihood:
                self.currentAction[2] = True
            if randNumEject > self.ejectLikelihood:
                self.currentAction[3] = True

        if self.player.getIsAlive():
            midPoint = self.player.getFovPos()
            size = self.player.getFovSize()
            x = int(midPoint[0])
            y = int(midPoint[1])
            left = x - int(size / 2)
            top = y - int(size / 2)
            size = int(size)
            xChoice = left + self.currentAction[0] * size
            yChoice = top + self.currentAction[1] * size
            splitChoice = True if self.currentAction[2] > 0.5 else False
            ejectChoice = True if self.currentAction[3] > 0.5 else False

            self.player.setCommands(xChoice, yChoice, splitChoice, ejectChoice)

    def calculateTarget(self, newState, reward, alive):
        targetNetworkEnabled = True
        target = reward
        if alive:
            # The target is the reward plus the discounted prediction of the value network
            if targetNetworkEnabled:
                action_Q_values = self.targetNetwork.predict(numpy.array([newState]))[0]
            else:
                action_Q_values = self.valueNetwork.predict(numpy.array([newState]))[0]
            newActionIdx = numpy.argmax(action_Q_values)
            target += self.discount * action_Q_values[newActionIdx]
        return target

    def createInputOutputPair(self, oldState, actionIdx, reward, newState, alive):
        state_Q_values = self.valueNetwork.predict(numpy.array([oldState]))[0]
        target = self.calculateTarget(newState, reward, alive)

        td_error = target - state_Q_values[actionIdx]
        if  __debug__ and self.player.getSelected():
            print("")
            print("State to be updated: ", oldState)
            print("Action: ", self.actions[actionIdx])
            print("Reward: " ,round(reward, 2))
            print("S\': ", newState)
            print("Qvalue of action before trainig: ", round(state_Q_values[actionIdx], 4))
            print("Target Qvalue of that action: ", round(target, 4))
            print("All qvalues: ", numpy.round(state_Q_values, 3))
            print("TD-Error: ", td_error)
        state_Q_values[actionIdx] = target
        return numpy.array([oldState]), numpy.array([state_Q_values]), td_error

    def qLearn(self):
        #After S has been initialized, set S as oldState and take action A based on policy
        alive = self.player.getIsAlive()

        # Do not train if we are skipping this frame
        if self.skipFrames > 0 :
            self.skipFrames -= 1
            reward = self.getReward()
            self.cumulativeReward += reward
            self.currentAction[2:4] = [0, 0]
            self.latestTDerror = None
            if alive:
                return

        newState = self.getStateRepresentation()

        # Only train when we there is an old state to train
        if self.currentAction != None:
            # Get reward of skipped frames
            reward = self.cumulativeReward
            input, target, td_error = self.createInputOutputPair(self.oldState, self.currentActionIdx, reward,
                                                                 newState, alive)
            # Fit value network using experience replay of random past states:
            if self.expRepEnabled:
                self.experienceReplay(reward, newState, td_error)
            # Fit value network using only the current experience
            else:
                self.valueNetwork.train_on_batch(input, target)

            if  __debug__ and self.player.getSelected():
                updatedQvalueOfAction = self.valueNetwork.predict(numpy.array([self.oldState]))[0][
                    self.currentActionIdx]
                print("Qvalue of action after training: ", round(updatedQvalueOfAction, 4))
                print("(also after experience replay, so last shown action is not necessarily this action )")
                print("TD-Error: ", td_error)
                print("")

            self.latestTDerror = td_error


            # Update the target network after 1000 steps
            # Save the weights of the model when updating the target network to avoid losing progress on program crashes
            self.targetNetworkSteps -= 1
            if self.targetNetworkSteps == 0:
                self.targetNetwork.set_weights(self.valueNetwork.get_weights())
                self.targetNetworkSteps = self.targetNetworkMaxSteps * self.num_NNbots
                self.targetNetwork.save("mostRecentAutosave.h5")

        if self.player.getIsAlive():
            self.takeAction(newState)
            self.lastMass = self.player.getTotalMass()
            self.oldState = newState
        else:
            self.currentActionIdx = None
            self.currentAction = None
            self.lastMass = None
            self.oldState = None
            self.latestTDerror = None
            self.skipFrames = 0
            self.cumulativeReward = 0

    def testNetwork(self):
        alive = self.player.getIsAlive()
        if alive:
            newState = self.getStateRepresentation()
            self.takeAction(newState)


    def experienceReplay(self, reward, newState, td_error):
        if self.player.getIsAlive():
            self.remember(self.oldState, self.currentActionIdx, reward, newState, td_error)
        self.train_on_experience()

    def remember(self, state, action, reward, newState, td_error):
        # Store current state, action, reward, state pair in memory
        # Delete oldest memory if memory is at full capacity
        if len(self.memories) > self.memoryCapacity:
            #if numpy.random.random() > 0.0:
                del self.memories[0]
            #else:
            #    self.memories.remove(min(self.memories, key = lambda memory: abs(memory[-1])))
        if self.player.getIsAlive():
            newMemory = [state.tolist(), action, reward, newState.tolist()]
        else:
            newMemory = [state.tolist(), action, reward, None]
        heapq.heappush(self.memories, ((td_error * td_error) * -1, newMemory))
        #self.memories.append(newMemory)

    def memoryToInputOutput(self, memory):
        s = memory[0]
        a = memory[1]
        r = memory[2]
        sPrime = memory[3]
        alive = (sPrime is not None)
        return self.createInputOutputPair(s, a, r, sPrime, alive)

    def train_on_experience(self):
        # Fit value network on memories
        len_memory = len(self.memories)
        if len_memory < self.memoriesPerUpdate:
            return
        inputSize = STATE_REPR_LEN
        outputSize = self.num_actions
        batch_size = self.memoriesPerUpdate
        # Initialize vectors
        inputs = numpy.zeros((batch_size, inputSize))
        targets = numpy.zeros((batch_size, outputSize))
        batch_count = 0

        # Get most surprising memories:
        popped_memories = []
        for idx in range(int(batch_size / 2)):
            # Get the item with highest priority (td-error)
            memory = heapq.heappop(self.memories)[1]
            input, target, td_error = self.memoryToInputOutput(memory)
            # Set input and target
            inputs[idx] = input
            targets[idx] = target
            # Update td-error for memory
            popped_memories.append(((td_error * td_error) * -1, memory))
            batch_count += 1
        # Put the retrieved memories back in memory
        for poppedMemory in popped_memories:
            heapq.heappush(self.memories, poppedMemory)

        # Get Random memories:
        for idx in range(int(batch_size / 4)):
            randIdx = numpy.random.randint(len(self.memories))
            memory = self.memories[randIdx][1]
            input, target, td_error = self.memoryToInputOutput(memory)
            inputs[batch_count] = input
            targets[batch_count] = target
            # Update TD-Error of memory:
            self.memories[randIdx] = (td_error, memory)
            batch_count += 1

        # Get recent memories:
        for idx in range(int(batch_size / 4)):
            memory = self.memories[len_memory - idx - 1][1]
            input, target, td_error = self.memoryToInputOutput(memory)
            inputs[batch_count] = input
            targets[batch_count] = target
            # Update TD-Error of memory:
            self.memories[idx] = (td_error, memory)
            batch_count += 1

        # Train on memories
        self.valueNetwork.train_on_batch(inputs, targets)

    def takeAction(self, newState):
        # Take random action with probability 1 - epsilon
        if numpy.random.random(1) < self.epsilon:
            self.currentActionIdx = numpy.random.randint(len(self.actions))
            if __debug__:
                self.player.setExploring(True)
        else:
            # Take action based on greediness towards Q values
            qValues = self.valueNetwork.predict(numpy.array([newState]))
            argMax = numpy.argmax(qValues)
            self.currentActionIdx = argMax
            if __debug__:
                self.player.setExploring(False)
        self.currentAction = self.actions[self.currentActionIdx]
        self.skipFrames = self.frameSkipRate
        self.cumulativeReward = 0


    def getStateRepresentation(self):
        stateRepr = None
        if self.player.getIsAlive():
            if self.gridViewEnabled:
                stateRepr =  self.getGridStateRepresentation()
            else:
                stateRepr =  self.getSimpleStateRepresentation()
        return stateRepr

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
        closestPelletPos = self.getRelativeCellPos(closestPellet, left, top, size)
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
        gsSize = fovSize / self.gridSquaresPerFov  # (gs = grid square)
        pelletSHT = spatialHashTable(fovSize, gsSize, left, top) #SHT = spatial hash table
        enemySHT =  spatialHashTable(fovSize, gsSize, left, top)
        virusSHT =  spatialHashTable(fovSize, gsSize, left, top)
        playerSHT = spatialHashTable(fovSize, gsSize, left, top)
        totalPellets = self.field.getPelletsInFov(fovPos, fovSize)
        pelletSHT.insertAllFloatingPointObjects(totalPellets)
        if __debug__ and self.player.getSelected():
            print("Total pellets: ",len(totalPellets))
            print("pellet view: ")
            buckets = pelletSHT.getBuckets()
            for idx in range(len(buckets)):
                print(len(buckets[idx]), end = " ")
                if idx != 0 and (idx + 1) % self.gridSquaresPerFov == 0:
                    print(" ")
        playerCells = self.field.getPortionOfCellsInFov(self.player.getCells(), fovPos, fovSize)
        playerSHT.insertAllFloatingPointObjects(playerCells)
        enemyCells = self.field.getEnemyPlayerCellsInFov(self.player)
        enemySHT.insertAllFloatingPointObjects(enemyCells)
        virusCells = self.field.getVirusesInFov(fovPos, fovSize)
        virusSHT.insertAllFloatingPointObjects(virusCells)

        # Mass vision grid related
        enemyCellsCount = len(enemyCells)
        allCellsInFov = playerCells + enemyCells + virusCells
        biggestMassInFov = max(allCellsInFov, key = lambda cell: cell.getMass()).getMass() if allCellsInFov else None

        # Initialize grid squares with zeros:
        gridNumberSquared = self.gridSquaresPerFov * self.gridSquaresPerFov
        gsBiggestEnemyCellMassProportion = numpy.zeros(gridNumberSquared)
        gsBiggestOwnCellMassProportion = numpy.zeros(gridNumberSquared)
        gsWalls = numpy.zeros(gridNumberSquared)
        gsVirus = numpy.zeros(gridNumberSquared)
        gsPelletProportion = numpy.zeros(gridNumberSquared)
        # gsMidPoint is adjusted in the loops
        gsMidPoint = [left + gsSize / 2, top + gsSize/ 2]
        pelletcount = 0
        for c in range(self.gridSquaresPerFov):
            for r in range(self.gridSquaresPerFov):
                count = r + c * self.gridSquaresPerFov

                # Only check for cells if the grid square fov is within the playing field
                if not(gsMidPoint[0] + gsSize / 2 < 0 or gsMidPoint[0] - gsSize / 2 > fieldSize or
                        gsMidPoint[1] + gsSize / 2 < 0 or gsMidPoint[1] - gsSize / 2 > fieldSize):
                    # Create pellet representation
                    # Make the visionGrid's pellet count a percentage so that the network doesn't have to
                    # work on interpreting the number of pellets relative to the size (and Fov) of the player
                    pelletMassSum = 0
                    pelletsInGS = pelletSHT.getBucketContent(count)
                    if pelletsInGS:
                        for pellet in pelletsInGS:
                            pelletcount += 1
                            pelletMassSum += pellet.getMass()
                        gsPelletProportion[count] = pelletMassSum / biggestMassInFov

                    # TODO: add relative fov pos of closest pellet to allow micro management

                    # Create Enemy Cell mass representation
                    # Make the visionGrid's enemy cell representation a percentage. The player's mass
                    # in proportion to the biggest enemy cell's mass in each grid square.
                    enemiesInGS = enemySHT.getBucketContent(count)
                    if enemiesInGS:
                        biggestEnemyInCell = max(enemiesInGS, key = lambda cell: cell.getMass())
                        gsBiggestEnemyCellMassProportion[count] = biggestEnemyInCell.getMass() / biggestMassInFov

                    # Create Own Cell mass representation
                    playerCellsInGS = playerSHT.getBucketContent(count)
                    if playerCellsInGS:
                        biggestFriendInCell = max(playerCellsInGS, key=lambda cell: cell.getMass())
                        gsBiggestOwnCellMassProportion[count] = biggestFriendInCell.getMass() / biggestMassInFov
                    # TODO: also add a count grid for own cells?

                    # Create Virus Cell representation
                    virusesInGS = virusSHT.getBucketContent(count)
                    if virusesInGS:
                        biggestVirus = max(virusesInGS, key = lambda virus: virus.getRadius()).getMass()
                        gsVirus[count] = biggestVirus / biggestMassInFov

                # Create Wall representation
                # 1s indicate a wall present in the grid square (regardless of amount of wall in square), else 0
                if gsMidPoint[0] - gsSize/2 < 0 or gsMidPoint[0] + gsSize/2 > fieldSize or \
                        gsMidPoint[1] - gsSize/2 < 0 or gsMidPoint[1] + gsSize/2 > fieldSize:
                    gsWalls[count] = 1
                # Increment grid square position horizontally
                gsMidPoint[0] += gsSize
            # Reset horizontal grid square, increment grid square position
            gsMidPoint[0] = left + gsSize / 2
            gsMidPoint[1] += gsSize
        if __debug__ and self.player.getSelected():
            print("counted pellets: ", pelletcount)
            print(" ")
        # Collect all relevant data
        totalInfo = numpy.concatenate((gsPelletProportion, gsBiggestEnemyCellMassProportion, gsBiggestOwnCellMassProportion,
                                        gsWalls, gsVirus))
        return totalInfo

    def getGridStateRepresentationOld(self):
        # Get Fov infomation
        fieldSize = self.field.getWidth()
        size = self.player.getFovSize()
        midPoint = self.player.getFovPos()
        x = int(midPoint[0])
        y = int(midPoint[1])
        left = x - int(size / 2)
        top = y - int(size / 2)
        gsSize = int(size / self.gridSquaresPerFov)  # (gs = grid square)

        totalPellets = self.field.getPelletsInFov(midPoint, size)
        playerCells = self.field.getPortionOfCellsInFov(self.player.getCells(), midPoint, size)
        enemyCells = self.field.getEnemyPlayerCellsInFov(self.player)
        virusCells = self.field.getVirusesInFov(midPoint, size)


        # Pellet vision grid related
        totalPelletMass = [pellet.getMass() for pellet in totalPellets]
        totalPelletsMassSum = sum(totalPelletMass)
        # Mass vision grid related
        enemyCellsCount = len(enemyCells)
        allCellsInFov = playerCells + enemyCells + virusCells
        biggestMassInFov = max(allCellsInFov, key = lambda cell: cell.getMass()).getMass() if allCellsInFov else None

        # Initialize grid squares with zeros:
        gridNumberSquared = self.gridSquaresPerFov * self.gridSquaresPerFov
        gsBiggestEnemyCellMassProportion = numpy.zeros(gridNumberSquared)
        gsBiggestOwnCellMassProportion = numpy.zeros(gridNumberSquared)
        gsEnemyCellCount = numpy.zeros(gridNumberSquared)
        gsWalls = numpy.zeros(gridNumberSquared)
        gsVirus = numpy.zeros(gridNumberSquared)
        gsPelletProportion = numpy.zeros(gridNumberSquared)
        # gsMidPoint is adjusted in the loops
        gsMidPoint = [left + gsSize / 2, top + gsSize/ 2]
        for c in range(self.gridSquaresPerFov):
            for r in range(self.gridSquaresPerFov):
                count = r + c * self.gridSquaresPerFov

                # Only check for cells if the grid square fov is within the playing field
                if not(gsMidPoint[0] + gsSize / 2 <= 0 or gsMidPoint[0] - gsSize / 2 >= fieldSize or
                        gsMidPoint[1] + gsSize / 2 <= 0 or gsMidPoint[1] - gsSize / 2 >= fieldSize):
                    # Create pellet representation
                    # Make the visionGrid's pellet count a percentage so that the network doesn't have to
                    # work on interpreting the number of pellets relative to the size (and Fov) of the player
                    pelletMassSum = 0
                    for pellet in self.field.getPelletsInFov(gsMidPoint, gsSize):
                        pelletMassSum += pellet.getMass()
                    if totalPelletsMassSum > 0:
                        gsPelletProportion[count] = pelletMassSum / totalPelletsMassSum

                    # TODO: add relative fov pos of closest pellet to allow micro management

                    # Create Enemy Cell mass representation
                    # Make the visionGrid's enemy cell representation a percentage. The player's mass
                    # in proportion to the biggest enemy cell's mass in each grid square.
                    enemiesInCell = []
                    for enemy in enemyCells:
                        if enemy.isInFov(gsMidPoint, gsSize):
                            enemiesInCell.append(enemy)
                    if enemiesInCell:
                        biggestEnemyInCell = max(enemiesInCell, key = lambda cell: cell.getMass())
                        gsBiggestEnemyCellMassProportion[count] = biggestEnemyInCell.getMass() / biggestMassInFov

                    # Create Enemy Cell number representation
                    # Just a grid with number of enemy cells on each square
                    if enemyCells:
                        gsEnemyCellCount[count] = len(enemiesInCell) / enemyCellsCount

                    # Create Own Cell mass representation
                    ownCellsInCell = []
                    for friend in playerCells:
                        if friend.isInFov(gsMidPoint, gsSize):
                            ownCellsInCell.append(friend)
                    if ownCellsInCell:
                        biggestFriendInCell = max(ownCellsInCell, key=lambda cell: cell.getMass())
                        gsBiggestOwnCellMassProportion[count] = biggestFriendInCell.getMass() / biggestMassInFov
                    # TODO: also add a count grid for own cells?

                    # Create Virus Cell representation
                    virusesInGridCell = []
                    for virus in virusCells:
                        if virus.isInFov(gsMidPoint, gsSize):
                            virusesInGridCell.append(virus)
                    if virusesInGridCell:
                        biggestVirus = max(virusesInGridCell, key = lambda virus: virus.getRadius()).getMass()
                        gsVirus[count] = biggestVirus / biggestMassInFov

                # Create Wall representation
                # 1s indicate a wall present in the grid square (regardless of amount of wall in square), else 0
                if gsMidPoint[0] - gsSize/2 < 0 or gsMidPoint[0] + gsSize/2 > fieldSize or \
                        gsMidPoint[1] - gsSize/2 < 0 or gsMidPoint[1] + gsSize/2 > fieldSize:
                    gsWalls[count] = 1

                # Increment grid square position horizontally
                gsMidPoint[0] += gsSize
            # Reset horizontal grid square, increment grid square position
            gsMidPoint[0] = left + gsSize / 2
            gsMidPoint[1] += gsSize
        # Collect all relevant data
        totalInfo = numpy.concatenate((gsPelletProportion, gsBiggestEnemyCellMassProportion, gsBiggestOwnCellMassProportion,
                                       gsEnemyCellCount, gsWalls, gsVirus))
        return totalInfo

    def saveModel(self, path):
        self.targetNetwork.set_weights(self.valueNetwork.get_weights())
        self.targetNetwork.save(path + self.type + "_model.h5")

    def setEpsilon(self, val):
        self.epsilon = val

    def isCellData(self, cell):
        return [cell.getX(), cell.getY(), cell.getRadius()]

    def isRelativeCellData(self, cell, left, top, size):
        return self.getRelativeCellPos(cell, left, top, size) + \
               ([round(cell.getRadius() / size if cell.getRadius() <= size else 1, 5)] if cell != None else [0])

    def getRelativeCellPos(self, cell, left, top, size):
        if cell != None:
            return [round((cell.getX() - left) / size, 5), round((cell.getY() - top) / size, 5)]
        else:
            return [0, 0]

    def checkNan(self, value):
        if math.isnan(value):
            print("ERROR: predicted reward is nan")
            quit()

    def getCellDataOwnPlayer(self, left, top, size):
        cells = self.player.getCells()
        totalCells = len(cells)
        return [self.isRelativeCellData(cells[idx], left, top, size) if idx < totalCells else [0, 0, 0]
                     for idx in range(1)]

    def getTDError(self):
        return self.latestTDerror

    def getReward(self):
        if self.lastMass is None:
            return None
        if not self.player.getIsAlive():
            return -1 * self.lastMass
        currentMass = self.player.getTotalMass()
        reward = currentMass - self.lastMass
        #if abs(reward) < 0.1:
        #    reward -=  1
        return reward


    def getType(self):
        return self.type

    def getPlayer(self):
        return self.player

    def getTrainMode(self):
        return self.trainMode