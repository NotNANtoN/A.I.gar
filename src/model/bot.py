import heapq
import keras
import numpy
import tensorflow as tf
import importlib.util
from keras.layers import Dense, LSTM, Softmax
from keras.models import Sequential
from keras.utils.training_utils import multi_gpu_model

from .parameters import *
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

    #spec = importlib.util.spec_from_file_location("networkParameters",".networkParameters.py")
    #parameters = importlib.util.module_from_spec(spec)
    #spec.loader.exec_module(parameters)

    parameters =  importlib.import_module('.networkParameters', package="model")
    #model = __import__("model")
    #parameters = model.
    #print(parameters)

    num_actions = len(actions)
    stateReprLen = parameters.STATE_REPR_LEN
    actionLen = 4

    gpus = parameters.GPUS

    # Experience replay:
    memoryCapacity = parameters.MEMORY_CAPACITY
    memoriesPerUpdate = parameters.MEMORIES_PER_UPDATE # Must be divisible by 4 atm due to experience replay
    memories = []

    num_NNbots = 0
    num_Greedybots = 0

    # Q-learning
    targetNetworkSteps = parameters.TARGET_NETWORK_STEPS
    targetNetworkMaxSteps = parameters.TARGET_NETWORK_MAX_STEPS
    discount = parameters.DISCOUNT
    epsilon = parameters.EPSILON
    frameSkipRate = parameters.FRAME_SKIP_RATE
    gridSquaresPerFov = parameters.GRID_SQUARES_PER_FOV # is modified by the user later on anyways

    #ANN
    learningRate = parameters.ALPHA
    optimizer = parameters.OPTIMIZER
    activationFuncHidden = parameters.ACTIVATION_FUNC_HIDDEN
    activationFuncOutput = parameters.ACTIVATION_FUNC_OUTPUT

    hiddenLayer1 = parameters.HIDDEN_LAYER_1
    hiddenLayer2 = parameters.HIDDEN_LAYER_2
    hiddenLayer3 = parameters.HIDDEN_LAYER_3

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
            hidden1 = Dense(cls.hiddenLayer1, input_dim=cls.stateReprLen, activation=cls.activationFuncHidden,
                                       bias_initializer=initializer, kernel_initializer=initializer)

            cls.valueNetwork.add(hidden1)
            #cls.valueNetwork.add(Dropout(0.5))
            if cls.hiddenLayer2 > 0:
                hidden2 = Dense(cls.hiddenLayer2, input_dim=cls.stateReprLen, activation=cls.activationFuncHidden,
                                bias_initializer=initializer, kernel_initializer=initializer)
                cls.valueNetwork.add(hidden2)
                #cls.valueNetwork.add(Dropout(0.5))

            if cls.hiddenLayer3 > 0:
                hidden3 = Dense(cls.hiddenLayer3, input_dim=cls.stateReprLen, activation=cls.activationFuncHidden,
                                bias_initializer=initializer, kernel_initializer=initializer)
                cls.valueNetwork.add(hidden3)
                #cls.valueNetwork.add(Dropout(0.5))

            cls.valueNetwork.add(
                Dense(cls.num_actions, activation=cls.activationFuncOutput, bias_initializer=initializer
                      , kernel_initializer=initializer))

            if cls.parameters.USE_POLICY_NETWORK:
                cls.policyNetwork = Sequential()
                hidden1 = Dense(50, input_dim=cls.stateReprLen, activation='sigmoid',
                                bias_initializer=initializer, kernel_initializer=initializer)
                cls.policyNetwork.add(hidden1)
                out = Dense(cls.num_actions, activation='softmax', bias_initializer=initializer,
                            kernel_initializer=initializer)
                cls.policyNetwork.add(out)


        cls.targetNetwork = keras.models.clone_model(cls.valueNetwork)
        cls.targetNetwork.set_weights(cls.valueNetwork.get_weights())



        if cls.optimizer == "Adam":
            optimizer = keras.optimizers.Adam(lr=cls.learningRate)
        elif cls.optimizer == "SGD":
            optimizer = keras.optimizers.SGD(lr=cls.learningRate)

        cls.valueNetwork.compile(loss='mse', optimizer=optimizer)
        cls.targetNetwork.compile(loss='mse', optimizer=optimizer)
        if cls.parameters.USE_POLICY_NETWORK:
            cls.policyNetwork.compile(loss='mse', optimizer=optimizer)

    def __init__(self, player, field, type, trainMode):
        self.expRepEnabled = self.parameters.EXP_REPLAY_ENABLED
        self.gridViewEnabled = self.parameters.GRID_VIEW_ENABLED
        self.trainMode = trainMode
        self.type = type
        self.player = player
        self.field = field
        if self.type == "Greedy":
            self.splitLikelihood = 100000 #numpy.random.randint(9950,10000)
            self.ejectLikelihood = 100000 #numpy.random.randint(9990,10000)
        self.totalMasses = []
        self.reset()

    def reset(self):
        self.lastMass = None
        self.oldState = None
        self.latestTDerror = None
        self.lastMemory = None
        self.skipFrames = 0
        self.cumulativeReward = 0
        self.lastReward = 0
        self.rewardAvgOfEpisode = 0
        self.rewardLenOfEpisode = 0
        #self.actionHistory = [[0,0,0,0]]
        if self.type == "NN":
            self.currentActionIdx = None
            self.currentAction = None
        elif self.type == "Greedy":
            self.currentAction = [0, 0, 0, 0]


    def makeMove(self):
        self.totalMasses.append(self.player.getTotalMass())
        if self.type == "NN":
            if self.trainMode:
                self.qLearn()
                #self.epsilon = self.epsilon * EPSILON_DECREASE_RATE
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

    def createInputOutputPair(self, oldState, actionIdx, reward, newState, alive, verbose = False):
        state_Q_values = self.valueNetwork.predict(numpy.array([oldState]))[0]
        target = self.calculateTarget(newState, reward, alive)

        td_error = target - state_Q_values[actionIdx]
        if  __debug__ and self.player.getSelected() and verbose:
            print("")
            #print("State to be updated: ", oldState)
            print("Action: ", self.actions[actionIdx])
            print("Reward: " ,round(reward, 2))
            #print("S\': ", newState)
            print("Qvalue of action before training: ", round(state_Q_values[actionIdx], 4))
            print("Target Qvalue of that action: ", round(target, 4))
            print("All qvalues: ", numpy.round(state_Q_values, 3))
            print("Expected Q-value: ", round(max(state_Q_values), 3))
            print("TD-Error: ", td_error)
        if self.parameters.USE_TARGET:
            state_Q_values[actionIdx] = target
        else:
            state_Q_values[actionIdx] = td_error
        return numpy.array([oldState]), numpy.array([state_Q_values]), td_error

    def qLearn(self):
        #After S has been initialized, set S as oldState and take action A based on policy
        alive = self.player.getIsAlive()
        self.cumulativeReward += self.getReward() if self.lastMass else 0
        self.lastReward = self.cumulativeReward
        if alive:
            self.rewardAvgOfEpisode = (self.rewardAvgOfEpisode * self.rewardLenOfEpisode + self.lastReward)\
                                      / (self.rewardLenOfEpisode + 1)
            self.rewardLenOfEpisode += 1
        # Do not train if we are skipping this frame
        if self.skipFrames > 0 :
            self.skipFrames -= 1
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
                                                                 newState, alive, True)
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


        # If the player is alive then save the action, state and mass of this update
        if self.player.getIsAlive():
            self.takeAction(newState)
            self.lastMass = self.player.getTotalMass()
            self.oldState = newState
        # Otherwise reset values to start a new episode for this actor
        else:
            print(self.player, " died.")
            print("Average reward of ", self.player, " for this episode: ", self.rewardAvgOfEpisode)
            self.reset()

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
            del self.memories[-1]
            #else:
            #    self.memories.remove(min(self.memories, key = lambda memory: abs(memory[-1])))
        if self.player.getIsAlive():
            newMemory = [state.tolist(), action, reward, newState.tolist()]
        else:
            newMemory = [state.tolist(), action, reward, None]
        # Square the TD-error and multiply by minus one, because the heap pops the smallest number
        heapq.heappush(self.memories, ((td_error * td_error) * -1, newMemory, self.lastMemory))
        self.lastMemory = newMemory

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
        inputSize = self.parameters.STATE_REPR_LEN
        outputSize = self.num_actions
        batch_size = self.memoriesPerUpdate
        # Initialize vectors
        inputs = numpy.zeros((batch_size, inputSize))
        targets = numpy.zeros((batch_size, outputSize))
        batch_count = 0
        # Get most surprising memories:
        popped_memories = []
        for idx in range(int(batch_size * 0)):
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

        # Get recent memories:
        # WARNING: this does not acutally get the most recent memories, so we have to change it
        # TODO: change it
        for idx in range(int(batch_size * 0)):
            memory = self.memories[len_memory - idx - 1][1]
            input, target, td_error = self.memoryToInputOutput(memory)
            inputs[batch_count] = input
            targets[batch_count] = target
            # Update TD-Error of memory:
            self.memories[idx] = (td_error, memory)
            batch_count += 1

        # Fill up the rest of the batch with random memories:
        while batch_count < batch_size:
            randIdx = numpy.random.randint(len(self.memories))
            memory = self.memories[randIdx][1]
            input, target, td_error = self.memoryToInputOutput(memory)
            inputs[batch_count] = input
            targets[batch_count] = target
            # Update TD-Error of memory:
            self.memories[randIdx] = (td_error, memory)
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
            if self.parameters.USE_POLICY_NETWORK:
                numpyNewState = numpy.array([newState])
                qValues = self.valueNetwork.predict(numpyNewState)
                qValueSum = sum(qValues)
                normalizedQValues = numpy.array([qValue / qValueSum for qValue in qValues])
                self.policyNetwork.train_on_batch(numpyNewState, normalizedQValues)
                actionValues = self.policyNetwork.predict(numpyNewState)
                self.currentActionIdx = numpy.argmax(actionValues)
            else:
                # Take action based on greediness towards Q values
                qValues = self.valueNetwork.predict(numpy.array([newState]))
                self.currentActionIdx = numpy.argmax(qValues)
                if __debug__:
                    self.player.setExploring(False)
        self.currentAction = self.actions[self.currentActionIdx]
        self.skipFrames = self.frameSkipRate
        self.cumulativeReward = 0


        # # #Testing implementation of action history
        # self.actionHistory.insert(0,self.currentAction)
        # self.actionHistory.pop(len(self.actionHistory)-1)


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
        #enemyCellsCount = len(enemyCells)
        #allCellsInFov = playerCells + enemyCells + virusCells
        #biggestMassInFov = max(allCellsInFov, key = lambda cell: cell.getMass()).getMass() if allCellsInFov else None

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
                        gsPelletProportion[count] = pelletMassSum

                    # TODO: add relative fov pos of closest pellet to allow micro management

                    # Create Enemy Cell mass representation
                    # Make the visionGrid's enemy cell representation a percentage. The player's mass
                    # in proportion to the biggest enemy cell's mass in each grid square.
                    enemiesInGS = enemySHT.getBucketContent(count)
                    if enemiesInGS:
                        biggestEnemyInCell = max(enemiesInGS, key = lambda cell: cell.getMass())
                        gsBiggestEnemyCellMassProportion[count] = biggestEnemyInCell.getMass()

                    # Create Own Cell mass representation
                    playerCellsInGS = playerSHT.getBucketContent(count)
                    if playerCellsInGS:
                        biggestFriendInCell = max(playerCellsInGS, key=lambda cell: cell.getMass())
                        gsBiggestOwnCellMassProportion[count] = biggestFriendInCell.getMass()
                    # TODO: also add a count grid for own cells?

                    # Create Virus Cell representation
                    virusesInGS = virusSHT.getBucketContent(count)
                    if virusesInGS:
                        biggestVirus = max(virusesInGS, key = lambda virus: virus.getRadius()).getMass()
                        gsVirus[count] = biggestVirus

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
        # Concatenate the basis data about the location of pellets, own cells and the walls:
        totalInfo = numpy.concatenate((gsPelletProportion, gsBiggestOwnCellMassProportion, gsWalls))
        # If there are other players in the game, the bot needs information about them:
        if len(self.field.getPlayers() ) > 1:
            totalInfo = numpy.concatenate((totalInfo, gsBiggestEnemyCellMassProportion))
        # If there are viruses in the game, the bot needs to know their location
        if self.field.getVirusEnabled():
            totalInfo = numpy.concatenate((totalInfo, gsVirus))
        # Add total Mass of player and field size:
        totalMass = self.player.getTotalMass()
        totalInfo = numpy.concatenate((totalInfo, [totalMass, fovSize]))

        #Testing implementation of action memory
        #totalInfo = numpy.concatenate((totalInfo, numpy.array(self.actionHistory).flatten()))
        # for i in range(GRID_SQUARES_PER_FOV):
        #     print(str(gsPelletProportion[GRID_SQUARES_PER_FOV*i:GRID_SQUARES_PER_FOV*(i+1)]) + " "
        #           + str(gsBiggestOwnCellMassProportion[GRID_SQUARES_PER_FOV*i:GRID_SQUARES_PER_FOV*(i+1)]) + " "
        #           + str(gsWalls[GRID_SQUARES_PER_FOV*i:GRID_SQUARES_PER_FOV*(i+1)]))
        # print("\n")
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

    def getMassOverTime(self):
        return self.totalMasses

    def getAvgReward(self):
        return self.rewardAvgOfEpisode

    def getLastReward(self):
        return self.lastReward

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
