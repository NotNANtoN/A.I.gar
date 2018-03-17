import numpy
import heapq
import math
import keras
from .parameters import *
from keras.models import Sequential
from keras.layers import Dense, Activation
import os.path


class Memory(object):
    def __init__(self, state, action, reward, new_state, td_error):
        self.state = state
        self.action = action
        self.reward = reward
        self.new_state = new_state
        self.td_error = td_error

    def __lt__(self, other):
        return self.td_error < other.td_error

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
    stateReprLen = 12
    actionLen = 4

    weight_initializer_range = math.sqrt(6 / (stateReprLen + num_actions))

    memoryCapacity = 10000
    memoriesPerUpdate = 32 # Must be divisible by 2 atm due to experience replay
    memories = []

    num_NNbots = 0

    targetNetworkSteps = 500
    discount = 0.99
    epsilon = 0.1
    frameSkipRate = 3
    learningRate = 0.00025

    initializer = keras.initializers.RandomUniform(minval=-weight_initializer_range, maxval=weight_initializer_range, seed=None)

    valueNetwork = Sequential()
    valueNetwork.add(Dense(50, input_dim = stateReprLen, activation ='sigmoid', bias_initializer=initializer
                           , kernel_initializer=initializer))
    valueNetwork.add(Dense(num_actions, activation ='linear', bias_initializer=initializer
                           , kernel_initializer=initializer))
    valueNetwork.compile(loss ='mse', optimizer=keras.optimizers.Adam(lr=learningRate))

    targetNetwork = keras.models.clone_model(valueNetwork)
    targetNetwork.set_weights(valueNetwork.get_weights())

    def __init__(self, player, field, type, expRepEnabled, gridViewEnabled):
        self.expRepEnabled = expRepEnabled
        self.gridViewEnabled = gridViewEnabled
        self.type = type
        self.player = player
        self.field = field
        self.oldState = None
        self.currentAction = None
        self.currentActionIdx = None

        if self.type == "NN":
            self.lastMass = None
            self.reward = None
            self.cumulativeReward = 0
            self.num_NNbots += 1
            self.skipFrames = 0
        else:
            self.splitLikelihood = numpy.random.randint(9950,10000)
            self.ejectLikelihood = numpy.random.randint(9990,10000)
            self.currentAction = [0, 0, 0, 0]

    def update(self):
        if self.type == "NN":
            self.qLearn()
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
        newState = self.getStateRepresentation()

        # Do not train if we are skipping this frame
        if self.skipFrames > 0 :
            self.skipFrames -= 1
            if not self.oldState:
                print("no old state in skip frame")
                quit()
            reward = self.getReward()
            self.cumulativeReward += reward
            self.currentAction[2:4] = [0, 0]
            if alive:
                return

        # Only train when we there is an old state to train
        if self.oldState:
            # Get reward of skipped frames
            reward = self.cumulativeReward

            # Fit value network using only the current experience
            # If the player died, the target is the reward
            input, target, td_error = self.createInputOutputPair(self.oldState, self.currentActionIdx, reward, newState, alive)
            self.valueNetwork.train_on_batch(input, target)


            if self.expRepEnabled:
                # Fit value network using experience replay of random past states:
                self.experienceReplay(reward, newState, td_error)

            if  __debug__ and self.player.getSelected():
                updatedQvalueOfAction = self.valueNetwork.predict(numpy.array([self.oldState]))[0][
                    self.currentActionIdx]
                print("Qvalue of action after training: ", round(updatedQvalueOfAction, 4))
                print("TD-Error: ", td_error)
                print("")


            # Update the target network after 1000 steps
            # Save the weights of the model when updating the target network to avoid losing progress on program crashes
            self.targetNetworkSteps -= 1
            if self.targetNetworkSteps == 0:
                self.targetNetwork.set_weights(self.valueNetwork.get_weights())
                self.targetNetworkSteps = 1000 * self.num_NNbots
                self.valueNetwork.save("mostRecentAutosave.h5")


        if alive:
            self.takeAction(newState)
            self.lastMass = self.player.getTotalMass()
            self.oldState = newState
        else:
            self.currentActionIdx = None
            self.currentAction = None
            self.lastMass = None
            self.oldState = None
            self.skipFrames = 0
            self.cumulativeReward = 0

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
            newMemory = [state, action, reward, newState]
        else:
            newMemory = [state, action, reward, None]
        heapq.heappush(self.memories, ((td_error * td_error) * -1, newMemory))
        #self.memories.append(newMemory)

    def memoryToInputOutput(self, memory):
        s = memory[0]
        a = memory[1]
        r = memory[2]
        sPrime = memory[3]
        alive = (sPrime != None)
        return self.createInputOutputPair(s, a, r, sPrime, alive)

    def train_on_experience(self):
        # Fit value network on memories
        len_memory = len(self.memories)
        if len_memory < self.memoriesPerUpdate:
            return
        inputSize = self.stateReprLen
        outputSize = self.num_actions
        batch_size = self.memoriesPerUpdate
        # Initialize vectors
        inputs = numpy.zeros((batch_size, inputSize))
        targets = numpy.zeros((batch_size, outputSize))
        partial_batch = int(batch_size / 2)
        batch_count = 0
        # Get most surprising memories:
        popped_memories = []
        for idx in range(batch_size):
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
        '''
        # Get random memories
        for idx in range(partial_batch):
            randIdx = numpy.random.randint(len(self.memories))
            memory = self.memories[randIdx][1]
            input, target, td_error = self.memoryToInputOutput(memory)
            inputs[batch_count + idx] = input
            targets[batch_count + idx] = target
            # Update TD-Error of memory:
            self.memories[randIdx] = (td_error, memory)
        '''
        self.valueNetwork.train_on_batch(inputs, targets)

    def takeAction(self, newState):
        # Take random action with probability 1 - epsilon
        if numpy.random.random(1) < self.epsilon:
            self.currentActionIdx = numpy.random.randint(len(self.actions))
        else:
            # Take action based on greediness towards Q values
            qValues = self.valueNetwork.predict(numpy.array([newState]))
            argMax = numpy.argmax(qValues)
            self.currentActionIdx = argMax
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
                #stateRepr = self.getOnlyPelletStateRepresentation()
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

    def getOnlyPelletStateRepresentation(self):
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
        #cellInfos = self.getCellDataOwnPlayer(left, top, size)
        #for info in cellInfos:
        #    totalInfo += info
        # Add data about the closest enemy cell
        #playerCellsInFov = self.field.getEnemyPlayerCellsInFov(self.player)
        #closestEnemyCell = min(playerCellsInFov,
        #                       key=lambda p: p.squaredDistance(firstPlayerCell)) if playerCellsInFov else None
        #totalInfo += self.isRelativeCellData(closestEnemyCell, left, top, size)
        # Add data about the closest pellet
        pelletsInFov = self.field.getPelletsInFov(midPoint, size)
        closestPellet = min(pelletsInFov, key=lambda p: p.squaredDistance(firstPlayerCell)) if pelletsInFov else None
        closestPelletPos = self.getRelativeCellPos(closestPellet, left, top, size)
        totalInfo += closestPelletPos
        # Add data about distances to the visible edges of the field
        #width = self.field.getWidth()
        #height = self.field.getHeight()
        #distLeft = x / size if left <= 0 else 1
        #distRight = (width - x) / size if left + size >= width else 1
        #distTop = y / size if top <= 0 else 1
        #distBottom = (height - y) / size if top + size >= height else 1
        #totalInfo += [distLeft, distRight, distTop, distBottom]
        return totalInfo

    def getGridStateRepresentation(self):
        size = self.player.getFovSize()
        midPoint = self.player.getFovPos()
        x = int(midPoint[0])
        y = int(midPoint[1])
        left = x - int(size / 2)
        top = y - int(size / 2)
        # ATTENTION: We are assuming gridSquares don't have the ability to be rectangular
        gsSize = [size / GRID_COLUMNS_NUMBER, size / GRID_ROWS_NUMBER]  # (gs = grid square)
        gsMidPoint = [left + gsSize[0] / 2, top + gsSize[1] / 2]
        # Pellet vision grid related
        gsPelletProportion = []
        totalPellets = len(self.field.getPelletsInFov(midPoint, size))
        # Mass vision grid related
        gsBiggestEnemyCellMassProportion = []
        playerMass = self.player.getCells()[0].getMass()
        enemyCells = self.field.getEnemyPlayerCellsInFov(self.player)
        # Player cell number vision greed related
        gsEnemyCellCount = []
        totalEnemyCells = len(enemyCells)
        for c in range(GRID_ROWS_NUMBER):
            for r in range(GRID_COLUMNS_NUMBER):
                # Create pellet representation
                # Make the visionGrid's pellet count a percentage so that the network doesn't have to
                # work on interpretting the number of pellets relative to the size (and Fov) of the player
                gridPelletNumber = len(self.field.getPelletsInFov(gsMidPoint, gsSize[0]))
                gsPelletProportion.append(gridPelletNumber / totalPellets if totalPellets != 0 else 0)

                # Create Enemy Cell mass representation
                # Make the visionGrid's enemy cell representation a percentage. The player's mass
                # in proportion to the biggest enemy cell's mass in each grid square.
                gsEnemyCells = self.field.getEnemyPlayerCellsInGivenFov(self.player, gsMidPoint, gsSize[0])
                if gsEnemyCells == []:
                    gsBiggestEnemyCellMassProportion.append(0)
                else:
                    biggestEnemyCellMassInSquare = max(gsEnemyCells, key=lambda p: p.getMass()).getMass()
                    gsBiggestEnemyCellMassProportion.append(playerMass / biggestEnemyCellMassInSquare)

                # Create Enemy Cell number representation
                # Just a grid with number of enemy cells on each square
                gsEnemyCellCount.append(len(gsEnemyCells) / totalEnemyCells if totalEnemyCells != 0 else 0)
                # Increment grid square position horizontally
                gsMidPoint[0] += gsSize[0]
            # Reset horizontal grid square, increment grid square position
            gsMidPoint[0] = left + gsSize[0] / 2
            gsMidPoint[1] += gsSize[1]
        # Collect all relevant data
        totalInfo = gsPelletProportion + gsBiggestEnemyCellMassProportion + gsEnemyCellCount
        totalInfo += [self.player.getCells()[0].getMass()]
        return totalInfo

    def saveModel(self, name = None):
        if name == None:
            decision = int(input("Do you want to give the model a name? (1=yes)"))
            if decision == 1:
                name = input("Enter the name of the model: ")
                self.saveModel(name)
                return
            else:
                path = self.type + "_latestModel.h5"
                print("No specific name chosen, saving model under: ", path )
        else:
            path = name + ".h5"

        if os.path.exists(path):
            decision = 0
            while decision != 1 and decision != 2 and decision != 3:
                decision = int(input("Model with name \'" +  path +
                             "\' already exists. Do you want to overwrite(1) it, save it under a different name(2), or don't save it(3)?\n"))
            if decision == 1:
                self.valueNetwork.save(path)
            elif decision == 2:
                name = input("Enter the changed name: ")
                self.saveModel(name)
            elif decision == 3:
                print("Model of type ", self.type, " not saved!")
            return
        self.valueNetwork.save(path)

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

    def getTDError(self, reward):
        if self.oldState:
            newState = self.getStateRepresentation()
            target = self.calculateTarget(newState, reward, self.player.getIsAlive())
            predictedValue = self.valueNetwork.predict(numpy.array([self.oldState]))[0][self.currentActionIdx]
            return (target - predictedValue)
        else:
            return None

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