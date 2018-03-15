import numpy
import math
import keras
from .parameters import *
from keras.models import Sequential
from keras.layers import Dense, Activation
import os.path

class Bot(object):
    """docstring for Bot"""
    # Create all possible discrete actions
    actions = [[x, y, split, eject] for x in [0, 0.5, 1] for y in [0, 0.5, 1] for split in [0, 1] for
               eject in [0, 1]]
    # Filter out actions that do a split and eject at the same time
    for action in actions[:]:
        if action[2] and action[3]:
            actions.remove(action)
    num_actions = len(actions)
    stateReprLen = 12
    actionLen = 4

    valueNetwork = Sequential()
    valueNetwork.add(Dense(50, input_dim = stateReprLen, activation ='sigmoid'))
    valueNetwork.add(Dense(num_actions, activation ='linear'))
    valueNetwork.compile(loss ='mean_squared_error', optimizer=keras.optimizers.SGD(lr=0.001))

    memoryCapacity = 1000
    memoriesPerUpdate = 25
    memories = []

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
            self.discount = 0.9
            self.epsilon = 0.9
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
        target = reward
        if alive:
            # The target is the reward plus the discounted prediction of the value network
            action_Q_values = self.valueNetwork.predict(numpy.array([newState]))[0]
            newActionIdx = numpy.argmax(action_Q_values)
            target += self.discount * action_Q_values[newActionIdx]
        return target

    def createInputOutputPair(self, oldState, actionIdx, reward, newState, alive):
        state_Q_values = self.valueNetwork.predict(numpy.array([oldState]))[0]
        target = self.calculateTarget(newState, reward, alive)
        if abs(round(reward, 2)) > 1.5:
            print("state to be updated: ", oldState)
            print("reward: " ,round(reward, 2))
            print("Qvalue of action before trainig: ", round(state_Q_values[actionIdx], 4))
            print("Target Qvalue of that action: ", round(target, 4))
            print("All qvalues: ", numpy.round(state_Q_values, 3))
        state_Q_values[actionIdx] = target
        return numpy.array([oldState]), numpy.array([state_Q_values])

    def qLearn(self):
        #After S has been initialized, set S as oldState and take action A based on policy
        alive = self.player.getIsAlive()
        newState = self.getStateRepresentation()

        if self.oldState:
            # Get current reward
            reward = self.getReward()
            # Either use experience replay or fit on current experience
            if self.expRepEnabled:
                # Fit value network using experience replay of random past states:
                self.experienceReplay(reward, newState)
            else:
                # Fit value network using only the current experience
                # If the player died, the target is the reward
                input, target = self.createInputOutputPair(self.oldState, self.currentActionIdx, reward, newState, alive)
                self.valueNetwork.train_on_batch(input, target)

                updatedQvalueOfAction = self.valueNetwork.predict(numpy.array([self.oldState]))[0][self.currentActionIdx]
                if abs(round(reward, 2)) > 1.5:
                    print("Qvalue of action after training: ", round(updatedQvalueOfAction, 4))
                    print("")

        if alive:
            self.takeAction(newState)
            self.lastMass = self.player.getTotalMass()
            self.oldState = newState
        else:
            self.lastMass = None
            self.oldState = None

    def experienceReplay(self, reward, newState):
        if self.player.getIsAlive():
            self.remember(self.oldState, self.currentActionIdx, reward, newState)
        self.train_on_experience()

    def remember(self, state, action, reward, newState):
        # Store current state, action, reward, state pair in memory
        # Delete oldest memory if memory is at full capacity
        if len(self.memories) > self.memoryCapacity:
            if numpy.random.random() > 0.0:
                del self.memories[0]
            else:
                self.memories.remove(min(self.memories, key = lambda memory: abs(memory[2])))
        if not self.player.getIsAlive():
            self.memories.append([state, action, reward, None])
        else:
            self.memories.append([state, action, reward, newState])

    def train_on_experience(self):
        len_memory = len(self.memories)
        inputSize = self.stateReprLen
        outputSize = self.num_actions
        training_memory_count = min(self.memoriesPerUpdate, len_memory)
        # Fit value network on memories
        inputs = numpy.zeros((training_memory_count, inputSize))
        targets = numpy.zeros((training_memory_count, outputSize))
        for idx in range(training_memory_count):
            # Get random memory
            memory = self.memories[numpy.random.randint(len(self.memories))]
            s = memory[0]
            a = memory[1]
            r = memory[2]
            sPrime = memory[3]
            alive = (sPrime != None)
            input, target = self.createInputOutputPair(s, a, r, sPrime, alive)
            inputs[idx] = input
            targets[idx] = target
        self.valueNetwork.train_on_batch(inputs, targets)

    def takeAction(self, newState):
        # Take random action with probability 1 - epsilon
        if numpy.random.random(1) > self.epsilon:
            self.currentActionIdx = numpy.random.randint(len(self.actions))
        else:
            # Take action based on greediness towards Q values
            qValues = self.valueNetwork.predict(numpy.array([newState]))
            argMax = numpy.argmax(qValues)
            self.currentActionIdx = argMax
        self.currentAction = self.actions[self.currentActionIdx]

    def getStateRepresentation(self):
        if self.player.getIsAlive():
            if self.gridViewEnabled:
                return self.getGridStateRepresentation()
            else:
                return self.getSimpleStateRepresentation()
        else:
            return None

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

    def printRewardInfo(self, reward, predictedReward, oppositeAction, predictedOppositeReward):
        if self.oldState:
            print("state: ", end=" ")
            for number in self.oldState:
                print(round(number, 2), end=" ")
            print(" ")
        if self.currentAction:
            print("currentAction: ", self.currentAction)
            print("oppositeAction: ", oppositeAction)
        print("reward: ", round(reward, 5))
        print("predicted reward (Q(s,a)): ", predictedReward)
        print("predicted rewar opposite: ", predictedOppositeReward)

    def setEpsilon(self, val):
        self.epsilon = val

    def generateOppositeAction(self, action):
        oppositeAction = [0, 0, 0, 0]
        oppositeAction[0] = abs(1 -action[0])
        oppositeAction[1] = abs(1 - action[1])
        oppositeAction[2:4] = action[2:4]
        return oppositeAction

    def isCellData(self, cell):
        return [cell.getX(), cell.getY(), cell.getRadius()]

    def isRelativeCellData(self, cell, left, top, size):
        return self.getRelativeCellPos(cell, left, top, size) + \
               ([cell.getRadius() / size if cell.getRadius() <= size else 1] if cell != None else [0])

    def getRelativeCellPos(self, cell, left, top, size):
        if cell != None:
            return [(cell.getX() - left) / size, (cell.getY() - top) / size]
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
        if self.oldState and self.player.getIsAlive():
            newState = self.getStateRepresentation()
            target = self.calculateTarget(newState, reward, True)
            predictedValue = self.valueNetwork.predict(numpy.array([self.oldState]))[0][self.currentActionIdx]
            return target - predictedValue
        return None

    def getReward(self):
        if not self.player.getIsAlive():
            return 0
        return self.player.getTotalMass()

        '''
        if self.lastMass is None:
            return None
        if not self.player.getIsAlive():
            return -1 * self.lastMass
        currentMass = self.player.getTotalMass()
        reward = currentMass - self.lastMass
        if abs(reward) < 0.1:
            reward -=  1
        return reward
        '''

    def getType(self):
        return self.type

    def getPlayer(self):
        return self.player