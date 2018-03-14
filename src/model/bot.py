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

    stateReprLen = 12
    actionLen = 4

    valueNetwork = Sequential()
    valueNetwork.add(Dense(50, input_dim= stateReprLen + actionLen, activation='relu'))
    #valueNetwork.add(Dense(30,  activation='relu'))
    # self.valueNetwork.add(Dense(10, activation = 'relu'))
    valueNetwork.add(Dense(1, activation='linear'))
    valueNetwork.compile(loss='mean_squared_error',
                              optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True))


    memoryCapacity = 1000
    memoriesPerUpdate = 25
    memories = []

    valueNetwork2 = Sequential()
    valueNetwork2.add(Dense(50, input_dim= stateReprLen + actionLen, activation='relu',
                           bias_initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None),
                           kernel_initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)))
    valueNetwork2.add(Dense(25, activation='relu',
                        bias_initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None),
                        kernel_initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)))
    # self.valueNetwork.add(Dense(10, activation = 'relu'))
    valueNetwork2.add(Dense(1, activation='linear', bias_initializer = keras.initializers.TruncatedNormal(mean=0.0,
                        stddev=0.05, seed=None),))
    valueNetwork2.compile(loss='mean_squared_error',
                              optimizer=keras.optimizers.SGD(lr=0.5, momentum=0.0, nesterov=True))

    def __init__(self, player, field, type, expRepEnabled, gridViewEnabled):
        self.expRepEnabled = expRepEnabled
        self.gridViewEnabled = gridViewEnabled
        self.type = type
        self.player = player
        self.field = field
        self.oldState = None
        self.currentAction = [0, 0, 0, 0]

        if self.type == "NN":
            self.lastMass = None
            self.reward = None
            self.discount = 0.9
            self.epsilon = 0.975
            self.exploring = 0
            self.exploreStepsMax = 10
        else:
            self.splitLikelihood = numpy.random.randint(9950,10000)
            self.ejectLikelihood = numpy.random.randint(9990,10000)

    def isCellData(self, cell):
        return [cell.getX(), cell.getY(), cell.getRadius()]

    def isRelativeCellData(self, cell, left, top, size):
        return self.getRelativeCellPos(cell, left, top, size) + [cell.getRadius() / size if cell.getRadius() <= size else 1]


    def getRelativeCellPos(self, cell, left, top, size):
        return [(cell.getX() - left) / size, (cell.getY() - top) / size]

    def getSimpleStateRepresentation(self):
        size = self.player.getFovSize()
        midPoint = self.player.getFovPos()
        x = int(midPoint[0])
        y = int(midPoint[1])
        left = x - int(size / 2)
        top = y - int(size / 2)
        size = int(size)
        pelletsInFov = self.field.getPelletsInFov(midPoint, size)
        playerCellsInFov = self.field.getEnemyPlayerCellsInFov(self.player)
        firstPlayerCell = self.player.getCells()[0]
        closestEnemyCell = min(playerCellsInFov,
                               key=lambda p: p.squaredDistance(firstPlayerCell)) if playerCellsInFov else None
        if closestEnemyCell:
            maximumCellMass = max([closestEnemyCell.getMass(), firstPlayerCell.getMass()])
        else:
            maximumCellMass = firstPlayerCell.getMass()
        cells = self.player.getCells()
        totalCells = len(cells)
        cellInfos = [self.isRelativeCellData(cells[idx], left, top, size) if idx < totalCells else [0, 0, 0]
            for idx in range(1)]
        # cellInfoTransform should have length 48: three values for 16 cells. Or now length 3, because we are only
        # handling the first cell to test it
        totalInfo = []
        for info in cellInfos:
            totalInfo += info
        if closestEnemyCell == None:
            totalInfo += [0, 0, 0]
        else:
            totalInfo += self.isRelativeCellData(closestEnemyCell, left, top, size)
        closestPelletPos = self.getRelativeCellPos(min(pelletsInFov, key=lambda p: p.squaredDistance(firstPlayerCell)),
                                                   left, top,
                                                   size) if pelletsInFov else [0, 0]
        totalInfo += closestPelletPos
        width = self.field.getWidth()
        height = self.field.getHeight()
        distLeft = x / size if left <= 0 else 1
        distRight = (width - x) / size if left + size >= width else 1
        distTop = y / size if top <= 0 else 1
        distBottom = (height - y) / size if top + size >= height else 1
        totalInfo += [distLeft, distRight, distTop, distBottom]
        return totalInfo

    def remember(self, state, action, reward, newState):
        # Store current state, action, reward, state pair in memory
        # Delete oldest memory if memory is at full capacity
        if len(self.memories) > self.memoryCapacity:
            if numpy.random.random() > 0.0:
                del self.memories[0]
            else:
                self.memories.remove(min(self.memories, key = lambda memory: abs(memory[2])))
        if not self.player.getIsAlive():
            self.memories.append([state, action, reward, numpy.array([])])
        else:
            self.memories.append([state, action, reward, newState])

    def getGridStateRepresentation(self):
        size = self.player.getFovSize()
        midPoint = self.player.getFovPos()
        x = int(midPoint[0])
        y = int(midPoint[1])
        left = x - int(size / 2)
        top = y - int(size / 2)
        # ATTENTION: We are assuming gridSquares don't have the ability to be rectangular
        gsSize = [size/GRID_COLUMNS_NUMBER, size/GRID_ROWS_NUMBER] #(gs = grid square)
        gsMidPoint = [left + gsSize[0]/2, top + gsSize[1]/2]
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
                gsPelletProportion.append(gridPelletNumber/totalPellets if totalPellets != 0 else 0)
                
                # Create Enemy Cell mass representation
                # Make the visionGrid's enemy cell representation a percentage. The player's mass
                # in proportion to the biggest enemy cell's mass in each grid square.
                gsEnemyCells = self.field.getEnemyPlayerCellsInGivenFov(self.player, gsMidPoint, gsSize[0])
                if gsEnemyCells == []:
                    gsBiggestEnemyCellMassProportion.append(0)
                else:
                    biggestEnemyCellMassInSquare = max(gsEnemyCells, key = lambda p: p.getMass()).getMass()
                    gsBiggestEnemyCellMassProportion.append(playerMass/biggestEnemyCellMassInSquare)
                
                # Create Enemy Cell number representation
                # Just a grid with number of enemy cells on each square
                gsEnemyCellCount.append(len(gsEnemyCells)/totalEnemyCells if totalEnemyCells !=0 else 0)
                # Increment grid square position horizontally
                gsMidPoint[0] += gsSize[0]
            # Reset horizontal grid square, increment grid square position
            gsMidPoint[0] = left + gsSize[0]/2
            gsMidPoint[1] += gsSize[1]
        # Collect all relevant data
        totalInfo = gsPelletProportion + gsBiggestEnemyCellMassProportion + gsEnemyCellCount
        totalInfo += [self.player.getCells()[0].getMass()]
        return totalInfo

    def qLearn(self):
        #After S has been initialized, set S as oldState and take action A based on policy
        if self.oldState == None:
            self.lastMass = self.player.getTotalMass()
            if self.gridViewEnabled:
                newState = self.getGridStateRepresentation()
            else:
                newState = self.getSimpleStateRepresentation()
            self.currentAction = max(self.actions, key=lambda p: self.valueNetwork.predict(numpy.array([p + newState])))
        else:
            # Get current State, Reward and the old State
            reward = self.getReward()
            newState = None
            if self.player.getIsAlive():
                if self.gridViewEnabled:
                    newState = self.getGridStateRepresentation()
                else:
                    newState = self.getSimpleStateRepresentation()

            oppositeAction = [0, 0, 0, 0]
            oppositeAction[0] = abs(1 - self.currentAction[0])
            oppositeAction[1] = abs(1 - self.currentAction[1])
            oppositeAction[2:4] = self.currentAction[2:4]
            predictedReward = round(self.valueNetwork.predict(numpy.array([self.oldState + self.currentAction]))[0][0],
                                    5)
            predictedOppositeReward = round(
                self.valueNetwork.predict(numpy.array([self.oldState + oppositeAction]))[0][0], 5)

            if round(reward, 1) > 1.1 or round(reward, 1) < -1.1:
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
                print("predicted reward opposite: ", predictedOppositeReward)
                if math.isnan(predictedReward):
                    print("ERROR: predicted reward is nan")
                    quit()

            if self.expRepEnabled:
                # Fit value network using experience replay of random past states:
                self.experienceReplay(reward, newState)

            else:
                # Fit value network using only the current experience
                # If the player died, the target is the reward
                if not self.player.getIsAlive():
                    target = numpy.array([reward])
                else:
                    newAction = max(self.actions, key=lambda p: self.valueNetwork.predict(numpy.array([p + newState])))
                    qValueNew = self.valueNetwork.predict(numpy.array([newAction + newState]))
                    target = reward + self.discount * qValueNew
                if round(reward, 1) > 1.1 or round(reward, 1) < -1.1:
                    print("target: ", target[0])
                #self.valueNetwork.fit(numpy.array([self.oldState + self.currentAction]), target, verbose=0)
                self.valueNetwork.train_on_batch(numpy.array([self.oldState + self.currentAction]), target)
            if round(reward, 1) > 1.1 or round(reward, 1) < -1.1:
                updatedPrediction = round(self.valueNetwork.predict(numpy.array([self.oldState
                                                                   + self.currentAction]))[0][0], 5)

                print("delta prediction: ", round(updatedPrediction - predictedReward,6))
                updatedOppositePrediction = round(self.valueNetwork.predict(numpy.array([self.oldState
                                                                   + oppositeAction]))[0][0], 5)
                print("delta opposite: ", round(updatedOppositePrediction - predictedOppositeReward,6))

                print("")
            if self.player.getIsAlive():
                self.takeAction(newState)


        if not self.player.getIsAlive():
            self.oldState = None
        else:
            self.oldState = newState

    def experienceReplay(self, reward, newState):
        if self.player.getIsAlive():
            self.remember(self.oldState, self.currentAction, reward, newState)
        len_memory = len(self.memories)
        inputSize = len(self.oldState) + len(self.currentAction)
        outputSize = 1
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
            target = r
            # If the memory state is not final, then sPrime is not empty, so we need to adjust the target:
            if sPrime:
                aPrime = max(self.actions, key=lambda p: self.valueNetwork.predict(numpy.array([p + sPrime])))
                qValueNew = self.valueNetwork.predict(numpy.array([aPrime + sPrime]))
                target += self.discount * qValueNew
            inputs[idx] =  s + a
            targets[idx] = target
        self.valueNetwork.train_on_batch(inputs, targets)
        #self.valueNetwork.fit(inputs, targets)


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

    def takeAction(self, newState):
        if numpy.random.random(1) > self.epsilon and not self.exploring:
            self.exploring = self.exploreStepsMax
            self.currentAction = self.actions[numpy.random.randint(len(self.actions))]
        elif self.exploring:
            self.currentAction[2:4] = [0, 0]
            self.exploring -= 1
        else:
            qValues = [self.valueNetwork.predict(numpy.array([action + newState])) for action in self.actions]
            maxIndex = numpy.argmax(qValues)
            self.currentAction = self.actions[maxIndex]
        


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

    def getReward(self):
        if not self.player.getIsAlive():
            reward = -1 * self.lastMass
            self.lastMass = None
            return reward
        currentMass = self.player.getTotalMass()
        reward = currentMass - self.lastMass
        self.lastMass = currentMass
        if abs(reward) < 0.1:
            return reward - 1
        return reward * 10

    def getType(self):
        return self.type

    def getPlayer(self):
        return self.player