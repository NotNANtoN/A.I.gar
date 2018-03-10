import numpy
import math
import keras
from .parameters import *
from keras.models import Sequential
from keras.layers import Dense, Activation

class Bot(object):
    """docstring for Bot"""
    # Create all possible discrete actions
    actions = [[x, y, split, eject] for x in [0, 0.5, 1] for y in [0, 0.5, 1] for split in [0, 1] for
               eject in [0, 1]]
    # Filter out actions that do a split and eject at the same time
    for action in actions[:]:
        if action[2] and action[3]:
            actions.remove(action)

    stateReprLen = 6
    actionLen = 4

    valueNetwork = Sequential()
    valueNetwork.add(Dense(stateReprLen + actionLen, input_dim= stateReprLen + actionLen, activation='relu',
                           bias_initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None),
                           kernel_initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)))
    valueNetwork.add(Dense(int((stateReprLen + actionLen) / 3), activation='relu',
                        bias_initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None),
                        kernel_initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)))
    # self.valueNetwork.add(Dense(10, activation = 'relu'))
    valueNetwork.add(Dense(1, activation='linear', bias_initializer = keras.initializers.TruncatedNormal(mean=0.0,
                        stddev=0.05, seed=None),))
    valueNetwork.compile(loss='mean_squared_error',
                              optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.8, nesterov=True))


    memoryCapacity = 200
    memoriesPerUpdate = 10
    memories = []

    valueNetwork2 = Sequential()
    valueNetwork2.add(Dense(stateReprLen + actionLen, input_dim= stateReprLen + actionLen, activation='relu',
                           bias_initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None),
                           kernel_initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)))
    valueNetwork2.add(Dense(int((stateReprLen + actionLen) / 3), activation='relu',
                        bias_initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None),
                        kernel_initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)))
    # self.valueNetwork.add(Dense(10, activation = 'relu'))
    valueNetwork2.add(Dense(1, activation='linear', bias_initializer = keras.initializers.TruncatedNormal(mean=0.0,
                        stddev=0.05, seed=None),))
    valueNetwork2.compile(loss='mean_squared_error',
                              optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.0, nesterov=True))

    def __init__(self, player, field, type, expRepEnabled):
        self.expRepEnabled = expRepEnabled
        self.policyNetwork = None
        self.type = type
        self.player = player
        self.field = field
        self.oldState = None
        self.lastAction = [0,0,0,0]

        if self.type == "NN":
            self.lastMass = None
            self.reward = None
            self.discount = 0.99
            self.epsilon = 0.9
            self.lastAction = None
        else:
            self.splitLikelihood = numpy.random.randint(9950,10000)
            self.ejectLikelihood = numpy.random.randint(9990,10000)

    def isCellData(self, cell):
        return [cell.getX(), cell.getY(), cell.getMass()]

    def isRelativeCellData(self, cell, left, top, size, totalMass):
        return self.getRelativeCellPos(cell, left, top, size) + [cell.getMass() / totalMass]


    def getRelativeCellPos(self, cell, left, top, size):
        return [(cell.getX() - left) / size, (cell.getY() - top) / size]

    def getStateRepresentation(self):
        size = self.player.getFovSize()
        midPoint = self.player.getFovPos()
        x = int(midPoint[0])
        y = int(midPoint[1])
        left = x - int(size / 2)
        top = y - int(size / 2)
        size = int(size)
        pelletsInFov = self.field.getPelletsInFov(midPoint, size)
        closestPelletPos = self.getRelativeCellPos(max(pelletsInFov, key=lambda p: p.getMass()), left, top,
                                                   size) if pelletsInFov else [0, 0]
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
        cellInfos = [self.isRelativeCellData(cells[idx], left, top, size, maximumCellMass) if idx < totalCells else [0, 0, 0]
            for idx in range(1)]
        # cellInfoTransform should have length 48: three values for 16 cells. Or now length 3, because we are only
        # handling the first cell to test it
        totalInfo = []
        #for info in cellInfos:
        #    totalInfo += info
        totalInfo += [firstPlayerCell.getMass() / maximumCellMass]
        if closestEnemyCell == None:
            totalInfo += [0, 0, 0]
        else:
            totalInfo += self.isRelativeCellData(closestEnemyCell, left, top, size, maximumCellMass)
        totalInfo += closestPelletPos
        return totalInfo

    def remember(self, state, action, reward, newState):
        # Store current state, action, reward, state pair in memory
        # Delete oldest memory if memory is at full capacity
        if len(self.memories) > self.memoryCapacity:
            if numpy.random.random() > 0.8:
                del self.memories[0]
            else:
                self.memories.remove(min(self.memories, key = lambda memory: abs(memory[2])))
        if not self.player.getIsAlive():
            self.memories.append([state, action, reward, numpy.array([])])
        else:
            self.memories.append([state, action, reward, newState])

    def qLearn(self):
        #After S has been initialized, set S as oldState and take action A based on policy
        if self.oldState == None:
            self.lastMass = self.player.getTotalMass()
            newState = self.getStateRepresentation()
            self.lastAction = max(self.actions, key=lambda p: self.valueNetwork.predict(numpy.array([p + newState])))
        else:
            # Get current State, Reward and the old State
            reward = self.getReward()
            if self.player.getIsAlive():
                newState = self.getStateRepresentation()
            if round(reward, 2) > 0.1 or round(reward, 2) < -0.1:
                if self.player.getIsAlive():
                    print("state: ", end=" ")
                    for number in newState:
                        print(round(number, 2), end=" ")
                    print(" ")
                print("reward: ", round(reward, 2))
                print(" ")
            if self.expRepEnabled:
                # Fit value network using experience replay of random past states:
                self.experienceReplay(reward)
            else:
                # Fit value network using only the current experience


                # If the player died, the target is the reward
                if not self.player.getIsAlive():
                    target = numpy.array([reward])
                else:
                    newAction = max(self.actions, key=lambda p: self.valueNetwork.predict(numpy.array([p + newState])))
                    qValueNew = self.valueNetwork.predict(numpy.array([newAction + newState]))
                    target = reward + self.discount * qValueNew
                self.valueNetwork.fit(numpy.array([self.oldState + self.lastAction]), target, verbose=0)
                if self.player.getIsAlive():
                    # Choose actions according to policy:
                    if numpy.random.random(1) > self.epsilon:
                        self.lastAction = self.actions[numpy.random.randint(len(self.actions))]
                    else:
                        qValues = [self.valueNetwork.predict(numpy.array([action + newState])) for action in self.actions]
                        maxIndex = numpy.argmax(qValues)
                        self.lastAction = self.actions[maxIndex]
        if not self.player.getIsAlive():
            self.oldState = None
        else:
            self.oldState = newState

    def experienceReplay(self, reward):
        if self.player.getIsAlive():
            newState = self.getStateRepresentation()
            self.remember(self.oldState, self.lastAction, reward, newState)
        len_memory = len(self.memories)
        inputSize = len(self.oldState) + len(self.lastAction)
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
            # If the memory state is not final, then sPrime is not empty:
            if sPrime:
                aPrime = max(self.actions, key=lambda p: self.valueNetwork.predict(numpy.array([p + sPrime])))
                qValueNew = self.valueNetwork.predict(numpy.array([aPrime + sPrime]))
                target += self.discount * qValueNew
            inputs[idx] =  s + a
            targets[idx] = target
        self.valueNetwork.train_on_batch(inputs, targets)


    def update(self):
        if self.type == "NN":
            self.qLearn()
        elif self.type == "Greedy":
            if self.player.getIsAlive():
                midPoint = self.player.getFovPos()
                size = self.player.getFovSize()
                cellsInFov = self.field.getPelletsInFov(midPoint, size)

                playerCellsInFov = self.field.getEnemyPlayerCellsInFov(self.player)
                firstPlayerCell = self.player.getCells()[0]
                for opponentCell in playerCellsInFov:
                    # If the single celled bot can eat the opponent cell add it to list
                    if firstPlayerCell.getMass() > 1.25 * opponentCell.getMass():
                        cellsInFov.append(opponentCell)
                if cellsInFov:
                    bestCell = max(cellsInFov, key = lambda p: p.getMass() / (p.squaredDistance(firstPlayerCell) if p.squaredDistance(firstPlayerCell) != 0 else 1))
                    bestCellPos = bestCell.getPos()
                    self.lastAction[0] = bestCellPos[0]
                    self.lastAction[1] = bestCellPos[1]
                else:
                    size = int(size / 2)
                    self.lastAction[0] = numpy.random.random()
                    self.lastAction[1] = numpy.random.random()
                randNumSplit = numpy.random.randint(0,10000)
                randNumEject = numpy.random.randint(0,10000)
                self.lastAction[2] = False
                self.lastAction[3] = False
                if randNumSplit > self.splitLikelihood:
                    self.lastAction[2] = True
                if randNumEject > self.ejectLikelihood:
                    self.lastAction[3] = True

        if self.player.getIsAlive():
            midPoint = self.player.getFovPos()
            size = self.player.getFovSize()
            x = int(midPoint[0])
            y = int(midPoint[1])
            left = x - int(size / 2)
            top = y - int(size / 2)
            size = int(size)
            xChoice = left + self.lastAction[0] * size
            yChoice = top + self.lastAction[1] * size
            splitChoice = True if self.lastAction[2] > 0.5 else False
            ejectChoice = True if self.lastAction[3] > 0.5 else False

            self.player.setCommands(xChoice, yChoice, splitChoice, ejectChoice)

    def saveModel(self):
        self.valueNetwork.save(self.type + "_latestModel.h5")

    def getReward(self):
        if not self.player.getIsAlive():
            reward = -1 * self.lastMass
            self.lastMass = None
            return reward
        currentMass = self.player.getTotalMass()
        reward = currentMass - self.lastMass
        self.lastMass = currentMass
        return reward

    def getType(self):
        return self.type