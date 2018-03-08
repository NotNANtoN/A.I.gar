import numpy
import math
import keras
from .parameters import *
from keras.models import Sequential
from keras.layers import Dense, Activation


def squareDist(self, pos1, pos2):
    return (pos1[0] - pos2[0]) * (pos1[0] - pos2[0]) + (pos1[1] - pos2[1]) * (pos1[1] - pos2[1])


class Bot(object):
    """docstring for Bot"""
    # Create all possible discrete actions
    possibleLimitedActions = [[x, y, split, eject] for x in [0, 0.5, 1] for y in [0, 0.5, 1] for split in [0, 1] for
                              eject in [0, 1]]
    # Filter out actions that do a split and eject at the same time
    for action in possibleLimitedActions[:]:
        if action[2] and action[3]:
            possibleLimitedActions.remove(action)

    stateReprLen = 8
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
                              optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.0, nesterov=True))

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

    def __init__(self, player, field, type):
        self.policyNetwork = None
        self.type = type
        self.player = player
        self.field = field
        self.oldState = None

        if self.type == "NN":
            self.lastMass = START_MASS
            self.reward = 0
            self.lastAction = [0, 0, 0, 0]
            self.discount = 0.99
            self.epsilon = 0.9
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
        for info in cellInfos:
            totalInfo += info
        if closestEnemyCell == None:
            totalInfo += [0, 0, 0]
        else:
            totalInfo += self.isRelativeCellData(closestEnemyCell, left, top, size, maximumCellMass)
        totalInfo += closestPelletPos
        return totalInfo


    def qLearn(self, newState, reward):
        actions = self.possibleLimitedActions
        oldStateNumpy = numpy.array([self.lastAction + self.oldState])
        qValueOld = self.valueNetwork.predict(oldStateNumpy)
        # TODO can we merge newAction and self.lastAction into one? Is this then TD-Learning?
        # This is the policy: take the actions that yield the highest Q value
        newAction = max(actions, key=lambda p: self.valueNetwork.predict(numpy.array([p + newState])))
        qValueNew = self.valueNetwork.predict(numpy.array([newAction + newState]))

        if __debug__:
            print("State: ", end=" ")
            for info in newState:
                print(round(info, 2), end=" ")
            print(" ")
            print("Action: ", newAction)
        if round(reward, 2) > 0 or round(reward, 2) < 0:
            print("reward: ", round(reward, 2))

        print("qValueNew: ", round(qValueNew[0][0], 2))
        print(" ")
        if math.isnan(qValueNew):
            print("ERROR: qValueNew is nan!")
            quit()
        # If the player died, the target is the reward
        if self.player in self.field.getDeadPlayers():
            target = numpy.array([reward])
        else:
            #target = reward + self.discount * qValueNew - qValueOld
            target = reward + self.discount * qValueNew
        self.valueNetwork.fit(numpy.array([self.oldState + self.lastAction]), target, verbose=0)

        if numpy.random.random(1) > self.epsilon:
            if __debug__:
                print("Exploration!")
            self.lastAction = actions[numpy.random.randint(len(actions))]
        else:
            qValues = [self.valueNetwork.predict(numpy.array([action + newState])) for action in actions]
            if __debug__:
                print("qValues:")
                for value in qValues:
                    print(value[0][0], end = " ")
                print("")
            maxIndex = numpy.argmax(qValues)
            self.lastAction = actions[maxIndex]
        if __debug__:
          print("action:")
          print(self.lastAction)
        self.oldState = newState


    def update(self):
        if self.player.getIsAlive():
            midPoint = self.player.getFovPos()
            size = self.player.getFovSize()
            x = int(midPoint[0])
            y = int(midPoint[1])
            left = x - int(size / 2)
            top = y - int(size / 2)
            size = int(size)
            cellsInFov = self.field.getPelletsInFov(midPoint, size)
            if self.oldState == None:
                self.oldState = self.getStateRepresentation()

            if self.type == "NN":
                # Get current State, Reward and the old State
                newState = self.getStateRepresentation()
                reward = self.getReward()
                self.qLearn(newState, reward)

                xChoice = left + self.lastAction[0] * size
                yChoice = top + self.lastAction[1] * size
                splitChoice = True if self.lastAction[2] > 0.5 else False
                ejectChoice = True if self.lastAction[3] > 0.5 else False
                if __debug__:
                    print("xChoice: ", round(xChoice, 2), " yChoice: ", round(yChoice,2) , " Split: ", splitChoice, " Eject: ", ejectChoice)
                    print(" ")

            elif self.type == "Greedy":
                playerCellsInFov = self.field.getEnemyPlayerCellsInFov(self.player)
                firstPlayerCell = self.player.getCells()[0]
                for opponentCell in playerCellsInFov:
                    # If the single celled bot can eat the opponent cell add it to list
                    if firstPlayerCell.getMass() > 1.25 * opponentCell.getMass():
                        cellsInFov.append(opponentCell)
                if cellsInFov:
                    bestCell = max(cellsInFov, key = lambda p: p.getMass() / (p.squaredDistance(firstPlayerCell) if p.squaredDistance(firstPlayerCell) != 0 else 1))
                    bestCellPos = bestCell.getPos()
                    xChoice = bestCellPos[0]
                    yChoice = bestCellPos[1]
                else:
                    size = int(size / 2)
                    xChoice = numpy.random.randint(x - size, x + size)
                    yChoice = numpy.random.randint(y - size, y + size)
                randNumSplit = numpy.random.randint(0,10000)
                randNumEject = numpy.random.randint(0,10000)
                splitChoice = False
                ejectChoice = False
                if randNumSplit > self.splitLikelihood:
                    splitChoice = True
                if randNumEject > self.ejectLikelihood:
                    ejectChoice = True

            self.player.setCommands(xChoice, yChoice, splitChoice, ejectChoice)

    def saveModel(self):
        self.valueNetwork.save(self.type + "_latestModel.h5")

    def getReward(self):
        if self.player in self.field.getDeadPlayers():
            return -1 * self.lastMass
        currentMass = self.player.getTotalMass()
        reward = currentMass - self.lastMass
        self.lastMass = currentMass
        return reward

    def getType(self):
        return self.type