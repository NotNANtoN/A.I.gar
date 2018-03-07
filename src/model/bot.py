import numpy
import math
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation


def squareDist(self, pos1, pos2):
    return (pos1[0] - pos2[0]) * (pos1[0] - pos2[0]) + (pos1[1] - pos2[1]) * (pos1[1] - pos2[1])


class Bot(object):
    """docstring for Bot"""

    def __init__(self, player, field, type):
        self.policyNetwork = None
        self.type = type
        self.player = player
        self.field = field

        self.oldState = None

        if self.type == "NN":
            stateReprLen = 8
            actionLen = 4

            self.policyNetwork = Sequential()
            self.policyNetwork.add(Dense(50, input_dim = stateReprLen, activation ='relu'))
            self.policyNetwork.add(Dense(4, activation ='linear'))
            self.policyNetwork.compile(loss='mean_squared_error',
                                       optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))

            self.valueNetwork = Sequential()
            self.valueNetwork.add(Dense(20, input_dim = stateReprLen + actionLen, activation = 'relu'))
            #self.valueNetwork.add(Dense(10, activation = 'relu'))
            self.valueNetwork.add(Dense(1, activation = 'linear'))
            self.valueNetwork.compile(loss='mean_squared_error',
                                       optimizer=keras.optimizers.SGD(lr=0.0005, momentum = 0.0, nesterov=True))
            self.reward = 0
            self.lastAction = [0, 0, 0, 0]
            self.discount = 0.95
            self.epsilon = 0.95


        #print(self.model)
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

        inputNN = totalInfo
        return inputNN


    def update(self):
        if self.player.getIsAlive():
            totalMass = self.player.getTotalMass()
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
                possibleLimitedActions = [[x, y, split, eject] for x in [0, 0.5, 1] for y in [0, 0.5, 1] for split in [0, 1] for eject in [0, 1]]
                newState = self.getStateRepresentation()

                reward = self.field.getReward(self.player)
                oldStateNumpy = numpy.array([self.lastAction + self.oldState])
                qValueOld = self.valueNetwork.predict(oldStateNumpy)
                #TODO can we merge newAction and self.lastAction into one?
                newAction = max(possibleLimitedActions, key = lambda p: self.valueNetwork.predict(numpy.array([p + newState])))


                qValueNew = self.valueNetwork.predict(numpy.array([newAction + newState]))
                if __debug__:
                    print("State: ", end=" ")
                    for info in newState:
                        print(round(info, 2), end=" ")
                    print(" ")
                    print(("Action: ", newAction))
                    print("qValueNew: ", qValueNew[0][0])
                    print(" ")
                if math.isnan(qValueNew):
                    print("ERROR: qValueNew is nan!")
                    quit()
                target = reward + self.discount * qValueNew - qValueOld
                self.valueNetwork.fit(numpy.array([self.oldState + self.lastAction]), target, verbose = 0)

                if numpy.random.random(1) > self.epsilon:
                    self.lastAction = possibleLimitedActions[numpy.random.randint(len(possibleLimitedActions))]
                else:
                    self.lastAction = max(possibleLimitedActions, key = lambda p: self.valueNetwork.predict(numpy.array([p + newState])))
                self.oldState = newState

                xChoice = left + self.lastAction[0] * size
                yChoice = top + self.lastAction[1] * size
                splitChoice = True if self.lastAction[2] > 0.5 else False
                ejectChoice = True if self.lastAction[3] > 0.5 else False

                # TODO if we delete cells from the getCells list during the game the order changes. Order by id maybe?


                if __debug__:
                    print("xChoice: ", round(xChoice, 2), " yChoice: ", round(yChoice,2) , " Split: ", splitChoice, " Eject: ", ejectChoice)


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
