import numpy
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import time

class Bot(object):
    """docstring for Bot"""

    def __init__(self, player, field, type):
        self.model = None
        self.type = type
        if self.type == "NN":
            self.model = Sequential()
            self.model.add(Dense(100, input_dim = 52, activation = 'relu'))
            self.model.add(Dense(4, activation = 'sigmoid'))
            self.model.compile(loss=keras.losses.categorical_crossentropy,
                               optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))

        print(self.model)
        self.player = player
        self.field = field
        self.splitLikelihood = numpy.random.randint(9950,10000)
        self.ejectLikelihood = numpy.random.randint(9990,10000)

    def returnEssentialCellData(self, cell):
        return [cell.getX(), cell.getY(), cell.getMass()]

    def returnEssentialCellDataRelative(self, cell, left, top, size, totalMass):
        return [(cell.getX() - left) / size, (cell.getY() - top) / size, cell.getMass() / totalMass]


    def update(self):
        if self.player.getIsAlive():
            totalMass = self.player.getTotalMass()
            midPoint = self.player.getFovPos()
            size = self.player.getFovSize()
            x = int(midPoint[0])
            y = int(midPoint[1])
            left = x - size / 2
            top = y - size / 2
            size = int(size)
            cellsInFov = self.field.getPelletsInFov(midPoint, size)

            if self.type == "NN":
                cells = self.player.getCells()
                totalCells = len(cells)

                # TODO Make cell x and y relative to screen, do not give the model the info where it is situated in the field
                # TODO This approach is problematic, as the network needs the data to refer to the same cell continously
                # TODO but if we delete cells from the getCells list during the game the order changes. Order by id maybe?

                cellInfos = [self.returnEssentialCellDataRelative(cells[idx], left, top, size, totalMass) if idx < totalCells else [0,0,0] for idx in range(16)]
                # cellInfoTransform should have length 48: three values for 16 cells
                totalInfo = []
                for info in cellInfos:
                    #totalInfo = totalInfo.insert(info)
                    for val in info:
                        totalInfo.append(val)
                totalInfo.append(size)
                totalInfo.append(totalMass)
                closestPelletPos =  max(cellsInFov, key = lambda p: p.getMass()).getPos() if cellsInFov else [0,0]
                totalInfo.append(closestPelletPos[0])
                totalInfo.append(closestPelletPos[1])
                #totalInfo = numpy.asarray(totalInfo)
                inputNN = numpy.array(totalInfo)
                realInput = numpy.array([inputNN])
                #inputNN = numpy.random.random((1,8))
                #print(inputNN)
                #inputNN = numpy.array([numpy.random.randint(maxVal) for maxVal in range(1,9)])
                outputNN = self.model.predict(realInput)[0]
                if __debug__:
                    print("")
                    print("Network input: ", numpy.round(inputNN[0:2], 2))
                    print("Network output: ", numpy.round(outputNN, 2))
                xChoice = left + outputNN[0] * size
                yChoice = top + outputNN[1] * size
                splitChoice = True if outputNN[2] > 0.5 else False
                ejectChoice = True if outputNN[3] > 0.5 else False
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
