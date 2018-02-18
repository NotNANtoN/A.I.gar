import numpy

class Bot(object):
    """docstring for Bot"""

    def __init__(self, player, field):
        self.player = player
        self.field = field
        self.splitLikelihood = numpy.random.randint(9950,10000)
        self.ejectLikelihood = numpy.random.randint(9990,10000)


    def update(self):
        if self.player.getIsAlive():
            # This bot does random stuff
            midPoint = self.player.getFovPos()
            size = self.player.getFovSize()
            x = int(midPoint[0])
            y = int(midPoint[1])
            width = int(size)
            height = int(size)
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
                #closestCollectible = min(collectiblesInFov, key=lambda p: p.squaredDistance(firstPlayerCell))
                #closestCollectiblePos = closestCollectible.getPos()
                xChoice = bestCellPos[0]
                yChoice = bestCellPos[1]
            else:
                xChoice = numpy.random.randint(x - width, x + width)
                yChoice = numpy.random.randint(y - height, y + height)
            randNumSplit = numpy.random.randint(0,10000)
            randNumEject = numpy.random.randint(0,10000)
            splitChoice = False
            ejectChoice = False
            if randNumSplit > self.splitLikelihood:
                splitChoice = True
            if randNumEject > self.ejectLikelihood:
                ejectChoice = True
            self.player.setCommands(xChoice, yChoice, splitChoice, ejectChoice)
