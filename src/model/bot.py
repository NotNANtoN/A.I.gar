import numpy

class Bot(object):
    """docstring for Bot"""

    def __init__(self, player, field):
        self.player = player
        self.field = field

    def update(self):
        if self.player.getIsAlive():
            # This bot does random stuff
            midPoint = self.player.getFovPos()
            dims = self.player.getFovDims()
            x = int(midPoint[0])
            y = int(midPoint[1])
            width = int(dims[0])
            height = int(dims[1])
            fovPos = self.player.getFovPos()
            fovDims = self.player.getFovDims()
            cellsInFov = self.field.getPelletsInFov(fovPos, fovDims)
            playerCellsInFov = self.field.getEnemyPlayerCellsInFov(self.player)
            firstPlayerCell = self.player.getCells()[0]
            for opponentCell in playerCellsInFov:
                # If the single celled bot can eat the opponent cell add it to list
                if firstPlayerCell.getMass() > 1.25 * opponentCell.getMass():
                    cellsInFov.append(opponentCell)
            if len(cellsInFov) > 0:

                bestCell = max(cellsInFov, key = lambda p: p.getMass() / (p.squaredDistance(firstPlayerCell) if p.squaredDistance(firstPlayerCell) != 0 else 1))
                bestCellPos = bestCell.getPos()
                #closestCollectible = min(collectiblesInFov, key=lambda p: p.squaredDistance(firstPlayerCell))
                #closestCollectiblePos = closestCollectible.getPos()
                xChoice = bestCellPos[0]
                yChoice = bestCellPos[1]
            else:
                xChoice = numpy.random.randint(x - width, x + width)
                yChoice = numpy.random.randint(y - height, y + height)
            randNum = numpy.random.randint(0,1000)
            splitChoice = False
            ejectChoice = False
            if randNum > 995:
                splitChoice = True
            if randNum > 995:
                ejectChoice = True
            #ejectChoice = True if uniform(0, 1) > 0.99 else False
            self.player.setCommands(xChoice, yChoice, splitChoice, ejectChoice)
