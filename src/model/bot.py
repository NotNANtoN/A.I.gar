from random import randint, uniform


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

            cellsInFov = self.field.getCollectiblesInFov(self.player)
            playerCellsInFov = self.field.getPlayerCellsInFov(self.player)
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
                xChoice = randint(x - width, x + width)
                yChoice = randint(y - height, y + height)
            splitChoice = True if uniform(0, 1) > 0.95 else False
            ejectChoice = True if uniform(0, 1) > 0.99 else False
            self.player.setCommands(xChoice, yChoice, splitChoice, ejectChoice)
