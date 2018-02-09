from random import randint, uniform


class Bot(object):
    """docstring for Bot"""

    def __init__(self, player, field):
        self.player = player
        self.field = field

    def update(self):
        if (self.player.getIsAlive()):
            # This bot does random stuff
            midPoint = self.player.getFovPos()
            dims = self.player.getFovDims()
            x = int(midPoint[0])
            y = int(midPoint[1])
            width = int(dims[0])
            height = int(dims[1])

            collectiblesInFov = self.field.getCollectiblesInFov(self.player)
            if (len(collectiblesInFov) > 0):
                playerCellsInFov = self.field.getPlayerCellsInFov(self.player)
                firstPlayerCell = self.player.getCells()[0]
                for opponentCell in playerCellsInFov:
                    if firstPlayerCell.getMass() > 1.25 * opponentCell.getMass():
                        collectiblesInFov.append(opponentCell)

                closestCollectible = min(collectiblesInFov, key = lambda p: p.squaredDistance(firstPlayerCell))
                closestCollectiblePos = closestCollectible.getPos()
                xChoice = closestCollectiblePos[0]
                yChoice = closestCollectiblePos[1]
            else:
                xChoice = randint(x - width, x + width)
                yChoice = randint(y - height, y + height)
            splitChoice = True if uniform(0, 1) > 0.95 else False
            ejectChoice = True if uniform(0, 1) > 0.99 else False
            self.player.setCommands(xChoice, yChoice, splitChoice, ejectChoice)
