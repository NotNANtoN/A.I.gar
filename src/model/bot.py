from random import randint, uniform


class Bot(object):
    """docstring for Bot"""

    def __init__(self, player):
        self.player = player

    def update(self):
        # This bot does random stuff
        midPoint = self.player.getFovPos()
        dims = self.player.getFovDims()
        x = int(midPoint[0])
        y = int(midPoint[1])
        width = int(dims[0])
        height = int(dims[1])
        xChoice = randint(x - width, x + width)
        yChoice = randint(y - width, y + width)
        splitChoice = True if uniform(0, 1) > 0.95 else False
        ejectChoice = True if uniform(0, 1) > 0.99 else False
        self.player.setCommands(xChoice, yChoice, splitChoice, ejectChoice)
