import numpy

MOVESPEED = 5
SPLITSPEED = 2  # Speed of just spawned cell


class Cell(object):
    def __init__(self, x, y, radius, color):
        self.radius = radius
        self.x = x
        self.y = y
        self.color = color
        self.vx = 0
        self.vy = 0

    def setMoveDirection(self, commandPoint):
        difference = numpy.subtract(commandPoint, [self.x, self.y])
        ratio = difference[1] / difference[0]
        angle = numpy.arctan2(difference[1] , difference[0])
        self.vx = MOVESPEED * numpy.cos(angle)
        self.vy = MOVESPEED * numpy.sin(angle)

    def split(self):
        pass

    def eject(self):
        pass

    def updateDirection(self, x, v, maxX):
        return min(maxX, max(0, x + v))

    def updatePos(self, maxX, maxY):
        self.x = self.updateDirection(self.x, self.vx, maxX)
        self.y = self.updateDirection(self.y, self.vy, maxY)

    # Checks:
    def canSplit(self):
        return False

    def canEject(self):
        return False

    # Setters:
    def setPos(self, x, y):
        self.x = x
        self.y = y

    def setRadius(self, val):
        self.radius = val

    # Getters:
    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def getPos(self):
        return [self.x, self.y]

    def getColor(self):
        return self.color

    def getRadius(self):
        return self.radius

    def getVelocity(self):
        return [self.vx, self.vy]
