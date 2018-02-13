import numpy
from .parameters import *


class Cell(object):
    _cellId = 0

    @property
    def cellId(self):
        return type(self)._cellId

    @cellId.setter
    def cellId(self, val):
        type(self)._cellId = val

    def __repr__(self):
        return self.name + " id: " + str(self.id) + " -M:" + str(int(self.mass)) + " Pos:" + str(int(self.x)) + "," + str(int(self.y))

    def __init__(self, x, y, mass, player):
        if player != None:
            self.id = self.cellId
            self.cellId += 1
        else:
            self.id = -1
        self.player = player
        self.mass = None
        self.radius = None
        self.setMass(mass)
        self.x = x
        self.y = y
        if self.player == None:
            self.color = (numpy.random.randint(0, 255), numpy.random.randint(0, 255), numpy.random.randint(0, 255))
            self.name = "Pellet"
        else:
            self.name = player.getName()
            self.color = self.player.getColor()
        self.vx = 0
        self.vy = 0
        self.momentum = 1
        self.mergeTime = 0
        self.alive = True

    def setMoveDirection(self, commandPoint):
        difference = numpy.subtract(commandPoint, [self.x, self.y])
        # If cursor is within cell, reduce speed based on distance from cell center (as a percentage)
        hypotenuseSquared = numpy.sum(numpy.power(difference, 2))
        radiusSquared = numpy.power(self.radius, 2)
        speedModifier = min(hypotenuseSquared, radiusSquared) / radiusSquared
        # Check polar coordinate of cursor from cell center
        angle = numpy.arctan2(difference[1], difference[0])
        self.vx = (self.getReducedSpeed() * speedModifier) * numpy.cos(angle)
        self.vy = (self.getReducedSpeed() * speedModifier) * numpy.sin(angle)

    def split(self, commandPoint):
        pass

    def eject(self, commandPoint):
        pass

    def addMomentum(self, value):
        self.momentum = value

    # Increases the mass of the cell by value and updates the radius accordingly
    def grow(self, foodMass):
        newMass = self.mass + foodMass
        self.setMass(newMass)

    def decayMass(self):
        newMass = self.mass * CELL_MASS_DECAY_RATE
        self.setMass(newMass)

    def updateMomentum(self):
        if self.momentum > 1:
            self.momentum = self.momentum * 0.90 - 0.1
        else:
            self.momentum = 1

    def updateMerge(self):
        if self.mergeTime > 0:
            self.mergeTime -= 1

    def updateDirection(self, x, v, maxX):
        return min(maxX, max(0, x + v * self.momentum))

    def updatePos(self, maxX, maxY):
        self.x = self.updateDirection(self.x, self.vx, maxX)
        self.y = self.updateDirection(self.y, self.vy, maxY)

    def overlap(self, cell):
        if self.getMass() > cell.getMass():
            biggerCell = self
            smallerCell = cell
        else:
            biggerCell = cell
            smallerCell = self
        if biggerCell.squaredDistance(smallerCell) < biggerCell.getSquaredRadius():
            return True
        return False

    def resetMergeTime(self):
        self.mergeTime = (BASE_MERGE_TIME + self.mass * 0.0233) * FPS / 2 / GAME_SPEED



    # Returns the squared distance from the self cell to another cell
    def squaredDistance(self, cell):
        difference = self.getPos() - cell.getPos()
        squared = numpy.power(difference, 2)
        return squared[0] + squared[1]

    # Checks:
    def canEat(self, cell):
        return self.mass > 1.25 * cell.getMass()

    def isAlive(self):
        return self.alive == True

    def isInFov(self, fovPos, fovDims):
        xMin = fovPos[0] - fovDims[0] / 2
        xMax = fovPos[0] + fovDims[0] / 2
        yMin = fovPos[1] - fovDims[1] / 2
        yMax = fovPos[1] + fovDims[1] / 2
        x = self.x
        y = self.y
        radius = self.radius
        if x + radius < xMin or x - radius > xMax or y + radius < yMin or y - radius > yMax:
            return False
        return True

    def justEjected(self):
        return self.momentum > 1

    def canSplit(self):
        return self.mass > 36

    def canEject(self):
        return self.mass > 35

    def canMerge(self):
        return self.mergeTime <= 0

    # Setters:
    def setAlive(self, val):
        self.alive = val

    def setPos(self, x, y):
        self.x = x
        self.y = y

    #m = ((r - 4) / 6)²
    #r = sqrt(m) * 6 + 4
    def setRadius(self, val):
        self.radius = val
        #self.mass = numpy.power((self.radius - 4) * 6, 2)
        self.mass = numpy.power(self.radius, 2) * numpy.pi

    def setMass(self, val):
        self.mass = val
        self.radius = numpy.sqrt(self.mass / numpy.pi)
        #self.radius = numpy.sqrt(self.mass) * 6 + 4

    # Getters:
    def getPlayer(self):
        return self.player

    def getName(self):
        return self.name

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def getPos(self):
        return numpy.array([self.x, self.y])

    def getColor(self):
        return self.color

    def getRadius(self):
        return self.radius

    def getMass(self):
        return self.mass

    def getSquaredRadius(self):
        return numpy.power(self.radius, 2)

    def getReducedSpeed(self):
        #return CELL_MOVE_SPEED * numpy.power(self.mass, -0.439)
        return CELL_MOVE_SPEED * numpy.power(self.mass, -0.2)

    def getVelocity(self):
        return [self.vx, self.vy]
