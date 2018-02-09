import numpy
from .parameters import *


class Cell(object):
    def __init__(self, x, y, mass, color):
        self.mass = None
        self.radius = None
        self.setMass(mass)
        self.x = x
        self.y = y
        self.color = color
        self.vx = 0
        self.vy = 0

    def setMoveDirection(self, commandPoint):
        difference = numpy.subtract(commandPoint, [self.x, self.y])
        #If cursor is within cell, reduce speed based on distance from cell center (as a percentage)
        hypotenuseSquared = numpy.sum(numpy.power(difference, 2))
        radiusSquared = numpy.power(self.radius,2)
        speedModifier = min(hypotenuseSquared, radiusSquared) / radiusSquared
        #Check polar coordinate of cursor from cell center
        angle = numpy.arctan2(difference[1] , difference[0])
        self.vx = (self.getReducedSpeed() * speedModifier) * numpy.cos(angle)
        self.vy = (self.getReducedSpeed() * speedModifier) * numpy.sin(angle)

    def split(self):
        pass

    def eject(self):
        pass

    # Increases the mass of the cell by value and updates the radius accordingly
    def grow(self, foodMass):
        newMass = self.mass + foodMass
        self.setMass(newMass)

    def decayMass(self):
        newMass = self.mass * CELL_MASS_DECAY_RATE
        self.setMass(newMass)

    def updateDirection(self, x, v, maxX):
        return min(maxX, max(0, x + v))

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

    # Returns the squared distance from the self cell to another cell
    def squaredDistance(self, cell):
        difference = self.getPos() - cell.getPos()
        squared = numpy.power(difference, 2)
        return squared[0] + squared[1]


    #############################################
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
        self.mass = numpy.power(self.radius, 2) * numpy.pi

    def setMass(self, val):
        self.mass = val
        self.radius = numpy.sqrt(self.mass / numpy.pi)

    # Getters:
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
        return CELL_MOVE_SPEED * numpy.power(self.getMass(), -1.0 / 9 )

    def getVelocity(self):
        return [self.vx, self.vy]

