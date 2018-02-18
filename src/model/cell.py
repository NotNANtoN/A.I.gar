import numpy
import math
from .parameters import *

'''
# For euclidean distance
from distutils.core import setup, Extension
import numpy.distutils.misc_util

c_ext = Extension("_euclidean", ["_euclidean.c", "euclidean.c"])
setup(
            ext_modules=[c_ext],
            include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
        )

'''

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
        self.player = player
        self.mass = None
        self.radius = None
        self.setMass(mass)
        self.x = x
        self.y = y
        self.pos = numpy.array([x,y])
        if self.player == None:
            self.name = ""
            self.color = (numpy.random.randint(0, 255), numpy.random.randint(0, 255), numpy.random.randint(0, 255))
            self.id = -1
        else:
            self.name = player.getName()
            self.color = self.player.getColor()
            self.id = self.cellId
            self.cellId += 1
        self.velocity = numpy.array([0, 0])
        self.splitVelocity = numpy.array([0, 0])
        self.splitVelocityCounter = 0
        self.splitVelocityCounterMax = 15
        self.mergeTime = 0
        self.blobToBeEjected = None
        self.ejecterCell = None # Used in case of blobs to determine which player ejected this blob
        self.alive = True

    def setMoveDirection(self, commandPoint):
        difference = numpy.subtract(commandPoint, self.getPos())
        # If cursor is within cell, reduce speed based on distance from cell center (as a percentage)
        hypotenuseSquared = numpy.sum(numpy.power(difference, 2))
        radiusSquared = numpy.power(self.radius, 2)
        speedModifier = min(hypotenuseSquared, radiusSquared) / radiusSquared
        # Check polar coordinate of cursor from cell center
        angle = self.calculateAngle(commandPoint)
        self.velocity = self.getReducedSpeed() * speedModifier * numpy.array([numpy.cos(angle), numpy.sin(angle)])


    def calculateAngle(self, point):
        difference = numpy.subtract(point, self.getPos())
        return numpy.arctan2(difference[1], difference[0])

    def split(self, commandPoint, fieldWidth, fieldHeight):
        cellPos = self.getPos()
        newCell = Cell(cellPos[0], cellPos[1], self.mass / 2, self.player)
        angle = newCell.calculateAngle(commandPoint)

        xPoint = numpy.cos(angle) * newCell.getRadius() * 4.5 + cellPos[0]
        yPoint = numpy.sin(angle) * newCell.getRadius() * 4.5 + cellPos[1]
        movePoint = (xPoint, yPoint)
        #newCell.setMoveDirection(movePoint)
        newCell.addMomentum(movePoint, fieldWidth, fieldHeight, self)
        newCell.resetMergeTime(1)
        self.setMass(self.mass / 2)
        #self.resetMergeTime(1)
        return newCell

    def prepareEject(self):
        self.blobToBeEjected = True

    def eject(self, commandPoint):
        #blobSpawnPos can be None if commandPoint is in center of cell, in which case nothing is ejected
        self.mass -= EJECTEDBLOB_BASE_MASS
        self.blobToBeEjected = False
        return self.getPos()

    def addMomentum(self, commandPoint, fieldWidth, fieldHeight, originalCell):
        checkedX = max(0, min(fieldWidth, commandPoint[0]))
        checkedY = max(0, min(fieldHeight, commandPoint[1]))
        checkedPoint = (checkedX, checkedY)
        angle = self.calculateAngle(checkedPoint)
        speed = 2 + originalCell.getRadius() * 0.05
        self.splitVelocity = numpy.array([numpy.cos(angle), numpy.sin(angle)]) * speed
        self.splitVelocityCounter = self.splitVelocityCounterMax

    def updateMomentum(self):
        if self.splitVelocityCounter == -1:
            return
        elif self.splitVelocityCounter > 0:
            self.splitVelocityCounter -= 1
            counterRatio = self.splitVelocityCounter / self.splitVelocityCounterMax
            if counterRatio < 0.1:
                self.splitVelocity *= (1 - counterRatio)
        else:
            self.splitVelocity = numpy.array([0,0])
            self.splitVelocityCounter = -1

    # Increases the mass of the cell by value and updates the radius accordingly
    def grow(self, foodMass):
        newMass = min(MAX_MASS_SINGLE_CELL, self.mass + foodMass)
        self.setMass(newMass)

    def decayMass(self):
        newMass = self.mass * CELL_MASS_DECAY_RATE
        self.setMass(newMass)

    def updateMerge(self):
        if self.mergeTime > 0:
            self.mergeTime -= 1

    def updateDirection(self, x, v, maxX):
        return min(maxX, max(0, x + v))

    def updatePos(self, maxX, maxY):
        combinedVelocity = self.velocity + self.splitVelocity
        self.x = self.updateDirection(self.x, combinedVelocity[0], maxX)
        self.y = self.updateDirection(self.y, combinedVelocity[1], maxY)
        self.pos = numpy.array([self.x,self.y])
        if self.splitVelocityCounter and self.x == maxX or self.x == 0:
            self.splitVelocity[0] *= -1
        if self.splitVelocityCounter and self.y == maxY or self.y == 0:
            self.splitVelocity[1] *= -1

    def overlap(self, cell):
        if self.getMass() > cell.getMass():
            biggerCell = self
            smallerCell = cell
        else:
            biggerCell = cell
            smallerCell = self
        if biggerCell.squaredDistance(smallerCell) * 1.1 < biggerCell.getRadius() * biggerCell.getRadius():
            return True
        return False

    def resetMergeTime(self, factor):
        self.mergeTime = factor * (BASE_MERGE_TIME + self.mass * 0.0233) * FPS / 2 / GAME_SPEED

    # Returns the squared distance from the self cell to another cell
    def squaredDistance(self, cell):
        pos2 = cell.getPos()
        return (self.x - pos2[0]) * (self.x - pos2[0]) + (self.y - pos2[1]) * (self.y - pos2[1])

    def squareDist(self, pos1, pos2):
        return (pos1[0] - pos2[0]) * (pos1[0] - pos2[0])  + (pos1[1] - pos2[1]) *  (pos1[1] - pos2[1])

    # Checks:
    def canEat(self, cell):
        return self.mass > 1.25 * cell.getMass()

    def isAlive(self):
        return self.alive == True

    def isInFov(self, fovPos, fovSize):
        halvedFovDims = fovSize / 2
        xMin = fovPos[0] - halvedFovDims
        xMax = fovPos[0] + halvedFovDims
        yMin = fovPos[1] - halvedFovDims
        yMax = fovPos[1] + halvedFovDims
        if self.x + self.radius < xMin or self.x - self.radius > xMax or self.y + self.radius < yMin or self.y - self.radius > yMax:
            return False
        return True

    def justEjected(self):
        return self.splitVelocityCounter > 0

    def canSplit(self):
        return self.mass > 36

    def canEject(self):
        return self.mass >= 35

    def canMerge(self):
        return self.mergeTime <= 0

    # Setters:
    def setColor(self, color):
        self.color = color

    def setName(self, name):
        self.name = name

    def setAlive(self, val):
        self.alive = val

    def setPos(self, pos):
        self.pos = pos
        self.x = pos[0]
        self.y = pos[1]

    def setRadius(self, val):
        self.radius = val
        #self.mass = numpy.power((self.radius - 4) * 6, 2)
        self.mass = self.radius * self.radius * numpy.pi

    def setMass(self, val):
        self.mass = val
        self.radius = math.sqrt(self.mass / numpy.pi)
        #self.radius = numpy.sqrt(self.mass) * 6 + 4

    def setBlobToBeEjected(self, val):
        self.blobToBeEjected = False

    def setEjecterCell(self, cell):
        self.ejecterCell = cell
        self.color = cell.getColor()

    # Getters:
    def getMergeTime(self):
        return self.mergeTime

    def getSplitVelocityCounter(self):
        return self.splitVelocityCounter

    def getPlayer(self):
        return self.player

    def getName(self):
        return self.name

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def getPos(self):
        return self.pos

    def getColor(self):
        return self.color

    def getRadius(self):
        return self.radius

    def getMass(self):
        return self.mass

    def getReducedSpeed(self):
        #return CELL_MOVE_SPEED * math.pow(self.mass, -0.439)
        return CELL_MOVE_SPEED * math.pow(self.mass, -0.35)

    def getVelocity(self):
        return self.velocity + self.splitVelocity

    def getSplitVelocity(self):
        return self.splitVelocity

    def getClosestSurfacePoint(self, commandPoint):
        # TODO Change this method so that ejection also works properly if the mouse is inside of a cell that ejects
        difference = numpy.subtract(commandPoint, self.getPos())
        # Make sure commandPoint != center of cell since ratio is then a division by 0
        if difference[0] == 0 and difference[1] == 1:
            randomPointInCell = numpy.array(commandPoint) + 1
            #randomPointInCell = (numpy.random.randint(self.getPos()[0] - self.radius, self.getPos()[0] + self.radius), numpy.random.randint(self.getPos()[1] - self.radius, self.getPos()[1] + self.radius))
            return self.getClosestSurfacePoint(randomPointInCell)
        hypotenuseSquared = numpy.sum(numpy.power(difference, 2))
        ratio = numpy.sqrt(hypotenuseSquared / self.radius / self.radius)
        xFromCenter = difference[0] / ratio
        yFromCenter = difference[1] / ratio
        posFromCenter = numpy.array([xFromCenter, yFromCenter])
        surfacePoint = self.getPos() + posFromCenter
        return surfacePoint

    def getBlobToBeEjected(self):
        return self.blobToBeEjected

    def getEjecterCell(self):
        return self.ejecterCell
