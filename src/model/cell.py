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
            self.name = ""
            self.color = (numpy.random.randint(0, 255), numpy.random.randint(0, 255), numpy.random.randint(0, 255))
        else:
            self.name = player.getName()
            self.color = self.player.getColor()
        self.velocity = numpy.array([0, 0])
        self.splitVelocity = numpy.array([0, 0])
        self.splitVelocityCounter = 0
        self.splitVelocityCounterMax = 15
        self.momentum = 1
        self.mergeTime = 0
        self.blobToBeEjected = None
        self.ejecterPlayer = None # Used in case of blobs to determine which player ejected this blob
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

    def split(self, commandPoint):
        pass

    def prepareEject(self):
        self.blobToBeEjected = True

    def eject(self, commandPoint):
        #blobSpawnPos can be None if commandPoint is in center of cell, in which case nothing is ejected
        blobSpawnPos = self.getClosestSurfacePoint(commandPoint)
        if blobSpawnPos is not None:
            self.mass -= EJECTEDBLOB_BASE_MASS
        self.blobToBeEjected = False
        return blobSpawnPos

    def addMomentum(self, speed, commandPoint, fieldWidth, fieldHeight):
        checkedX = max(0, min(fieldWidth, commandPoint[0]))
        checkedY = max(0, min(fieldHeight, commandPoint[1]))
        checkedPoint = (checkedX, checkedY)
        angle = self.calculateAngle(checkedPoint)
        self.splitVelocity = numpy.array([numpy.cos(angle), numpy.sin(angle)]) * speed
        self.splitVelocityCounter = self.splitVelocityCounterMax

    def updateMomentum(self):
        if self.splitVelocityCounter == -1:
            return
        elif self.splitVelocityCounter > 0:
            self.splitVelocityCounter -= 1
            speedRatio = self.splitVelocityCounter / self.splitVelocityCounterMax
            self.splitVelocity *= speedRatio
        else:
            self.splitVelocity = numpy.array([0,0])
            self.splitVelocityCounter = -1

    # Increases the mass of the cell by value and updates the radius accordingly
    def grow(self, foodMass):
        newMass = self.mass + foodMass
        self.setMass(newMass)

    def decayMass(self):
        newMass = self.mass * CELL_MASS_DECAY_RATE
        self.setMass(newMass)

    def updateMerge(self):
        if self.mergeTime > 0:
            self.mergeTime -= 1

    def updateDirection(self, x, v, maxX):
        return min(maxX, max(0, x + v * self.momentum))

    def updatePos(self, maxX, maxY):
        combinedVelocity = self.velocity + self.splitVelocity
        self.x = self.updateDirection(self.x, combinedVelocity[0], maxX)
        self.y = self.updateDirection(self.y, combinedVelocity[1], maxY)

    def overlap(self, cell):
        if self.getMass() > cell.getMass():
            biggerCell = self
            smallerCell = cell
        else:
            biggerCell = cell
            smallerCell = self
        if biggerCell.squaredDistance(smallerCell) * 1.1 < biggerCell.getSquaredRadius():
            return True
        return False

    def resetMergeTime(self, factor):
        self.mergeTime = factor * (BASE_MERGE_TIME + self.mass * 0.0233) * FPS / 2 / GAME_SPEED

    # Returns the squared distance from the self cell to another cell
    def squaredDistance(self, cell):
        return self.squareDist(self.getPos(), cell.getPos())

    def squareDist(self, pos1, pos2):
        return numpy.sum(numpy.power(pos1-pos2, 2))

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

    def setPos(self, x, y):
        self.x = x
        self.y = y

    def setRadius(self, val):
        self.radius = val
        #self.mass = numpy.power((self.radius - 4) * 6, 2)
        self.mass = numpy.power(self.radius, 2) * numpy.pi

    def setMass(self, val):
        self.mass = val
        self.radius = numpy.sqrt(self.mass / numpy.pi)
        #self.radius = numpy.sqrt(self.mass) * 6 + 4

    def setBlobToBeEjected(self, val):
        self.blobToBeEjected = False

    def setEjecterPlayer(self, player):
        self.ejecterPlayer = player
        self.color = player.getColor()

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
        return CELL_MOVE_SPEED * numpy.power(self.mass, -0.275)

    def getVelocity(self):
        return self.velocity + self.splitVelocity

    def getSplitVelocity(self):
        return self.splitVelocity

    def getClosestSurfacePoint(self, commandPoint):
        # TODO Change this method so that ejection also works properly if the mouse is inside of a cell that ejects
        difference = numpy.subtract(commandPoint, self.getPos())
        # Make sure commandPoint != center of cell since ratio is then a division by 0
        if difference == [0,0]:
            randomPointInCell = numpy.array(commandPoint) + 1
            #randomPointInCell = (numpy.random.randint(self.getPos()[0] - self.radius, self.getPos()[0] + self.radius), numpy.random.randint(self.getPos()[1] - self.radius, self.getPos()[1] + self.radius))
            return self.getClosestSurfacePoint(randomPointInCell)
        hypotenuseSquared = numpy.sum(numpy.power(difference, 2))
        ratio = numpy.sqrt(hypotenuseSquared / self.getSquaredRadius())
        xFromCenter = difference[0] / ratio
        yFromCenter = difference[1] / ratio
        posFromCenter = numpy.array([xFromCenter, yFromCenter])
        surfacePoint = self.getPos() + posFromCenter
        return surfacePoint

    def getBlobToBeEjected(self):
        return self.blobToBeEjected

    def getEjecterPlayer(self):
        return self.ejecterPlayer