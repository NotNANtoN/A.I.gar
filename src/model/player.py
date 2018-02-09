from random import randint
import numpy


class Player(object):
    """docstring for Player"""

    def __init__(self, name):
        self.color = (randint(0, 255), randint(0, 255), randint(0, 255))
        self.name = name
        self.cells = []
        self.canSplit = False
        self.canEject = False
        self.isAlive = True
        # Commands:
        self.commandPoint = [-1, -1]
        self.split = False
        self.eject = False

    def update(self, fieldWidth, fieldHeight):
        if self.isAlive:
            self.decayMass()
            self.updateCellsMoveDir()
            self.updateCellsSplit()
            self.updateCellsEject()
            self.updateCellsMovement(fieldWidth, fieldHeight)

    def decayMass(self):
        for cell in self.cells:
            cell.decayMass()

    def updateCellsMoveDir(self):
        for cell in self.cells:
            cell.setMoveDirection(self.commandPoint)

    def updateCellsSplit(self):
        if not self.split:
            return
        for cell in self.cells:
            if cell.canSplit():
                cell.split()

    def updateCellsEject(self):
        if not self.eject:
            return
        for cell in self.cells:
            if cell.canEject():
                cell.eject()

    def updateCellsMovement(self, fieldWidth, fieldHeight):
        for cell in self.cells:
            cell.updatePos(fieldWidth, fieldHeight)

    def split(self):
        for cell in self.cells:
            if cell.canSplit():
                cell.split()

    def eject(self):
        for cell in self.cells:
            if cell.canEject():
                cell.eject()

    # Setters:
    def setMoveTowards(self, relativeMousePos):
        self.commandPoint = relativeMousePos

    def addCell(self, cell):
        self.cells.append(cell)

    def removeCell(self, cell):
        self.cells.remove(cell)

    def setCommands(self, x, y, split, eject):
        self.commandPoint = [x, y]
        self.split = split
        self.eject = eject

    def setSplit(self, val):
        self.split = val

    def setDead(self):
        self.isAlive = False

    def setAlive(self):
        self.isAlive = True

    # Checks:

    # Getters:
    def getCells(self):
        return self.cells

    def getCanSplit(self):
        return False

    def getCanEject(self):
        return False

    def getFovPos(self):
        meanX = sum(cell.getX() for cell in self.cells) / len(self.cells)
        meanY = sum(cell.getY() for cell in self.cells) / len(self.cells)
        return meanX, meanY

    def getFovDims(self):
        width = numpy.power(max(self.cells, key=lambda p: p.getRadius()).getRadius(), 0.6) * 40
        height = width
        return width, height

    def getFov(self):
        fovPos = self.getFovPos()
        fovDims = self.getFovDims()
        return fovPos, fovDims

    def getColor(self):
        return self.color

    def getName(self):
        return self.name

    def getIsAlive(self):
        return self.isAlive
