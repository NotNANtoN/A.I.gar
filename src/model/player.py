from random import randint
from .cell import Cell
from .parameters import *
import numpy


class Player(object):
    """docstring for Player"""
    def __repr__(self):
        return self.name

    def __init__(self, name):
        self.color = (randint(0, 255), randint(0, 255), randint(0, 255))
        self.name = name
        self.cells = []
        self.canSplit = False
        self.canEject = False
        self.isAlive = True
        # Commands:
        self.commandPoint = [-1, -1]
        self.doSplit = False
        self.doEject = False

    def update(self, fieldWidth, fieldHeight):
        if self.isAlive:
            self.decayMass()
            self.updateCellProperties()
            self.split()
            self.eject()
            self.updateCellsMovement(fieldWidth, fieldHeight)

    def decayMass(self):
        for cell in self.cells:
            cell.decayMass()

    def updateCellProperties(self):
        for cell in self.cells:
            cell.updateMomentum()
            cell.updateMerge()
            if not cell.justEjected():
                cell.setMoveDirection(self.commandPoint)

    def split(self):
        if not self.doSplit:
            return
        self.cells.sort(key=lambda p: p.getMass(), reverse=True)
        newCells = []
        for cell in self.cells:
            if cell.canSplit() and len(self.cells) + len(newCells) < 16:
                cellPos = cell.getPos()
                newCell = Cell(cellPos[0], cellPos[1], cell.getMass() / 2,  self)
                newCell.setMoveDirection(self.commandPoint)
                newCell.addMomentum(MOMENTUM_BASE + MOMENTUM_PROPORTION_TO_MASS * cell.getRadius())
                newCell.resetMergeTime(1)
                cell.setMass(cell.getMass() / 2)
                cell.resetMergeTime(1)
                newCells.append(newCell)
        for newCell in newCells:
            self.addCell(newCell)

    def eject(self):
        if not self.doEject:
            return
        for cell in self.cells:
            if cell.canEject():
                cell.eject(self.commandPoint)

    def updateCellsMovement(self, fieldWidth, fieldHeight):
        for cell in self.cells:
            cell.updatePos(fieldWidth, fieldHeight)

    # Setters:
    def setMoveTowards(self, relativeMousePos):
        self.commandPoint = relativeMousePos

    def addCell(self, cell):
        self.cells.append(cell)

    def addMass(self, value):
        for cell in self.cells:
            mass = cell.getMass()
            cell.setMass(mass + value)

    def removeCell(self, cell):
        cell.setAlive(False)
        self.cells.remove(cell)

    def setCommands(self, x, y, split, eject):
        self.commandPoint = [x, y]
        self.doSplit = split
        self.doEject = eject

    def setSplit(self, val):
        self.doSplit = val

    def setEject(self, val):
        self.doEject = val

    def setDead(self):
        self.isAlive = False

    def setAlive(self):
        self.isAlive = True

    # Checks:

    # Getters:
    def getTotalMass(self):
        return sum(cell.getMass() for cell in self.cells)

    def getCells(self):
        return self.cells

    def getMergableCells(self):
        cells = []
        for cell in self.cells:
            if cell.canMerge():
                cells.append(cell)
        return cells

    def getCanSplit(self):
        if len(self.cells) >= 16:
            return False
        for cell in self.cells:
            if cell.canSplit():
                return True
        return False

    def getCanEject(self):
        for cell in self.cells:
            if cell.canEject():
                return True
        return False

    def getFovPos(self):
        meanX = sum(cell.getX() for cell in self.cells) / len(self.cells)
        meanY = sum(cell.getY() for cell in self.cells) / len(self.cells)
        return meanX, meanY

    def getFovDims(self):
        biggestCellRadius = max(self.cells, key=lambda p: p.getRadius()).getRadius()
        width = numpy.power(biggestCellRadius, 0.4) * 40 * numpy.power(len(self.cells), 1 / 6)
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
