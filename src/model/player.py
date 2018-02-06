from .cell import Cell
from random import randint


class Player(object):
    STARTRADIUS = 10

    """docstring for Player"""
    def __init__(self, name):
        self.color = (randint(0,255), randint(0,255), randint(0,255))
        self.name = name
        self.cells = []
        self.canSplit = False
        self.canEject = False
         # Commands:
        self.moveCellsTowards = [-1, -1]
        self.split = False
        self.eject = False

    def update(self, fieldWidth, fieldHeight):
        self.updateCellsMoveDir()
        self.updateCellsSplit()
        self.updateCellsEject()
        self.updateCellsMovement(fieldWidth, fieldHeight)

    def updateCellsMoveDir(self):
        for cell in self.cells:
            cell.setMoveDirection(self.moveCellsTowards)

    def updateCellsSplit(self):
        if self.split == False:
            return
        for cell in self.cells:
            if cell.canSplit():
                cell.split()

    def updateCellsEject(self):
        if self.eject == False:
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
        self.moveCellsTowards = relativeMousePos

    def addCell(self, cell):
        self.cells.append(cell)

    def removeCell(self, cell):
        self.cells.remove(cell)

    def setCommands(self, x, y, split, eject):
        self.moveCellsTowards = [x,y]
        self.split = split
        self.eject = eject

    def setSplit(self, bool):
        self.split = bool

    # Checks:
  

    # Getters:
    def getCanSplit(self):
        return False

    def getCanEject(self):
        return False
        
    def getFovPos(self):
        meanX = sum(cell.getX() for cell in self.cells) / len(self.cells)
        meanY = sum(cell.getY() for cell in self.cells) / len(self.cells)
        return meanX, meanY

    def getFovDims(self):
        width = max(self.cells, key = lambda p: p.getRadius()).getRadius() * 5
        height = width
        return width, height

    def getFov(self):
        fovPos = self.getFovPos()
        fovDims = self.getFovDims()
        return fovPos, fovDims
        
    def getCells(self):
        return self.cells

    def getColor(self):
        return self.color

    def getName(self):
        return self.name



        