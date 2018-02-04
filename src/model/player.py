from .cell import Cell
from random import randint


class Player(object):
    STARTRADIUS = 10



    """docstring for Player"""
    def __init__(self, name, x, y):
        self.color = (randint(0,255), randint(0,255), randint(0,255))
        self.name = name
        startCell = Cell(x,y, STARTRADIUS, self.color)
        self.field = field
        self.cells = [startCell]
        self.canSplit = False
        self.canEject = False
        self.fovCenter = (x,y)
        self.fovSize = (150,120)
         # Commands:
        self.moveCellsTowards = [-1, -1]
        self.split = False
        self.eject = False

    def update(self, fieldWidth, fieldHeight, moveCellTowards, split, eject):
        for cell in self.cells:
            cell.setMoveDirection(moveCellTowards)
            if( split and cell.canSplit() ):
                cell.split()
            elif( eject and cell.canEject() ):
                cell.eject()
            cell.updatePos(fieldWidth, fieldHeight) 

    def split(self):
        for cell in self.cells:
            if( cell.canSplit() ):
                cell.split()

    def eject(self):
        for cell in self.cells:
            if( cell.canEject() ):
                cell.eject()

    # Setters:
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
    def canSplit(self):
        return False

    def canEject(self):
        return False

    # Getters:
    def getFovPos():
        meanX = sum(cell.getX() for cell in self.cells) / len(self.cells)
        meanY = sum(cell.getY() for cell in self.cells) / len(self.cells)
        return (meanX, meanY)

    def getFovdims():
        width = self.radius * 5
        height = width
        return (width, height)

    def getFov(self):
        fovPos = self.getFovPos()
        fovDims = self.getFovDims()
        return (fovPos, fovDims)
        
    def getCells(self):
        return self.cells



        