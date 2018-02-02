from .cell import Cell
from random import randint


class Player(object):
    STARTRADIUS = 10



    """docstring for Player"""
    def __init__(self, name, x, y):
        self.color = (randint(0,255), randint(0,255), randint(0,255))
        self.name = name
        startCell = Cell(x,y, STARTRADIUS, self.color)
        self.cells = [startCell]
        self.canSplit = False
        self.canEject = False
        self.fovCenter = (x,y)
        self.fovSize = (150,120)

        self.commands = (-1, -1, 0, 0) # x, y, split, ejectMass


    def update(self):
        for cell in self.cells:
            cell.updatePos(10000, 1000) # CHANGE THIS!!!!

    def canSplit(self):
        return False

    def canEject(self):
        return False

    def split(self):
        for cell in self.cells:
            if( cell.canSplit() ):
                cell.split()

    def eject(self):
        for cell in self.cells:
            if( cell.canEject() ):
                cell.eject()

    def addCell(self, cell):
        self.cells.append(cell)

    def removeCell(self, cell):
        self.cells.remove(cell)

        