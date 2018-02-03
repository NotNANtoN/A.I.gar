from .cell import Cell
from random import randint


class Player(object):
    STARTRADIUS = 10



    """docstring for Player"""
    def __init__(self, name, x, y, field):
        self.color = (randint(0,255), randint(0,255), randint(0,255))
        self.name = name
        startCell = Cell(x,y, STARTRADIUS, self.color, self)
        self.field = field
        self.cells = [startCell]
        self.canSplit = False
        self.canEject = False
        self.fovCenter = (x,y)
        self.fovSize = (150,120)

        self.commands = (-1, -1, False, False) # x, y, split, ejectMass


    def update(self):
        for cell in self.cells:
            cell.updatePos(field.getWidth(), field.getHeight()) 

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
        self.commands = (x,y,split,eject)

    # Checks:
    def canSplit(self):
        return False

    def canEject(self):
        return False

    # Getters:
    def getFov(self):

        
        xAvg = avg(self.getCells() )

    def getCells(self):
        return self.cells



        