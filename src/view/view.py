import pygame 
import os

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

class View():


    def __init__(self, sizeX, sizeY, model):
        self.model = model
        model.register_listener(self.model_event)
        self.screen = pygame.display.set_mode((640,480))
        pygame.display.set_caption('A.I.gar')
        
    def drawCells(self, cells):
        fovPos = self.model.getFovPos()
        fovDims = self.model.getFovDims()
        for cell in cells:
            if( self.isInFov(cell, fovPos, fovDims) ):
                print("One cell in the fov! :)")
                pos = cell.getPos()
                roundedPos = [int(pos[0]), int(pos[1])]
                roundedRad = int(cell.getRadius())
                print("pos: ", pos[0], "-", pos[1], " raidus: ", roundedRad)
                pygame.draw.circle(self.screen, cell.getColor(), roundedPos, roundedRad)


    def drawAllCells(self):
        self.drawCells(self.model.getCollectibles())
        self.drawCells(self.model.getViruses())
        self.drawCells(self.model.getPlayerCells())

    def draw(self):
        self.screen.fill(WHITE)
        self.drawAllCells()
        pygame.display.update()

    def model_event(self, event_name):
        print("Draw some stuff")
        self.draw()



    # Checks:
    def isInFov(self, cell, fovPos, fovDims):
        xMin = fovPos[0] - fovDims[0] / 2
        xMax = fovPos[0] + fovDims[0] / 2
        yMin = fovPos[1] - fovDims[1] / 2
        yMax = fovPos[1] + fovDims[1] / 2
        x = cell.getX()
        y = cell.getY()
        radius = cell.getRadius()
        if( x + radius < xMin or x - radius > xMax or y + radius < yMin or y - radius > yMax ):
            return False
        return True
     