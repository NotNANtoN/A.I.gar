import pygame
import numpy

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255,0,0)
GREEN = (0,255,0)


class View:

    def __init__(self, model, width, height):
        self.width = width
        self.height = height
        self.screenDims = numpy.array([self.width, self.height])
        self.model = model
        self.model.register_listener(self.model_event)
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('A.I.gar')

    def modelToViewScaling(self, pos):
        fovPos = numpy.array(self.model.human.getFovPos())
        fovDims = numpy.array(self.model.human.getFovDims())
        adjustedPos = pos - fovPos + (fovDims / 2)
        scaledPos = adjustedPos * (self.screenDims / fovDims)
        return scaledPos

    def viewToModelScaling(self, pos):
        fovPos = numpy.array(self.model.human.getFovPos())
        fovDims = numpy.array(self.model.human.getFovDims())
        scaledPos = pos / (self.screenDims / fovDims)
        adjustedPos = scaledPos + fovPos - (fovDims / 2)
        return adjustedPos


    def drawCells(self, cells):

        fovPos = numpy.array(self.model.getFovPos())
        fovDims = numpy.array(self.model.getFovDims())
        for cell in cells:
            if self.isInFov(cell, fovPos, fovDims):
                roundedRad = int(cell.getRadius())
                pos = numpy.array(cell.getPos())
                adjustedPos = pos - fovPos + (fovDims / 2)
                scaledPos = adjustedPos * ( self.screenDims / fovDims)
                pygame.draw.circle(self.screen, cell.getColor(), scaledPos.astype(int), roundedRad)
                if(self.model.getDebugStatus()):
                    print("One cell in the fov! :)")
                    print("pos: (", pos[0], ",", pos[1], ") radius: ", roundedRad)
                    if(cells == self.model.getPlayerCells()):
                        pygame.draw.line(self.screen, RED, scaledPos.astype(int), \
                            numpy.array(cell.getVelocity())*10+\
                            numpy.array(scaledPos.astype(int)))


    def drawAllCells(self):
        self.drawCells(self.model.getCollectibles())
        self.drawCells(self.model.getViruses())
        self.drawCells(self.model.getPlayerCells())

    def draw(self):
        self.screen.fill(WHITE)
        self.drawAllCells()
        pygame.display.update()

    def model_event(self):
        if(self.model.getDebugStatus()):
            print("Draw some stuff:")
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
        if x + radius < xMin or x - radius > xMax or y + radius < yMin or y - radius > yMax:
            return False
        return True

    def getScreenDims(self):
        return numpy.array([self.width, self.height])

