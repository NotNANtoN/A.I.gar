import pygame
import numpy

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)


class View:

    def __init__(self, model, width, height):
        self.width = width
        self.height = height
        self.screenDims = numpy.array([self.width, self.height])
        self.model = model
        self.model.register_listener(self.model_event)
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('A.I.gar')


    def drawDebugInfo(self, cell, cells, scaledPos, fovPos, fovDims):
        print("pos: (", fovPos[0], ",", fovPos[1], ") fovDims: ", fovDims[0], ", ", fovDims[1])
        if cells == self.model.getPlayerCells():
            pygame.draw.line(self.screen, RED, scaledPos.astype(int),
                             numpy.array(cell.getVelocity()) * 10 +
                             numpy.array(scaledPos.astype(int)))

    def drawCells(self, cells, fovPos, fovDims):
        for cell in cells:
            if cell.isInFov(fovPos, fovDims):
                rad = cell.getRadius()
                pos = numpy.array(cell.getPos())
                scaledRad = self.modelToViewScaleRadius(rad, fovDims)
                scaledPos = self.modelToViewScaling(pos, fovPos, fovDims)
                pygame.draw.circle(self.screen, cell.getColor(), scaledPos.astype(int), scaledRad)
                if self.model.getDebugStatus():
                    self.drawDebugInfo(cell, cells, scaledPos, fovPos, fovDims)

    def drawAllCells(self):
        fovPos = self.model.getFovPos()
        fovDims = self.model.getFovDims()
        print("pos: (", fovPos[0], ",", fovPos[1], ") fovDims: ", fovDims[0], ", ", fovDims[1])

        self.drawCells(self.model.getCollectibles(), fovPos, fovDims)
        self.drawCells(self.model.getViruses(), fovPos, fovDims)
        self.drawCells(self.model.getPlayerCells(), fovPos, fovDims)

    def draw(self):
        self.screen.fill(WHITE)
        self.drawAllCells()
        pygame.display.update()

    def model_event(self):
        if self.model.hasHuman() or self.model.hasSpectator:
            if self.model.getDebugStatus():
                print("Draw some stuff:")
            self.draw()

    def modelToViewScaling(self, pos, fovPos, fovDims):
        adjustedPos = pos - fovPos + (fovDims / 2)
        scaledPos = adjustedPos * (self.screenDims / fovDims)
        return scaledPos

    def viewToModelScaling(self, pos):
        fovPos = self.model.getFovPos()
        fovDims = self.model.getFovDims()
        scaledPos = pos / (self.screenDims / fovDims)
        adjustedPos = scaledPos + fovPos - (fovDims / 2)
        return adjustedPos

    def modelToViewScaleRadius(self, rad, fovDims):
        return int(rad * (self.screenDims[0] / fovDims[0]))

    # Checks:
    def getScreenDims(self):
        return numpy.array([self.width, self.height])
