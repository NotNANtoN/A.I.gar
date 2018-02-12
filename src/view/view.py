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
        pygame.init()
        pygame.display.set_caption('A.I.gar')


    def drawDebugInfo(self):
        cells = self.model.getPlayerCells()
        fovPos = self.model.getFovPos()
        fovDims = self.model.getFovDims()
        for cell in cells:
            pos = numpy.array(cell.getPos())
            scaledPos = self.modelToViewScaling(pos, fovPos, fovDims)
            pygame.draw.line(self.screen, RED, scaledPos.astype(int),
                             numpy.array(cell.getVelocity()) * 10 +
                             numpy.array(scaledPos.astype(int)))
        if self.model.hasHuman():
            for cell in self.model.field.pelletHashtable.getNearbyObjects(self.model.getHuman().cells[0]):
                rad = cell.getRadius()
                pos = numpy.array(cell.getPos())
                scaledRad = self.modelToViewScaleRadius(rad, fovDims)
                scaledPos = self.modelToViewScaling(pos, fovPos, fovDims)
                self.drawSingleCell(scaledPos.astype(int), int(scaledRad), RED, cell.getPlayer())

    def drawCells(self, cells, fovPos, fovDims):
        for cell in cells:
            if cell.isInFov(fovPos, fovDims):
                rad = cell.getRadius()
                pos = numpy.array(cell.getPos())
                scaledRad = self.modelToViewScaleRadius(rad, fovDims)
                scaledPos = self.modelToViewScaling(pos, fovPos, fovDims)
                self.drawSingleCell(scaledPos.astype(int), int(scaledRad), cell.getColor(), cell.getPlayer())


    def drawSingleCell(self, pos, rad, color, player):
        pygame.draw.circle(self.screen, color, pos, rad)
        if player != None:
            font = pygame.font.SysFont(None, int(rad / 2))

            text = font.render(player.getName(), False, (0,0,0))
            pos = (pos[0] - text.get_width() / 2, pos[1] - text.get_height() / 2 )
            self.screen.blit(text, pos)


    def drawAllCells(self):
        fovPos = self.model.getFovPos()
        fovDims = self.model.getFovDims()

        self.drawCells(self.model.getPellets(), fovPos, fovDims)
        self.drawCells(self.model.getViruses(), fovPos, fovDims)
        self.drawCells(self.model.getPlayerCells(), fovPos, fovDims)


    def drawHumanStats(self):
        if self.model.hasHuman():
            totalMass = self.model.getHuman().getTotalMass()
            name = "Total Mass: " + str(int(totalMass))
            font = pygame.font.SysFont(None, int(30 + numpy.sqrt(totalMass)))
            text = font.render(name, False, (min(255,int(totalMass / 5)), min(100,int(totalMass / 10)), min(100,int(totalMass / 10))))
            pos = (0, self.height - text.get_height())
            self.screen.blit(text, pos)

    def draw(self):
        self.screen.fill(WHITE)
        self.drawAllCells()
        self.drawHumanStats()
        if self.model.getDebugStatus():
            self.drawDebugInfo()
        pygame.display.update()

    def model_event(self):
        if self.model.hasHuman() or self.model.hasSpectator:
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
        return rad * (self.screenDims[0] / fovDims[0])

    # Checks:
    def getScreenDims(self):
        return numpy.array([self.width, self.height])
