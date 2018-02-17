import pygame
import numpy
import os
from pygame import gfxdraw

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)


class View:

    def __init__(self, model, width, height):
        self.width = width
        self.height = height
        #self.screenDims = numpy.array([self.width, self.height])
        self.screenDims = None
        self.model = model
        self.numberOfScreens = None
        self.model.register_listener(self.model_event)
        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (0,30)
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.splitScreen = False
        self.playerScreens = []
        self.setNumberOfScreens()
        pygame.init()
        pygame.display.set_caption('A.I.gar')

    def setNumberOfScreens(self):
        humansNr = len(self.model.getHumans())
        if humansNr > 1:
            self.numberOfScreens = humansNr
            self.screenDims = numpy.array([int((self.width - (humansNr - 1)) / humansNr), self.height])
            self.splitScreen = True
            for playerScreen in range(humansNr):
                screen = pygame.Surface(self.screenDims)
                self.playerScreens.append(screen)
        else:
            self.numberOfScreens = 1
            self.screenDims = numpy.array([self.width, self.height])
            self.playerScreens.append(self.screen)

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
            for cell in self.model.field.pelletHashTable.getNearbyObjects(self.model.getHumans()[0].cells[0]):
                rad = cell.getRadius()
                pos = numpy.array(cell.getPos())
                scaledRad = self.modelToViewScaleRadius(rad, fovDims)
                scaledPos = self.modelToViewScaling(pos, fovPos, fovDims)
                self.drawSingleCell(scaledPos.astype(int), int(scaledRad), RED, cell.getPlayer())

    def drawCells(self, cells, fovPos, fovDims, screen):
        for cell in cells:
            rad = cell.getRadius()
            pos = numpy.array(cell.getPos())
            scaledRad = self.modelToViewScaleRadius(rad, fovDims)
            scaledPos = self.modelToViewScaling(pos, fovPos, fovDims)
            self.drawSingleCell(scaledPos.astype(int), int(scaledRad), cell.getColor(), cell.getPlayer(), screen)


    def drawSingleCell(self, pos, rad, color, player, screen):
        if rad >= 4:
            pygame.gfxdraw.aacircle(screen, pos[0], pos[1], rad, color)
            pygame.gfxdraw.filled_circle(screen, pos[0], pos[1], rad, color)
        else:
            pygame.draw.circle(screen, color, pos, rad)
        if player != None:
            font = pygame.font.SysFont(None, int(rad / 2))

            text = font.render(player.getName(), False, (0,0,0))
            pos = (pos[0] - text.get_width() / 2, pos[1] - text.get_height() / 2 )
            screen.blit(text, pos)


    def drawAllCells(self):
        for humanNr in range(self.numberOfScreens):
            fovPos = self.model.getFovPos(humanNr)
            fovDims = self.model.getFovDims(humanNr)
            pellets = self.model.getField().getPelletsInFov(fovPos, fovDims)
            blobs = self.model.getField().getBlobsInFov(fovPos, fovDims)
            viruses = self.model.getField().getVirusesInFov(fovPos, fovDims)
            playerCells = self.model.getField().getPlayerCellsInFov(fovPos, fovDims)
            allCells = pellets + blobs + viruses + playerCells
            allCells.sort(key = lambda p: p.getMass())

            self.drawCells(allCells, fovPos, fovDims, self.playerScreens[humanNr])


    def drawHumanStats(self):
        if self.model.hasHuman():
            for humanNr in range(self.numberOfScreens):
                totalMass = self.model.getHumans()[humanNr].getTotalMass()
                name = "Total Mass: " + str(int(totalMass))
                font = pygame.font.SysFont(None, int(max(150, 30 + numpy.sqrt(totalMass))))
                text = font.render(name, False, (min(255,int(totalMass / 5)), min(100,int(totalMass / 10)), min(100,int(totalMass / 10))))
                pos = (self.screenDims[0], self.height - text.get_height())
                self.playerScreens[humanNr].blit(text, pos)

    def drawScreenSeparators(self):
        for screenNumber in range(self.numberOfScreens):
            x = self.screenDims[0] * screenNumber + 1
            pygame.gfxdraw.line(self.screen, x, 0, x, self.screenDims[1], BLACK)


    def drawLeaderBoard(self):
        pass

    def draw(self):
        self.screen.fill(WHITE)
        if self.splitScreen:
            for screenNr in range(len(self.playerScreens)):
                self.screen.blit(self.playerScreens[screenNr], (self.screenDims[0] * screenNr + screenNr, 0))
                self.playerScreens[screenNr].fill(WHITE)
            self.drawScreenSeparators()
        self.drawAllCells()
        self.drawHumanStats()
        self.drawLeaderBoard()
        if __debug__:
            self.drawDebugInfo()
        pygame.display.update()

    def model_event(self):
        self.draw()

    def modelToViewScaling(self, pos, fovPos, fovDims):
        adjustedPos = pos - fovPos + (fovDims / 2)
        scaledPos = adjustedPos * (self.screenDims / fovDims)
        return scaledPos

    def viewToModelScaling(self, pos, humanNr):
        fovPos = self.model.getFovPos(humanNr)
        fovDims = self.model.getFovDims(humanNr)
        scaledPos = pos / (self.screenDims / fovDims)
        adjustedPos = scaledPos + fovPos - (fovDims / 2)
        return adjustedPos

    def modelToViewScaleRadius(self, rad, fovDims):
        return rad * (self.screenDims[0] / fovDims[0])

    # Checks:
    def getScreenDims(self):
        return self.screenDims

    def getWindowWidth(self):
        return self.width

    def getWindowHeight(self):
        return self.height
