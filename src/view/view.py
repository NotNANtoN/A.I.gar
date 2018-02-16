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
        self.screenDims = numpy.array([self.width, self.height])
        self.model = model
        self.model.register_listener(self.model_event)
        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (0,30)
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.init()
        pygame.display.set_caption('A.I.gar')

        # Rendering fonts for the leaderboard and initializing it
        numbOfPlayers = min(10, len(model.getPlayers()))
        self.leaderBoardTextHeight = 25
        self.leaderBoardTitleHeight = 28
        self.leaderBoardFont = pygame.font.SysFont(None, self.leaderBoardTextHeight)
        leaderBoardTitleFont = pygame.font.SysFont(None, self.leaderBoardTitleHeight)
        #leaderBoardTitleFont.set_bold(True)
        self.leaderBoardTitle = leaderBoardTitleFont.render("Leaderboard", True, (255, 255, 255))
        self.leaderBoardWidth = self.leaderBoardTitle.get_width() + 15
        self.leaderBoardHeight = self.leaderBoardTitleHeight + self.leaderBoardTextHeight * numbOfPlayers + 2
        self.leaderBoard = pygame.Surface((self.leaderBoardWidth, self.leaderBoardHeight))  # the size of your rect
        self.leaderBoard.set_alpha(128)


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

    def drawCells(self, cells, fovPos, fovDims):
        for cell in cells:
            self.drawSingleCell(cell, fovPos, fovDims)


    def drawSingleCell(self, cell, fovPos, fovDims):
        unscaledRad = cell.getRadius()
        unscaledPos = numpy.array(cell.getPos())
        color = cell.getColor()
        player = cell.getPlayer()
        rad = int(self.modelToViewScaleRadius(unscaledRad, fovDims))
        pos = self.modelToViewScaling(unscaledPos, fovPos, fovDims).astype(int)

        if rad >= 4:
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], rad, color)
            if cell.getName() == "Virus":
                # Give Viruses a black surrounding circle
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], rad, (0,0,0))
            else:
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], rad, color)
        else:
            # Necessary to avoid that collectibles are drawn as little X's when the fov is huge
            pygame.draw.circle(self.screen, color, pos, rad)
        if player != None or (__debug__ and cell.getName() == "Virus"):
            font = pygame.font.SysFont(None, int(rad / 2))
            name = font.render(cell.getName(), True, (0,0,0))
            textPos = [pos[0] - name.get_width() / 2, pos[1] - name.get_height() / 2]
            self.screen.blit(name, textPos)

            if __debug__:
                mass = font.render("Mass:" + str(int(cell.getMass())), True, (0, 0, 0))
                textPos = [pos[0] - mass.get_width() / 2, pos[1] - mass.get_height() / 2 + name.get_height()]
                self.screen.blit(mass, textPos)
                if cell.getMergeTime() > 0:
                    text = font.render(str(int(cell.getMergeTime())), True, (0, 0, 0))
                    textPos = [pos[0] - text.get_width() / 2, pos[1] - text.get_height() / 2 + name.get_height() + mass.get_height()]
                    self.screen.blit(text, textPos)


    def drawAllCells(self):
        fovPos = self.model.getFovPos()
        fovDims = self.model.getFovDims()
        pellets = self.model.getField().getPelletsInFov(fovPos, fovDims)
        blobs = self.model.getField().getBlobsInFov(fovPos, fovDims)
        viruses = self.model.getField().getVirusesInFov(fovPos, fovDims)
        playerCells = self.model.getField().getPlayerCellsInFov(fovPos, fovDims)
        allCells = pellets + blobs + viruses + playerCells
        allCells.sort(key = lambda p: p.getMass())

        self.drawCells(allCells, fovPos, fovDims)


    def drawHumanStats(self):
        if self.model.hasHuman():
            totalMass = self.model.getHumans()[0].getTotalMass()
            name = "Total Mass: " + str(int(totalMass))
            font = pygame.font.SysFont(None, int(min(150, 30 + numpy.sqrt(totalMass))))
            text = font.render(name, False, (min(255,int(totalMass / 5)), min(100,int(totalMass / 10)), min(100,int(totalMass / 10))))
            pos = (0, self.height - text.get_height())
            self.screen.blit(text, pos)

    def drawLeaderBoard(self):
        self.leaderBoard.fill((0, 0, 0))
        players = self.model.getTopTenPlayers()
        numberOfPositionsShown = len(players)
        self.leaderBoard.blit(self.leaderBoardTitle, (8, self.leaderBoardTitleHeight / 4))
        for i in range(numberOfPositionsShown):
            currentPlayer = players[i]
            string = str(i + 1) + ". " + currentPlayer.getName() + ": " + str(int(currentPlayer.getTotalMass()))
            text = self.leaderBoardFont.render(string, True, (255, 255, 255))
            pos = (8, self.leaderBoardTitleHeight + i * self.leaderBoardTextHeight)
            self.leaderBoard.blit(text, pos)
        self.screen.blit(self.leaderBoard, (self.width - self.leaderBoardWidth - 10, 10))

    def draw(self):
        self.screen.fill(WHITE)
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
