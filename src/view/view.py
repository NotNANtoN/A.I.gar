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

        # Rendering fonts for the leaderboard and initializing it
        numbOfPlayers = min(10, len(model.getPlayers()))
        self.leaderBoardTextHeight = 25
        self.leaderBoardTitleHeight = 28
        self.leaderBoardFont = pygame.font.SysFont(None, self.leaderBoardTextHeight)
        leaderBoardTitleFont = pygame.font.SysFont(None, self.leaderBoardTitleHeight)
        # leaderBoardTitleFont.set_bold(True)
        self.leaderBoardTitle = leaderBoardTitleFont.render("Leaderboard", True, (255, 255, 255))
        self.leaderBoardWidth = self.leaderBoardTitle.get_width() + 15
        self.leaderBoardHeight = self.leaderBoardTitleHeight + self.leaderBoardTextHeight * numbOfPlayers + 2
        self.leaderBoard = pygame.Surface((self.leaderBoardWidth, self.leaderBoardHeight))  # the size of your rect
        self.leaderBoard.set_alpha(128)

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
        for humanNr in range(self.numberOfScreens):
            cells = self.model.getPlayerCells()
            fovPos = self.model.getFovPos(humanNr)
            fovDims = self.model.getFovDims(humanNr)
            for cell in cells:
                pos = numpy.array(cell.getPos())
                scaledPos = self.modelToViewScaling(pos, fovPos, fovDims)
                pygame.draw.line(self.playerScreens[humanNr], RED, scaledPos.astype(int),
                                 numpy.array(cell.getVelocity()) * 10 +
                                 numpy.array(scaledPos.astype(int)))

    def drawCells(self, cells, fovPos, fovDims, screen):
        for cell in cells:
            self.drawSingleCell(cell, fovPos, fovDims, screen)

    def drawSingleCell(self, cell, fovPos, fovDims, screen):
        unscaledRad = cell.getRadius()
        unscaledPos = numpy.array(cell.getPos())
        color = cell.getColor()
        player = cell.getPlayer()
        rad = int(self.modelToViewScaleRadius(unscaledRad, fovDims))
        pos = self.modelToViewScaling(unscaledPos, fovPos, fovDims).astype(int)
        if rad >= 4:
            pygame.gfxdraw.filled_circle(screen, pos[0], pos[1], rad, color)
            if cell.getName() == "Virus":
                # Give Viruses a black surrounding circle
                pygame.gfxdraw.aacircle(screen, pos[0], pos[1], rad, (0,0,0))
            else:
                pygame.gfxdraw.aacircle(screen, pos[0], pos[1], rad, color)
        else:
            # Necessary to avoid that collectibles are drawn as little X's when the fov is huge
            pygame.draw.circle(screen, color, pos, rad)
        if player != None or (__debug__ and cell.getName() == "Virus"):
            font = pygame.font.SysFont(None, int(rad / 2))
            name = font.render(cell.getName(), True, (0,0,0))
            textPos = [pos[0] - name.get_width() / 2, pos[1] - name.get_height() / 2]
            screen.blit(name, textPos)
            if __debug__:
                mass = font.render("Mass:" + str(int(cell.getMass())), True, (0, 0, 0))
                textPos = [pos[0] - mass.get_width() / 2, pos[1] - mass.get_height() / 2 + name.get_height()]
                screen.blit(mass, textPos)
                if cell.getMergeTime() > 0:
                    text = font.render(str(int(cell.getMergeTime())), True, (0, 0, 0))
                    textPos = [pos[0] - text.get_width() / 2, pos[1] - text.get_height() / 2 + name.get_height() + mass.get_height()]
                    screen.blit(text, textPos)


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
                font = pygame.font.SysFont(None, int(min(150, 30 + numpy.sqrt(totalMass))))
                text = font.render(name, False, (min(255,int(totalMass / 5)), min(100,int(totalMass / 10)), min(100,int(totalMass / 10))))
                pos = (self.screenDims[0], self.height - text.get_height())
                self.playerScreens[humanNr].blit(text, pos)

    def drawScreenSeparators(self):
        for screenNumber in range(self.numberOfScreens):
            x = self.screenDims[0] * screenNumber + 1
            pygame.gfxdraw.line(self.screen, x, 0, x, self.screenDims[1], BLACK)


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
        for screen in self.playerScreens:
            screen.blit(self.leaderBoard, (self.screenDims[0] - self.leaderBoardWidth - 10, 10))

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
