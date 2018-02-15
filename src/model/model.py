import time
import numpy

from .bot import Bot
from .field import Field
from .parameters import *
from .player import Player
import matplotlib.pyplot as plt

# The model class is the main wrapper for the game engine.
# It contains the field and the players.
# It links the actions of the players to consequences in the field and updates information.


class Model(object):
    def __init__(self, width, height, debug, viewEnabled):
        self.listeners = []
        self.debugStatus = debug
        self.viewEnabled = viewEnabled

        self.players = []
        self.bots = []
        self.human = None
        self.playerSpectator = None
        self.spectatedPlayer = None
        self.players = []
        self.field = Field()
        self.field.setDebug(debug)
        self.screenWidth = width
        self.screenHeight = height
        self.counter = 0
        self.timings = []
        self.maxMasses = []


    def initialize(self):
        self.field.initialize()

    def printDebugInfo(self):
        if self.hasHuman():
            pass
            #print("Human merge time first cell: ", self.human.cells[0].mergeTime)

    def update(self):

        timeStart = time.time()
        # Get the decisions of the bots/human. Update the field accordingly.
        for bot in self.bots:
            bot.update()
        self.field.update()
        if self.viewEnabled:
            self.notify()
        if self.debugStatus == True:
            self.printDebugInfo()
        time.sleep(max( (1/FPS) - (time.time() - timeStart),0))

        #self.visualize(timeStart)

    def visualize(self, timeStart):

        print(" ")
        print("time since update start: ", str(time.time() - timeStart))
        print("counter: ", self.counter)
        playerCells = self.field.getPlayerCells()
        maxMass = max(playerCells, key=lambda p: p.getMass()).getMass()
        print("biggest cell mass: ", maxMass)
        self.counter += 1
        self.timings.append(time.time() - timeStart)
        self.maxMasses.append(maxMass)
        print(" ")

        if self.counter % 100 == 0:
            plt.plot(self.maxMasses, self.timings, 'o')
            plt.xlabel("Maximum Masses")
            plt.ylabel("Time taken for update")
            print("mean time: ", str(numpy.mean(self.timings)))
            plt.show()

    # Setters:
    def createPlayer(self, name):
        newPlayer = Player(name)
        self.addPlayer(newPlayer)
        return newPlayer

    def createBot(self):
        name = "Bot " + str(len(self.bots))
        newPlayer = self.createPlayer(name)
        bot = Bot(newPlayer, self.field)
        self.addBot(bot)

    def createHuman(self, name):
        newPlayer = self.createPlayer(name)
        self.addHuman(newPlayer)

    def addPlayer(self, player):
        self.players.append(player)
        self.field.addPlayer(player)

    def addBot(self, bot):
        self.bots.append(bot)

    def addHuman(self, player):
        self.human = player

    def addPlayerSpectator(self):
        self.playerSpectator = True
        self.setSpectatedPlayer(self.players[0])

    def setSpectatedPlayer(self, player):
        self.spectatedPlayer = player

    def setViewEnabled(self, boolean):
        self.viewEnabled = boolean

    # Checks:
    def hasHuman(self):
        return self.human is not None

    def hasPlayerSpectator(self):
        return self.playerSpectator is not None

    # Getters:
    def getHuman(self):
        return self.human

    def getFovPos(self):
        if self.hasHuman():
            fovPos = numpy.array(self.human.getFovPos())
        elif self.hasPlayerSpectator():
            fovPos = numpy.array(self.spectatedPlayer.getFovPos())
        else:
            fovPos = numpy.array([self.field.getWidth() / 2, self.field.getHeight() / 2])
        return fovPos

    def getFovDims(self):
        if self.hasHuman():
            fovDims = numpy.array(self.human.getFovDims())
        elif self.hasPlayerSpectator():
            fovDims = numpy.array(self.spectatedPlayer.getFovDims())
        else:
            fovDims = numpy.array([self.field.getWidth(), self.field.getHeight()])
        return fovDims

    def getField(self):
        return self.field

    def getPellets(self):
        return self.field.getPellets()

    def getViruses(self):
        return self.field.getViruses()

    def getPlayers(self):
        return self.players

    def getPlayerCells(self):
        return self.field.getPlayerCells()

    def getDebugStatus(self):
        return self.debugStatus

    def getSpectatedPlayer(self):
        if self.hasHuman():
            return self.human
        if self.hasPlayerSpectator():
            return self.spectatedPlayer
        return None

    # MVC related method
    def register_listener(self, listener):
        self.listeners.append(listener)

    def notify(self):
        for listener in self.listeners:
            listener()
