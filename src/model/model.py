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
    def __init__(self, viewEnabled):
        self.listeners = []
        self.viewEnabled = viewEnabled

        self.players = []
        self.bots = []
        self.humans = []
        self.playerSpectator = None
        self.spectatedPlayer = None
        self.players = []
        self.field = Field()
        self.screenWidth = None
        self.screenHeight = None
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
        timeProcessStart = time.process_time()
        # Get the decisions of the bots/human. Update the field accordingly.
        for bot in self.bots:
            bot.update()
        self.field.update()
        if self.viewEnabled:
            self.notify()
        if __debug__:
            self.printDebugInfo()
        if self.humans:
            time.sleep(max( (1/FPS) - (time.time() - timeStart),0))

            
        self.visualize(timeProcessStart)


    def visualize(self, timeStart):
        playerCells = self.field.getPlayerCells()
        maxMass = max(playerCells, key=lambda p: p.getMass()).getMass()
        stepsTillUpdate = 100
        self.timings.append(time.process_time() - timeStart)
        if self.counter % stepsTillUpdate == 0:
            print(" ")
            print("avg time since update start for the last ", stepsTillUpdate, " steps: ", str(round(numpy.mean(self.timings[len(self.timings) - stepsTillUpdate:]),3)))
            print("counter: ", self.counter)
            print("biggest cell mass: ", maxMass)
            print(" ")
        self.counter += 1
        self.maxMasses.append(maxMass)

        if self.counter % 5000 == 0:
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

    def createBot(self, type):
        name = "Bot " + str(len(self.bots))
        newPlayer = self.createPlayer(name)
        bot = Bot(newPlayer, self.field, type)
        self.addBot(bot)

    def createHuman(self, name):
        newPlayer = self.createPlayer(name)
        self.addHuman(newPlayer)

    def addPlayer(self, player):
        self.players.append(player)
        self.field.addPlayer(player)

    def addBot(self, bot):
        self.bots.append(bot)

    def addHuman(self, human):
        self.humans.append(human)

    def addPlayerSpectator(self):
        self.playerSpectator = True
        self.setSpectatedPlayer(self.players[0])

    def setSpectatedPlayer(self, player):
        self.spectatedPlayer = player

    def setViewEnabled(self, boolean):
        self.viewEnabled = boolean

    def setScreenSize(self, width, height):
        self.screenWidth = width
        self.screenHeight = height

    # Checks:
    def hasHuman(self):
        return bool(self.humans)

    def hasPlayerSpectator(self):
        return self.playerSpectator is not None

    # Getters:
    def getTopTenPlayers(self):
        players = self.getPlayers()[:]
        players.sort(key=lambda p: p.getTotalMass(), reverse=True)
        return players[0:10]


    def getHumans(self):
        return self.humans

    def getFovPos(self, humanNr):
        if self.hasHuman():
            fovPos = numpy.array(self.humans[humanNr].getFovPos())
        elif self.hasPlayerSpectator():
            fovPos = numpy.array(self.spectatedPlayer.getFovPos())
        else:
            fovPos = numpy.array([self.field.getWidth() / 2, self.field.getHeight() / 2])
        return fovPos

    def getFovSize(self, humanNr):
        if self.hasHuman():
            fovSize = self.humans[humanNr].getFovSize()
        elif self.hasPlayerSpectator():
            fovSize = self.spectatedPlayer.getFovSize()
        else:
            fovSize = self.field.getWidth()
        return fovSize

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

    def getSpectatedPlayer(self):
        if self.hasHuman():
            return self.humans
        if self.hasPlayerSpectator():
            return self.spectatedPlayer
        return None

    # MVC related method
    def register_listener(self, listener):
        self.listeners.append(listener)

    def notify(self):
        for listener in self.listeners:
            listener()
