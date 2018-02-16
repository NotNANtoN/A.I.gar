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
    def __init__(self, width, height, viewEnabled):
        self.listeners = []
        self.viewEnabled = viewEnabled

        self.players = []
        self.bots = []
        self.humans = []
        self.playerSpectator = None
        self.spectatedPlayer = None
        self.players = []
        self.field = Field()
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
        timeProcessStart = time.process_time()
        # Get the decisions of the bots/human. Update the field accordingly.
        for bot in self.bots:
            bot.update()
        self.field.update()
        if self.viewEnabled:
            self.notify()
        if __debug__:
            self.printDebugInfo()
        time.sleep(max( (1/FPS) - (time.time() - timeStart),0))

            
        self.visualize(timeProcessStart)


    def visualize(self, timeStart):
        if self.counter % 100 == 0:
            print(" ")
            print("time since update start: ", str(time.process_time() - timeStart))
            print("counter: ", self.counter)
            playerCells = self.field.getPlayerCells()
            maxMass = max(playerCells, key=lambda p: p.getMass()).getMass()
            print("biggest cell mass: ", maxMass)
            #self.timings.append(time.time() - timeStart)
            #self.maxMasses.append(maxMass)
            print(" ")
        self.counter += 1
        '''
        if self.counter % 100 == 0:
            plt.plot(self.maxMasses, self.timings, 'o')
            plt.xlabel("Maximum Masses")
            plt.ylabel("Time taken for update")
            print("mean time: ", str(numpy.mean(self.timings)))
            plt.show()
        '''

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

    def addHuman(self, human):
        self.humans.append(human)

    def addPlayerSpectator(self):
        self.playerSpectator = True
        self.setSpectatedPlayer(self.players[0])

    def setSpectatedPlayer(self, player):
        self.spectatedPlayer = player

    def setViewEnabled(self, boolean):
        self.viewEnabled = boolean

    # Checks:
    def hasHuman(self):
        return self.humans is not None

    def hasPlayerSpectator(self):
        return self.playerSpectator is not None

    # Getters:
    def getHumans(self):
        return self.humans

    def getFovPos(self):
        if self.hasHuman():
            fovPos = numpy.array(self.humans[0].getFovPos())
        elif self.hasPlayerSpectator():
            fovPos = numpy.array(self.spectatedPlayer.getFovPos())
        else:
            fovPos = numpy.array([self.field.getWidth() / 2, self.field.getHeight() / 2])
        return fovPos

    def getFovDims(self):
        if self.hasHuman():
            fovDims = numpy.array(self.humans[0].getFovDims())
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
