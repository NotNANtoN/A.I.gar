import time
import numpy

from .bot import Bot
from .field import Field
from .parameters import *
from .player import Player


# The model class is the main wrapper for the game engine.
# It contains the field and the players.
# It links the actions of the players to consequences in the field and updates information.


class Model(object):
    def __init__(self, width, height, debug):
        self.listeners = []
        self.debugStatus = debug

        self.players = []
        self.bots = []
        self.human = None
        self.spectator = None
        self.players = []
        self.field = Field()

        self.screenWidth = width
        self.screenHeight = height

    def initialize(self):
        self.field.initialize()

    def printDebugInfo(self):
        if self.hasHuman():
            print("-----")
            print("Human total mass: ", self.human.getTotalMass())
            print("Human first cell mass: ", self.human.cells[0].mass)
            print("Human merge time first cell: ", self.human.cells[0].mergeTime)
            print("Human cells:")
            for cell in self.human.getCells():
                print(cell)
            print("Cells nearby human first cell")
            for cell in self.field.hashTable.getNearbyObjects(self.human.cells[0]):
                print(cell)

            #print("Human wants to split: ", self.human.doSplit)
            print("-----")
            #fovPos = self.getFovPos()
            #fovDims = self.getFovDims()
            #print("FovPos: ", fovPos[0], "|", fovPos[1])
            #print("FovDims: ", fovDims[0], "|", fovDims[1])

            #humanPos = self.human.cells[0].getPos()
            #print("HumanPos: ", humanPos[0], "|", humanPos[1])


    def update(self):
        # Get the decisions of the bots/human. Update the field accordingly.
        for bot in self.bots:
            bot.update()
        self.field.update()

        self.respawnPlayers()
        self.notify()
        # wait = input("PRESS ENTER TO CONTINUE.")
        time.sleep(1 / FPS)
        if self.debugStatus == True:
            self.printDebugInfo()

    def respawnPlayers(self):
        for dp in self.field.getDeadPlayers():
            self.field.removeDeadPlayer(dp)
            self.field.initializePlayer(dp)
            self.field.addPlayer(dp)

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

    def addSpectator(self):
        self.spectator = True

    # Checks:
    def hasHuman(self):
        return self.human is not None

    def hasSpectator(self):
        return self.spectator is not None

    # Getters:
    def getHuman(self):
        return self.human

    def getFovPos(self):
        if self.hasHuman():
            fovPos = numpy.array(self.human.getFovPos())
        else:
            fovPos = numpy.array([self.field.getWidth() / 2, self.field.getHeight() / 2])
        return fovPos

    def getFovDims(self):
        if self.hasHuman():
            fovDims = numpy.array(self.human.getFovDims())
        else:
            fovDims = numpy.array([self.field.getWidth(), self.field.getHeight()])
        return fovDims

    def getField(self):
        return self.field

    def getCollectibles(self):
        return self.field.getCollectibles()

    def getViruses(self):
        return self.field.getViruses()

    def getPlayerCells(self):
        return self.field.getPlayerCells()

    def getDebugStatus(self):
        return self.debugStatus

    # MVC related method
    def register_listener(self, listener):
        self.listeners.append(listener)

    def notify(self):
        for listener in self.listeners:
            listener()
