from random import randint
import numpy
from .cell import Cell
from .parameters import *

# The Field class is the main field on which cells of all sizes will move
# Its size depends on how many players are in the game
# It always contains a certain number of viruses and collectibles and regulates their number and spawnings




class Field(object):
    def __init__(self):
        self.width = 0
        self.height = 0
        self.collectibles = []
        self.players = []
        self.viruses = []
        self.maxCollectibleCount = MAX_COLLECTIBLE_COUNT

    def initializePlayer(self, player):
        x = randint(0, self.width)
        y = randint(0, self.height)
        newCell = Cell(x, y, START_RADIUS, player.getColor())
        player.addCell(newCell)

    def initialize(self):
        for player in self.players:
            self.initializePlayer(player)

        self.spawnStuff(MAX_COLLECTIBLE_SPAWN_PER_UPDATE)

    def update(self):
        self.updateViruses()
        self.updatePlayers()
        self.checkCollisions()

        self.spawnStuff(MAX_COLLECTIBLE_SPAWN_PER_UPDATE)

    def checkCollisions(self):
        self.collectibleCollisions()

    def collectibleCollisions(self):
        for player in self.players:
            for cell in player.getCells():
                for collectible in self.collectibles:
                    if cell.overlap(collectible):
                        self.eat(cell, collectible)

        # Eats the other cell. Increases own size and deletes other

    # Cell1 eats Cell2. Therefore Cell1 grows and Cell2 is deleted
    def eat(self,cell1, cell2):
        cell1.grow(cell2.getMass())
        self.deleteCollectible(cell2)

    def deleteCollectible(self, collectible):
        self.collectibles.remove(collectible)




    def updateViruses(self):
        for virus in self.viruses:
            virus.update()


    def updatePlayers(self):
        for player in self.players:
            player.update(self.width, self.height)

    def spawnStuff(self, maxSpawns):
        self.spawnCollectibles(maxSpawns)
        self.spawnViruses(maxSpawns)

    def spawnCollectibles(self, maxSpawns):
        count = 0
        while len(self.collectibles) < self.maxCollectibleCount and count < maxSpawns:
            self.spawnCollectible()
            count += 1

    def spawnCollectible(self):
        xPos = randint(0, self.width)
        yPos = randint(0, self.height)
        color = (randint(0, 255), randint(0, 255), randint(0, 255))
        collectible = Cell(xPos, yPos, COLLECTIBLE_SIZE, color)
        self.addCollectible(collectible)

    def addCollectible(self, collectible):
        self.collectibles.append(collectible)

    def spawnViruses(self, maxSpawns):
        pass

    # Setters:
    def addPlayer(self, player):
        self.players.append(player)
        self.width = numpy.round(120 * numpy.sqrt(len(self.players)))
        self.height = numpy.round(100 * numpy.sqrt(len(self.players)))

    # Getters:
    def getWidth(self):
        return self.width

    def getHeight(self):
        return self.height

    def getCollectibles(self):
        return self.collectibles

    def getViruses(self):
        return self.viruses

    def getPlayerCells(self):
        cells = []
        for player in self.players:
            cells += player.getCells()
        return cells
