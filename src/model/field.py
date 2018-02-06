from random import randint

from .cell import Cell

# The Field class is the main field on which cells of all sizes will move
# Its size depends on how many players are in the game
# It always contains a certain number of viruses and collectibles and regulates their number and spawnings

START_RADIUS = 30
MAX_COLLECTIBLE_SPAWN_PER_UPDATE = 5
COLLECTIBLE_SIZE = 30


class Field(object):
    def __init__(self):
        self.width = 0
        self.height = 0
        self.collectibles = []
        self.players = []
        self.viruses = []

    def initializePlayer(self, player):
        x = randint(0, self.width)
        y = randint(0, self.height)
        newCell = Cell(x, y, START_RADIUS, player.getColor())
        player.addCell(newCell)

    def initialize(self):
        for player in self.players:
            self.initializePlayer(player)

        self.spawnStuff()

    def update(self):
        self.updateViruses()
        self.updatePlayers()

        self.spawnStuff()

    def updateViruses(self):
        for virus in self.viruses:
            virus.update()

    def updatePlayers(self):
        for player in self.players:
            player.update(self.width, self.height)

    def spawnStuff(self):
        self.spawnCollectibles()
        self.spawnViruses()

    def spawnCollectibles(self):
        count = 0
        while len(self.collectibles) < 100 and count < MAX_COLLECTIBLE_SPAWN_PER_UPDATE:
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

    def spawnViruses(self):
        pass

    # Setters:
    def addPlayer(self, player):
        self.players.append(player)
        self.width = 120 * len(self.players)
        self.height = 100 * len(self.players)

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
