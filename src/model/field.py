from random import randint
# The Field class is the main field on which cells of all sizes will move
# Its size depends on how many players are in the game
# It always contains a certain number of viruses and collectibles and regulates their number and spawnings

class Field(object):
    MAXCOLLECTIBLESPAWNPERUPDATE = 5
    COLLECTIBLESIZE = 5

    def __init__(self):
        self.width = 0
        self.height = 0
        self.collectibles = []
        self.players = []
        self.viruses = []

    def update(self):
        self.updateViruses()
        self.updatePlayers()

        self.spawnStuff()


    def updateViruses(self):
        for virus in self.viruses:
            virus.update()

    def updatePlayers(self):
        for player in self.players:
            player.update()

    def spawnStuff(self):
        self.spawnCollectibles()
        self.spawnViruses()

    def spawnCollectibles(self):
        count = 0
        while( len(collectibles) < 100 and count < MAXCOLLECTIBLESPAWNPERUPDATE ):
            self.spawnCollectible(self)
            count += 1

    def spawnCollectible(self):
        xPos = randint(0, self.width)
        yPos = randint(0, self.height)
        color = (randint(0,255), randint(0,255), randint(0,255))
        collectible = Cell(xPos, yPos, COLLECTIBLESIZE ,color)

    def spawnViruses():
        pass

    # Setters:
    def addPlayer(self, player):
        self.players.append(player)

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