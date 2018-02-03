
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