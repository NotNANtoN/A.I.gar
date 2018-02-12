from random import randint
import numpy
from .cell import Cell
from .parameters import *
from .spatialHashTable import spatialHashTable


# The Field class is the main field on which cells of all sizes will move
# Its size depends on how many players are in the game
# It always contains a certain number of viruses and collectibles and regulates their number and spawnings


class Field(object):
    def __init__(self):
        self.width = 0
        self.height = 0
        self.pellets = []
        self.players = []
        self.deadPlayers = []
        self.viruses = []
        self.maxCollectibleCount = None
        self.pelletHashtable = None
        self.playerHashtable = None
        self.virusHashtable = None

    def initializePlayer(self, player):
        x = randint(0, self.width)
        y = randint(0, self.height)
        newCell = Cell(x, y, START_MASS, player)
        player.addCell(newCell)

    def initialize(self):
        self.width = numpy.round(SIZE_INCREASE_PER_PLAYER * numpy.sqrt(len(self.players)))
        self.height = numpy.round(SIZE_INCREASE_PER_PLAYER * numpy.sqrt(len(self.players)))
        self.pelletHashtable = spatialHashTable(self.width, self.height, HASH_CELL_SIZE)
        self.playerHashtable = spatialHashTable(self.width, self.height, HASH_CELL_SIZE)
        self.virusHashtable = spatialHashTable(self.width, self.height, HASH_CELL_SIZE)
        for player in self.players:
            self.initializePlayer(player)
        self.maxCollectibleCount = self.width * self.height * MAX_COLLECTIBLE_DENSITY

        self.spawnStuff()

    def update(self):
        self.updateViruses()
        self.updatePlayers()
        self.mergePlayerCells()
        self.updateHashTables()
        self.checkCollisions()
        self.spawnStuff()

    def updateHashTables(self):
        self.playerHashtable.clearBuckets()
        for player in self.players:
            playerCells = player.getCells()
            self.playerHashtable.insertAllObjects(playerCells)


    def mergePlayerCells(self):
        for player in self.players:
            cells = player.getMergableCells()
            if len(cells) > 1:
                for i in range(len(cells)):
                    if not cells[i].isAlive():
                        continue
                    for j in range(i + 1, len(cells)):
                        if cells[i] is cells[j] or not cells[j].isAlive():
                            continue
                        if cells[i].overlap(cells[j]):
                            self.mergeCells(cells[i], cells[j])
                        elif cells[j].overlap(cells[i]):
                            self.mergeCells(cells[j], cells[i])


    def checkCollisions(self):
        self.collectibleCollisions()
        self.playerCollisions()

    def collectibleCollisions(self):
        for player in self.players:
            for cell in player.getCells():
                for collectible in self.pelletHashtable.getNearbyObjects(cell):
                    if  cell.overlap(collectible):
                        self.eatCollectible(cell, collectible)


    def playerCollisions(self):
        for player in self.players:
            for playerCell in player.getCells():
                if not playerCell.isAlive():
                    continue
                opponentCells = self.playerHashtable.getNearbyEnemyObjects(playerCell)
                #print("Opponent cells near ", playerCell, ":")
                for opponentCell in opponentCells:
                        #print(opponentCell)
                        if playerCell.overlap(opponentCell):
                            #print("OOOOOOOOOOOOOOVERLAP")
                            if playerCell.getMass() > 1.25 * opponentCell.getMass():
                                self.eatPlayerCell(playerCell, opponentCell)
                            elif playerCell.getMass() * 1.25 < opponentCell.getMass():
                                self.eatPlayerCell(opponentCell, playerCell)
                                break
        '''
        for i in range(len(self.players)):
            for j in range(i + 1, len(self.players)):
                if self.players[i] is self.players[j]:
                    continue
                for playerCell in self.players[i].getCells():
                    if not playerCell.isAlive():
                        continue
                    for opponentCell in self.players[j].getCells():
                        if not opponentCell.isAlive():
                            continue
                        if playerCell.overlap(opponentCell):
                            if playerCell.getMass() > 1.25 * opponentCell.getMass():
                                self.eatPlayerCell(playerCell, opponentCell, self.players[j])
                            elif playerCell.getMass() * 1.25 < opponentCell.getMass():
                                self.eatPlayerCell(opponentCell, playerCell, self.players[i])
                                break
        '''

    # Cell1 eats Cell2. Therefore Cell1 grows and Cell2 is deleted
    def eatCollectible(self, cell, collectible):
        cell.grow(collectible.getMass())
        self.pellets.remove(collectible)
        self.pelletHashtable.deleteObject(collectible)
        collectible.setAlive(False)

    def eatPlayerCell(self, largerCell, smallerCell):
        largerCell.grow(smallerCell.getMass())
        self.deletePlayerCell(smallerCell)

    def mergeCells(self, biggerCell, smallerCell):
        #print(biggerCell, " AND ", smallerCell, " MERGED!")
        biggerCell.setMass(biggerCell.getMass() + smallerCell.getMass())
        self.deletePlayerCell(smallerCell)

    def deletePlayerCell(self, playerCell):
        self.playerHashtable.deleteObject(playerCell)
        player = playerCell.getPlayer()
        player.removeCell(playerCell)
        if len(player.getCells()) == 0:
            player.setDead()
            self.deadPlayers.append(player)

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
        # If beginning of the game, spawn all collectibles at once
        #if len(self.collectibles) == 0:
            while len(self.pellets) < self.maxCollectibleCount:
                self.spawnCollectible()
                '''
        else:  # Else, spawn at the max spawn rate
            count = 0
            totalMaxSpawnRate = MAX_COLLECTIBLE_SPAWN_PER_UPDATE * self.width * self.height
            while len(self.collectibles) < self.maxCollectibleCount and count < MAX_COLLECTIBLE_SPAWN_PER_UPDATE:
                self.spawnCollectible()
                count += 1
                '''

    def spawnCollectible(self):
        xPos = randint(0, self.width)
        yPos = randint(0, self.height)
        collectible = Cell(xPos, yPos, COLLECTIBLE_SIZE, None)
        self.addCollectible(collectible)

    def spawnViruses(self):
        pass

    def removeDeadPlayer(self, player):
        self.deadPlayers.remove(player)

    def addCollectible(self, collectible):
        self.pelletHashtable.insertObject(collectible)
        self.pellets.append(collectible)

    # Setters:
    def addPlayer(self, player):
        player.setAlive()
        self.players.append(player)

    # Getters:
    def getEnemyPlayerCellsInFov(self, fovPlayer):
        fovPos = fovPlayer.getFovPos()
        fovDims = fovPlayer.getFovDims()
        cellsInFov = self.getCellsFromHashtableInFov(self.playerHashtable, fovPos, fovDims)
        opponentCellsInFov = []
        for playerCell in cellsInFov:
            # If the playerCell is an opponent Cell
            if playerCell.getName() != fovPlayer.getName() and playerCell.isInFov(fovPos, fovDims):
                opponentCellsInFov.append(playerCell)
        return opponentCellsInFov


    def getPelletsInFov(self, fovPlayer):
        fovPos = fovPlayer.getFovPos()
        fovDims = fovPlayer.getFovDims()
        cellsInFov = self.getCellsFromHashtableInFov(self.pelletHashtable, fovPos, fovDims)
        pelletsInFov = []
        for pellet in cellsInFov:
            if pellet.isInFov(fovPos, fovDims):
                pelletsInFov.append(pellet)
        return pelletsInFov

    def getCellsFromHashtableInFov(self, hashtable, fovPos, fovDims):
        fovCell = Cell(fovPos[0], fovPos[1], 1, None)
        fovCell.setRadius(fovDims[0] / 2)
        return hashtable.getNearbyObjects(fovCell)


    def getWidth(self):
        return self.width

    def getHeight(self):
        return self.height

    def getCollectibles(self):
        return self.pellets

    def getViruses(self):
        return self.viruses

    def getPlayerCells(self):
        cells = []
        for player in self.players:
            cells += player.getCells()
        return cells

    def getDeadPlayers(self):
        return self.deadPlayers

    def getPlayers(self):
        return self.players
