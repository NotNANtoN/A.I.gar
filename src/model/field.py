import numpy
from .cell import Cell
from .parameters import *
from .spatialHashTable import spatialHashTable


# The Field class is the main field on which cells of all sizes will move
# Its size depends on how many players are in the game
# It always contains a certain number of viruses and collectibles and regulates their number and spawnings


class Field(object):
    def __init__(self):
        self.debug = False
        self.width = 0
        self.height = 0
        self.pellets = []
        self.players = []
        self.ejectedBlobs = [] # Ejected particles become pellets once momentum is lost
        self.deadPlayers = []
        self.viruses = []
        self.maxCollectibleCount = None
        self.maxVirusCount = None
        self.pelletHashtable = None
        self.playerHashtable = None
        self.virusHashtable = None

    def initializePlayer(self, player):
        x = numpy.random.randint(0, self.width)
        y = numpy.random.randint(0, self.height)

        newCell = Cell(x, y, START_MASS, player)
        player.addCell(newCell)
        player.setAlive()


    def initialize(self):
        self.width = numpy.round(SIZE_INCREASE_PER_PLAYER * numpy.sqrt(len(self.players)))
        self.height = numpy.round(SIZE_INCREASE_PER_PLAYER * numpy.sqrt(len(self.players)))
        self.pelletHashtable = spatialHashTable(self.width, self.height, HASH_CELL_SIZE)
        self.ejectedBlobHashTable = spatialHashTable(self.width, self.height, HASH_CELL_SIZE)
        self.playerHashtable = spatialHashTable(self.width, self.height, HASH_CELL_SIZE)
        self.virusHashtable = spatialHashTable(self.width, self.height, HASH_CELL_SIZE)
        for player in self.players:
            self.initializePlayer(player)
        self.maxCollectibleCount = self.width * self.height * MAX_COLLECTIBLE_DENSITY
        self.maxVirusCount = self.width * self.height * MAX_VIRUS_DENSITY
        self.spawnStuff()

    def update(self):
        self.updateViruses()
        self.updateEjectedBlobs()
        self.updatePlayers()
        self.updateHashTables()
        self.mergePlayerCells()
        self.checkOverlaps()
        self.spawnStuff()

    def updateViruses(self):
        for virus in self.viruses:
            virus.updateMomentum()
            virus.updatePos(self.width, self.height)

    def updateEjectedBlobs(self):
        for blob in self.ejectedBlobs:
            blob.updateMomentum()
            blob.updatePos(self.width, self.hieght)

    def updatePlayers(self):
        for player in self.players:
            player.update(self.width, self.height)
            updateEjections(player)
        self.handlePlayerCollisions()

    def updateEjections(self, player):
        for cell in player.getCells():
            if cell.blobToBeEjected():
                blobSpawnPos = cell.eject(player.getCommandPoint)
                ejectedBlob = Cell(blobSpawnPos[0], blobSpawnPos[1], EJECTEDBLOB_BASE_MASS, None)
                ejectedBlob.addMomentum(EJECTEDBLOB_BASE_MOMENTUM)
                self.addEjectedBlob(ejectedBlob)

    def handlePlayerCollisions(self):
        for player in self.players:
            self.handlePlayerCollision(player)

    def handlePlayerCollision(self, player):
        for cell in player.getCells():
            if cell.justEjected():
                continue
            for otherCell in player.getCells():
                if cell is otherCell or otherCell.justEjected() or cell.canMerge() and otherCell.canMerge():
                    continue
                distance = numpy.sqrt(cell.squaredDistance(otherCell))
                summedRadii = cell.getRadius() + otherCell.getRadius()
                if distance < summedRadii and distance != 0:
                    posDiff = cell.getPos() - otherCell.getPos()
                    scaling = (summedRadii - distance) / distance / 2
                    posDiffScaled = posDiff * scaling
                    self.adjustCellPos(cell, cell.getPos() + posDiffScaled, self.playerHashtable)
                    self.adjustCellPos(otherCell, otherCell.getPos() - posDiffScaled, self.playerHashtable)

    def updateHashTables(self):
        self.playerHashtable.clearBuckets()
        for player in self.players:
            playerCells = player.getCells()
            self.playerHashtable.insertAllObjects(playerCells)

    def mergePlayerCells(self):
        for player in self.players:
            cells = player.getMergableCells()
            if len(cells) > 1:
                cells.sort(key = lambda p: p.getMass(), reverse = True)
                for cell1 in cells:
                    if not cell1.isAlive():
                        continue
                    for cell2 in cells:
                        if (not cell2.isAlive()) or (cell2 is cell1):
                            continue
                        if cell1.overlap(cell2):
                            self.mergeCells(cell1, cell2)
                            if not cell1.isAlive():
                                break

    def checkOverlaps(self):
        self.playerVirusOverlap()
        self.playerPelletOverlap()
        self.playerPlayerOverlap()

    def playerPelletOverlap(self):
        for player in self.players:
            for cell in player.getCells():
                for pellet in self.pelletHashtable.getNearbyObjects(cell):
                    if cell.overlap(pellet):
                        self.eatPellet(cell, pellet)

    def playerVirusOverlap(self):
        for player in self.players:
            for cell in player.getCells():
                for virus in self.virusHashtable.getNearbyObjects(cell):
                    if cell.overlap(virus) and cell.getMass() > 1.5 * virus.getMass():
                        self.eatVirus(cell, virus)

    def playerPlayerOverlap(self):
        for player in self.players:
            for playerCell in player.getCells():
                opponentCells = self.playerHashtable.getNearbyEnemyObjects(playerCell)
                if self.debug:
                    if len(opponentCells) > 0:
                        print("\n_________")
                        print("Opponent cells of cell ", playerCell, ":")
                        for cell in opponentCells:
                            print(cell, end= " ")
                        print("\n____________\n")
                for opponentCell in opponentCells:
                        if playerCell.overlap(opponentCell):
                            if self.debug:
                                print(playerCell, " and ", opponentCell, " overlap!")
                            if playerCell.canEat(opponentCell):
                                self.eatPlayerCell(playerCell, opponentCell)
                            elif opponentCell.canEat(playerCell):
                                self.eatPlayerCell(opponentCell, playerCell)
                                break

    def virusEjectedOverlap(self):
        # After 7 feedings the virus splits in roughly the opposite direction of the last incoming ejectable
        # The ejected viruses bounce off of the edge of the fields
        pass

    def spawnStuff(self):
        self.spawnPellets()
        self.spawnViruses()
        self.spawnPlayers()

    def spawnViruses(self):
        while len(self.viruses) < self.maxVirusCount:
            self.spawnVirus()

    def spawnVirus(self):
        xPos = numpy.random.randint(0, self.width)
        yPos = numpy.random.randint(0, self.height)
        size = VIRUS_BASE_SIZE
        virus = Cell(xPos, yPos, size, None)
        virus.setName("Virus")
        virus.setColor((0,255,0))
        self.addVirus(virus)

    def spawnPlayers(self):
        for player in self.players:
            if len(player.getCells()) < 1:
                if self.debug:
                    print(player.getName(), " died!")
                self.initializePlayer(player)
                if self.debug:
                    print("REVIVE ", player.getName(), "!!!")

    def spawnPellets(self):
        while len(self.pellets) < self.maxCollectibleCount:
            self.spawnPellet()

    def spawnPellet(self):
        xPos = numpy.random.randint(0, self.width)
        yPos = numpy.random.randint(0, self.height)
        size = self.randomSize()
        pellet = Cell(xPos, yPos, size, None)
        pellet.setName("Pellet")
        self.addPellet(pellet)

    # Cell1 eats Cell2. Therefore Cell1 grows and Cell2 is deleted
    def eatPellet(self, cell, pellet):
        self.adjustCellSize(cell, pellet.getMass(), self.playerHashtable)
        self.pellets.remove(pellet)
        self.pelletHashtable.deleteObject(pellet)
        pellet.setAlive(False)

    def eatEjectedBlob(self, cell, ejectedBlob):
        self.adjustCellSize(cell, ejectedBlob.getMass(), self.playerHashtable)
        self.ejectedBlobs.remove(ejectedBlob)
        self.ejectedBlobHashTable.deleteObject(ejectedBlob)
        ejected.setAlive(False)

    def eatVirus(self, playerCell, virus):
        self.adjustCellSize(playerCell, virus.getMass(), self.playerHashtable)
        self.playerCellAteVirus(playerCell)
        self.viruses.remove(virus)
        self.virusHashtable.deleteObject(virus)
        virus.setAlive(False)

    def eatPlayerCell(self, largerCell, smallerCell):
        if self.debug:
            print(largerCell, " eats ", smallerCell, "!")
        self.adjustCellSize(largerCell, smallerCell.getMass(), self.playerHashtable)
        self.deletePlayerCell(smallerCell)

    def playerCellAteVirus(self, playerCell):
        player = playerCell.getPlayer()
        numberOfCells = len(player.getCells())
        numberOfNewCells = 16 - numberOfCells
        if numberOfNewCells == 0:
            return
        massPerCell = (playerCell.getMass() - 10) / numberOfNewCells
        playerCell.resetMergeTime(0.8)
        self.adjustCellSize(playerCell, -1 * massPerCell * numberOfNewCells, self.playerHashtable)
        for cellIdx in range(numberOfNewCells):
            cellPos = playerCell.getPos()
            newCell = Cell(cellPos[0], cellPos[1], massPerCell, player)
            cellAngle = (360 / numberOfNewCells) / (cellIdx + 1)

            xPoint = numpy.cos(cellAngle) * playerCell.getRadius() * 1.5 + cellPos[0]
            yPoint = numpy.sin(cellAngle) * playerCell.getRadius() * 1.5 + cellPos[1]
            newCell.setMoveDirection((xPoint, yPoint))
            newCell.addMomentum(MOMENTUM_BASE + 4 * playerCell.getRadius())
            newCell.resetMergeTime(0.8)
            self.addPlayerCell(newCell)



    def mergeCells(self, firstCell, secondCell):
        if firstCell.getMass() > secondCell.getMass():
            biggerCell = firstCell
            smallerCell = secondCell
        else:
            biggerCell = secondCell
            smallerCell = firstCell
        if self.debug:
            print(smallerCell, " is merged into ", biggerCell, "!")
        self.adjustCellSize(biggerCell, smallerCell.getMass(), self.playerHashtable)
        self.deletePlayerCell(smallerCell)

    def deletePlayerCell(self, playerCell):
        self.playerHashtable.deleteObject(playerCell)
        player = playerCell.getPlayer()
        player.removeCell(playerCell)

    def randomSize(self):
        maxRand = 20
        maxPelletSize = 5
        sizeRand = numpy.random.randint(0, maxRand)
        if sizeRand > (maxRand - maxPelletSize):
            return maxRand - sizeRand
        return 1

    def addPellet(self, pellet):
        self.pelletHashtable.insertObject(pellet)
        self.pellets.append(pellet)

    def addEjectedBlob(self, ejectedBlob):
        self.ejectedBlobHashTable.insertObject(ejectedBlob)
        self.ejectedBlobs.append(ejectedBlob)

    def addVirus(self, virus):
        self.virusHashtable.insertObject(virus)
        self.viruses.append(virus)

    def addPlayerCell(self, playerCell):
        self.playerHashtable.insertObject(playerCell)
        playerCell.getPlayer().addCell(playerCell)

    def adjustCellSize(self, cell, mass, hashtable):
        hashtable.deleteObject(cell)
        cell.grow(mass)
        hashtable.insertObject(cell)

    def adjustCellPos(self, cell, newPos, hashtable):
        #hashtable.deleteObject(cell)
        x = min(self.width, max(0, newPos[0]))
        y = min(self.height, max(0, newPos[1]))
        cell.setPos(x, y)
        #hashtable.insertObject(cell)

    # Setters:
    def setDebug(self, val):
        self.debug = val

    def addPlayer(self, player):
        player.setAlive()
        self.players.append(player)

    # Getters:
    def getPortionOfCellsInFov(self, cells, fovPos, fovDims):
        inFov = []
        for cell in cells:
            if cell.isInFov(fovPos, fovDims):
                inFov.append(cell)
        return inFov


    def getPlayerCellsInFov(self, fovPos, fovDims):

        cellsNearFov = self.getCellsFromHashtableInFov(self.playerHashtable, fovPos, fovDims)
        return self.getPortionOfCellsInFov(cellsNearFov, fovPos, fovDims)

    def getEnemyPlayerCellsInFov(self, fovPlayer):
        playerCellsInFov = self.getPlayerCellsInFov(fovPlayer.getFovPos(), fovPlayer.getFovDims())
        opponentCellsInFov = []
        for playerCell in playerCellsInFov:
            if playerCell.getPlayer() is not fovPlayer:
                opponentCellsInFov.append(playerCell)
        return opponentCellsInFov

    def getPelletsInFov(self, fovPos, fovDims):
        pelletsNearFov = self.getCellsFromHashtableInFov(self.pelletHashtable, fovPos, fovDims)
        return self.getPortionOfCellsInFov(pelletsNearFov, fovPos, fovDims)

    def getVirusesInFov(self, fovPos, fovDims):
        virusesNearFov = self.getCellsFromHashtableInFov(self.virusHashtable, fovPos, fovDims)
        return self.getPortionOfCellsInFov(virusesNearFov, fovPos, fovDims)

    def getCellsFromHashtableInFov(self, hashtable, fovPos, fovDims):
        fovCell = Cell(fovPos[0], fovPos[1], 1, None)
        fovCell.setRadius(fovDims[0] / 2)
        return hashtable.getNearbyObjects(fovCell)

    def getEjectedBlobsInFov(self, fovPos, fovDims):
        ejectedBlobsNearFov = self.getCellsFromHashtableInFov(self.ejectedBlobHashTable, fovPos, fovDims)
        return self.getPortionOfCellsInFov(pelletsNearFov, fovPos, fovDims)

    def getWidth(self):
        return self.width

    def getHeight(self):
        return self.height

    def getPellets(self):
        return self.pellets

    def getEjectedBlobs(self):
        return self.ejectedBlobs

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
