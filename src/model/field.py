import numpy
import math
import time
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
        self.blobs = [] # Ejected particles become pellets once momentum is lost
        self.deadPlayers = []
        self.viruses = []
        self.maxCollectibleCount = None
        self.maxVirusCount = None
        self.pelletHashTable = None
        self.blobHashTable = None
        self.playerHashTable = None
        self.virusHashTable = None

    def initializePlayer(self, player):
        x, y = self.getSpawnPos()
        newCell = Cell(x, y, START_MASS, player)
        player.addCell(newCell)
        player.setAlive()


    def initialize(self):
        self.width = int(SIZE_INCREASE_PER_PLAYER * math.sqrt(len(self.players)))
        self.height = int(SIZE_INCREASE_PER_PLAYER * math.sqrt(len(self.players)))
        self.pelletHashTable = spatialHashTable(self.width, self.height, HASH_BUCKET_SIZE)
        self.blobHashTable = spatialHashTable(self.width, self.height, HASH_BUCKET_SIZE)
        self.playerHashTable = spatialHashTable(self.width, self.height, HASH_BUCKET_SIZE)
        self.virusHashTable = spatialHashTable(self.width, self.height, HASH_BUCKET_SIZE)
        for player in self.players:
            self.initializePlayer(player)
        self.maxCollectibleCount = self.width * self.height * MAX_COLLECTIBLE_DENSITY
        self.maxVirusCount = self.width * self.height * MAX_VIRUS_DENSITY
        self.spawnStuff()

    def update(self):
        self.updateViruses()
        self.updateBlobs()
        self.updatePlayers()
        self.updateHashTables()
        self.mergePlayerCells()
        self.checkOverlaps()
        self.spawnStuff()

        '''
        if __debug__:
            print(" ")
            fovPos = self.players[0].getFovPos()
            fovDims = self.players[0].getFovDims()
            human = self.players[0]
            humanCell = human.cells[0]

            print("main function:", self.getPlayerCellsInFov(fovPos, fovDims))
            print("cellsNearFov: ", self.getCellsFromHashTableInFov(self.playerHashTable, fovPos, fovDims))
            print("hashtable.getNearbyObjectsInArea: ", self.playerHashTable.getNearbyObjectsInArea(fovPos, fovDims[0] / 2) )
            print("")
            print("fovpos, fovdims: ", human.getFovPos(), human.getFovDims())
            print("human ids for obj:", self.playerHashTable.getIdsForObj(humanCell))
            print("human ids for area:", self.playerHashTable.getIdsForArea(humanCell.getPos(),humanCell.getRadius() ))
            print("meeep: ", self.playerHashTable.getIdsForArea(fovPos, fovDims[0] / 2))
            print("radius: ", numpy.round(humanCell.getRadius(), 2))
            print(" ")
'''
    def updateViruses(self):
        for virus in self.viruses:
            virus.updateMomentum()
            virus.updatePos(self.width, self.height)

    def updateBlobs(self):
        notMovingBlobs = []
        for blob in self.blobs:
            if blob.getSplitVelocityCounter() == 0:
                notMovingBlobs.append(blob)
                continue
            blob.updateMomentum()
            blob.updatePos(self.width, self.height)
        for blob in notMovingBlobs:  
            self.blobs.remove(blob)
            self.blobHashTable.deleteObject(blob)
            self.addPellet(blob)

    def updatePlayers(self):
        for player in self.players:
            player.update(self.width, self.height)
            self.performEjections(player)
        self.handlePlayerCollisions()

    def updateHashTables(self):
        self.playerHashTable.clearBuckets()
        for player in self.players:
            playerCells = player.getCells()
            self.playerHashTable.insertAllObjects(playerCells)

        self.blobHashTable.clearBuckets()
        self.blobHashTable.insertAllObjects(self.blobs)

        self.virusHashTable.clearBuckets()
        self.virusHashTable.insertAllObjects(self.viruses)

    def performEjections(self, player):
        for cell in player.getCells():
            if cell.getBlobToBeEjected():
                blobSpawnPos = cell.eject(player.getCommandPoint())
                # Blobs are given a player such that cells of player who eject them don't instantly reabsorb them
                blob = Cell(blobSpawnPos[0], blobSpawnPos[1], EJECTEDBLOB_BASE_MASS * 0.8, None)

                blob.setColor(player.getColor())
                #blob.setEjecterPlayer(player)
                blob.addMomentum(player.getCommandPoint(), self.width, self.height, cell)

                self.addBlob(blob)
                blob.setEjecterCell(cell)


    def handlePlayerCollisions(self):
        for player in self.players:
            for cell in player.getCells():
                if cell.justEjected():
                    continue
                for otherCell in player.getCells():
                    if cell is otherCell or otherCell.justEjected() or (cell.canMerge() and otherCell.canMerge()):
                        continue
                    distance = numpy.sqrt(cell.squaredDistance(otherCell))
                    summedRadii = cell.getRadius() + otherCell.getRadius()
                    if distance < summedRadii and distance != 0:
                        self.adjustCellPositions(cell, otherCell, distance, summedRadii)

    def adjustCellPositions(self, cell1, cell2, distance, summedRadii):
        if cell1.getMass() > cell2.getMass():
            biggerCell = cell1
            smallerCell = cell2
        else:
            biggerCell = cell2
            smallerCell = cell1
        posDiff = biggerCell.getPos() - smallerCell.getPos()
        scaling = (summedRadii - distance) / distance
        posDiffScaled = posDiff * scaling
        massDifferenceScaling = smallerCell.getMass() /  biggerCell.getMass()
        biggerCellMoveTo =  biggerCell.getPos() + posDiffScaled * (massDifferenceScaling)
        smallerCellMoveTo = smallerCell.getPos() - posDiffScaled * (1 - massDifferenceScaling)
        self.adjustCellPos(biggerCell, biggerCellMoveTo, self.playerHashTable)
        self.adjustCellPos(smallerCell, smallerCellMoveTo, self.playerHashTable)

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
        self.virusBlobOverlap()
        self.playerVirusOverlap()
        self.playerPelletOverlap()
        self.playerBlobOverlap()
        self.playerPlayerOverlap()

    def playerPelletOverlap(self):
        for player in self.players:
            for cell in player.getCells():
                for pellet in self.pelletHashTable.getNearbyObjects(cell):
                    if cell.overlap(pellet):
                        self.eatPellet(cell, pellet)

    def playerBlobOverlap(self):
        for player in self.players:
            for cell in player.getCells():
                for blob in self.blobHashTable.getNearbyObjects(cell):
                    # If the ejecter player's cell is not the one overlapping with blob
                    if cell.overlap(blob) and blob.getEjecterCell() is not cell:
                        self.eatBlob(cell, blob)


    def playerVirusOverlap(self):
        for player in self.players:
            for cell in player.getCells():
                for virus in self.virusHashTable.getNearbyObjects(cell):
                    if cell.overlap(virus) and cell.getMass() > 1.5 * virus.getMass():
                        self.eatVirus(cell, virus)

    def playerPlayerOverlap(self):
        for player in self.players:
            for playerCell in player.getCells():
                opponentCells = self.playerHashTable.getNearbyEnemyObjects(playerCell)
                if __debug__:
                    if opponentCells:
                        print("\n_________")
                        print("Opponent cells of cell ", playerCell, ":")
                        for cell in opponentCells:
                            print(cell, end= " ")
                        print("\n____________\n")
                for opponentCell in opponentCells:
                        if playerCell.overlap(opponentCell):
                            if __debug__:
                                print(playerCell, " and ", opponentCell, " overlap!")
                            if playerCell.canEat(opponentCell):
                                self.eatPlayerCell(playerCell, opponentCell)
                            elif opponentCell.canEat(playerCell):
                                self.eatPlayerCell(opponentCell, playerCell)
                                break

    def virusBlobOverlap(self):
        # After 7 feedings the virus splits in roughly the opposite direction of the last incoming ejectable
        # The ejected viruses bounce off of the edge of the fields
        for virus in self.viruses:
            nearbyBlobs = self.blobHashTable.getNearbyObjects(virus)
            for blob in nearbyBlobs:
                if virus.overlap(blob):
                    self.virusEatBlob(virus, blob)


    def spawnStuff(self):
        self.spawnPellets()
        self.spawnViruses()
        self.spawnPlayers()

    def spawnViruses(self):
        while len(self.viruses) < self.maxVirusCount:
            self.spawnVirus()

    def spawnVirus(self):
        xPos, yPos = self.getSpawnPos()
        size = VIRUS_BASE_SIZE
        virus = Cell(xPos, yPos, size, None)
        virus.setName("Virus")
        virus.setColor((0,255,0))
        self.addVirus(virus)

    def spawnPlayers(self):
        for player in self.players:
            if not player.getCells():
                self.initializePlayer(player)

    def getSpawnPos(self):
        cols = self.playerHashTable.getCols()
        totalBuckets = self.playerHashTable.getRows() * cols
        spawnBucket = numpy.random.randint(0, totalBuckets)
        count = 0
        while self.playerHashTable.getBuckets()[spawnBucket] and count < totalBuckets:
            spawnBucket = (spawnBucket + 1) % totalBuckets
            count += 1
        if count == totalBuckets:
            xPos = numpy.random.randint(0, self.width)
            yPos = numpy.random.randint(0, self.height)
        else:
            x = spawnBucket % cols
            y = (spawnBucket - x) / cols
            xPos = (x - 0.5) * HASH_BUCKET_SIZE
            yPos = (y + 0.5) * HASH_BUCKET_SIZE
        return xPos, yPos

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
    def virusEatBlob(self, virus, blob):
        self.eatCell(virus, self.virusHashTable, blob, self.blobHashTable, self.blobs)
        if virus.getMass() >= VIRUS_BASE_SIZE + 7 * EJECTEDBLOB_BASE_MASS * 0.8:
            oppositePoint = 2 * virus.getPos() - blob.getPos()
            newVirus = virus.split(oppositePoint, self.width, self.height)
            newVirus.setColor(virus.getColor())
            newVirus.setName(virus.getName())
            self.addVirus(newVirus)

    def eatPellet(self, playerCell, pellet):
        self.eatCell(playerCell, self.playerHashTable, pellet, self.pelletHashTable, self.pellets)

    def eatBlob(self, playerCell, blob):
        self.eatCell(playerCell, self.playerHashTable, blob, self.blobHashTable, self.blobs)

    def eatVirus(self, playerCell, virus):
        self.eatCell(playerCell, self.playerHashTable, virus, self.virusHashTable, self.viruses)
        self.playerCellAteVirus(playerCell)

    def eatCell(self, eatingCell, eatingCellHashtable, cell, cellHashtable, list):
        self.adjustCellSize(eatingCell, cell.getMass(), eatingCellHashtable)
        list.remove(cell)
        cellHashtable.deleteObject(cell)
        cell.setAlive(False)

    def eatPlayerCell(self, largerCell, smallerCell):
        if __debug__:
            print(largerCell, " eats ", smallerCell, "!")
        self.adjustCellSize(largerCell, smallerCell.getMass(), self.playerHashTable)
        self.deletePlayerCell(smallerCell)

    def playerCellAteVirus(self, playerCell):
        player = playerCell.getPlayer()
        numberOfCells = len(player.getCells())
        numberOfNewCells = 16 - numberOfCells
        if numberOfNewCells == 0:
            return
        #massPerCell = (playerCell.getMass() * 0.9) / numberOfNewCells
        massPerCell = VIRUS_EXPLOSION_BASE_MASS + (playerCell.getMass() * 0.1 / numberOfNewCells)
        playerCell.resetMergeTime(0.8)
        self.adjustCellSize(playerCell, -1 * massPerCell * numberOfNewCells, self.playerHashTable)
        for cellIdx in range(numberOfNewCells):
            cellPos = playerCell.getPos()
            newCell = Cell(cellPos[0], cellPos[1], massPerCell, player)
            #cellAngle = (360 / numberOfNewCells) * (cellIdx + 1)
            cellAngle = numpy.deg2rad(numpy.random.randint(0,360))
            xPoint = numpy.cos(cellAngle) * playerCell.getRadius() * 12 + cellPos[0]
            yPoint = numpy.sin(cellAngle) * playerCell.getRadius() * 12 + cellPos[1]
            movePoint = (xPoint, yPoint)
            newCell.setMoveDirection(movePoint)
            newCell.addMomentum(movePoint, self.width, self.height, playerCell)
            newCell.resetMergeTime(0.8)
            self.addPlayerCell(newCell)



    def mergeCells(self, firstCell, secondCell):
        if firstCell.getMass() > secondCell.getMass():
            biggerCell = firstCell
            smallerCell = secondCell
        else:
            biggerCell = secondCell
            smallerCell = firstCell
        if __debug__:
            print(smallerCell, " is merged into ", biggerCell, "!")
        self.adjustCellSize(biggerCell, smallerCell.getMass(), self.playerHashTable)
        self.deletePlayerCell(smallerCell)

    def deletePlayerCell(self, playerCell):
        self.playerHashTable.deleteObject(playerCell)
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
        self.pelletHashTable.insertObject(pellet)
        self.pellets.append(pellet)

    def addBlob(self, blob):
        #self.blobHashTable.insertObject(blob)
        self.blobs.append(blob)

    def addVirus(self, virus):
        #self.virusHashTable.insertObject(virus)
        self.viruses.append(virus)

    def addPlayerCell(self, playerCell):
        self.playerHashTable.insertObject(playerCell)
        playerCell.getPlayer().addCell(playerCell)

    def adjustCellSize(self, cell, mass, hashtable):
        hashtable.deleteObject(cell)
        cell.grow(mass)
        hashtable.insertObject(cell)

    def adjustCellPos(self, cell, newPos, hashtable):
        #hashtable.deleteObject(cell)
        x = min(self.width, max(0, newPos[0]))
        y = min(self.height, max(0, newPos[1]))
        cell.setPos(numpy.array([x,y]))
        #hashtable.insertObject(cell)

    # Setters:
    def addPlayer(self, player):
        player.setAlive()
        self.players.append(player)

    # Getters:
    def getPortionOfCellsInFov(self, cells, fovPos, fovDims):
        return [cell for cell in cells if cell.isInFov(fovPos,fovDims)]

    def getPlayerCellsInFov(self, fovPos, fovDims):
        cellsNearFov = self.getCellsFromHashTableInFov(self.playerHashTable, fovPos, fovDims)
        return self.getPortionOfCellsInFov(cellsNearFov, fovPos, fovDims)

    def getEnemyPlayerCellsInFov(self, fovPlayer):
        playerCellsInFov = self.getPlayerCellsInFov(fovPlayer.getFovPos(), fovPlayer.getFovDims())
        return [cell for cell in playerCellsInFov if cell.getPlayer() is not fovPlayer]

    def getPelletsInFov(self, fovPos, fovDims):
        pelletsNearFov = self.getCellsFromHashTableInFov(self.pelletHashTable, fovPos, fovDims)
        return self.getPortionOfCellsInFov(pelletsNearFov, fovPos, fovDims)

    def getVirusesInFov(self, fovPos, fovDims):
        virusesNearFov = self.getCellsFromHashTableInFov(self.virusHashTable, fovPos, fovDims)
        return self.getPortionOfCellsInFov(virusesNearFov, fovPos, fovDims)
    
    def getBlobsInFov(self, fovPos, fovDims):
        blobsNearFov = self.getCellsFromHashTableInFov(self.blobHashTable, fovPos, fovDims)
        return self.getPortionOfCellsInFov(blobsNearFov, fovPos, fovDims)

    def getCellsFromHashTableInFov(self, hashtable, fovPos, fovDims):
        return hashtable.getNearbyObjectsInArea(fovPos, fovDims[0] / 2)
    
    def getWidth(self):
        return self.width

    def getHeight(self):
        return self.height

    def getPellets(self):
        return self.pellets

    def getBlobs(self):
        return self.blobs

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
