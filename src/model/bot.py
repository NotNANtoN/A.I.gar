import heapq
import numpy
import importlib.util
from .parameters import *
from .spatialHashTable import spatialHashTable

class Bot(object):
    num_NNbots = 0
    num_Greedybots = 0
    memories = []

    def __init__(self, player, field, type, trainMode, learningAlg):
        self.learningAlg = None
        self.trainMode = None
        if learningAlg is not None:
            self.learningAlg = learningAlg
            self.trainMode = trainMode
            # If just testing, set exploration to 0
            if trainMode == False:
                self.learningAlg.getNetwork().setEpsilon(0)
                self.learningAlg.getNetwork().setFrameSkipRate(0)

            self.expRepEnabled = self.learningAlg.getNetwork().getParameters().EXP_REPLAY_ENABLED
            self.gridViewEnabled = self.learningAlg.getNetwork().getParameters().GRID_VIEW_ENABLED

        self.type = type
        self.player = player
        self.field = field
        if self.type == "Greedy":
            self.splitLikelihood = numpy.random.randint(9950,10000)
            self.ejectLikelihood = 100000 #numpy.random.randint(9990,10000)
        self.totalMasses = []
        self.qValues = []
        self.reset()

    def reset(self):
        self.lastMass = None
        self.oldState = None
        self.latestTDerror = None
        self.lastMemory = None
        self.skipFrames = 0
        self.cumulativeReward = 0
        self.lastReward = 0
        self.rewardAvgOfEpisode = 0
        self.rewardLenOfEpisode = 0
        if self.type == "NN":
            self.currentActionIdx = None
            self.currentAction = None
        elif self.type == "Greedy":
            self.currentAction = [0, 0, 0, 0]

    def updateRewards(self):
        self.cumulativeReward += self.getReward() if self.lastMass else 0
        self.lastReward = self.cumulativeReward

        if self.player.getIsAlive():
            self.rewardAvgOfEpisode = (self.rewardAvgOfEpisode * self.rewardLenOfEpisode + self.lastReward) \
                                     / (self.rewardLenOfEpisode + 1)
            self.rewardLenOfEpisode += 1

    def updateFrameSkip(self):
        # Do not train if we are skipping this frame
        if self.skipFrames > 0:
            self.skipFrames -= 1
            self.currentAction[2:4] = [0, 0]
            self.latestTDerror = None
            if self.player.getIsAlive():
                return True
        return False

    def makeMove(self):
        self.totalMasses.append(self.player.getTotalMass())
        if self.type == "NN":
            newState = self.getStateRepresentation()
            if self.trainMode:
                currentlySkipping = False
                if self.currentAction is not None:
                    self.updateRewards()
                    currentlySkipping = self.updateFrameSkip()
                if not currentlySkipping:
                    td_e, qva, nlm, caIdx, ca, cr = self.learningAlg.learn(self, newState)
                    self.latestTDerror = td_e
                    self.qValues.append(qva)
                    if nlm is not None:
                        self.lastMemory = nlm
                    self.currentActionIdx = caIdx
                    self.currentAction = ca
                    self.cumulativeReward = cr
                    self.skipFrames = self.learningAlg.getNetwork().getFrameSkipRate()

                if self.player.getIsAlive():
                    self.lastMass = self.player.getTotalMass()
                    self.oldState = newState
                else:
                    print(self.player, " died.")
                    print("Average reward of ", self.player, " for this episode: ", self.rewardAvgOfEpisode)
                    self.reset()

            else:
                # If simply testing, no training required, only action decision
                if self.player.getIsAlive():
                    caIdx, ca = self.learningAlg.decideMove(newState, self.player)
                    self.currentActionIdx = caIdx
                    self.currentAction = ca
        elif self.type == "Greedy":
            if not self.player.getIsAlive():
                return
            midPoint = self.player.getFovPos()
            size = self.player.getFovSize()
            x = int(midPoint[0])
            y = int(midPoint[1])
            left = x - int(size / 2)
            top = y - int(size / 2)

            cellsInFov = self.field.getPelletsInFov(midPoint, size)

            playerCellsInFov = self.field.getEnemyPlayerCellsInFov(self.player)
            firstPlayerCell = self.player.getCells()[0]
            for opponentCell in playerCellsInFov:
                # If the single celled bot can eat the opponent cell add it to list
                if firstPlayerCell.getMass() > 1.25 * opponentCell.getMass():
                    cellsInFov.append(opponentCell)
            if cellsInFov:
                bestCell = max(cellsInFov, key = lambda p: p.getMass() / (p.squaredDistance(firstPlayerCell) if p.squaredDistance(firstPlayerCell) != 0 else 1))
                bestCellPos = self.getRelativeCellPos(bestCell, left, top, size)
                self.currentAction[0] = bestCellPos[0]
                self.currentAction[1] = bestCellPos[1]
            else:
                size = int(size / 2)
                self.currentAction[0] = numpy.random.random()
                self.currentAction[1] = numpy.random.random()
            randNumSplit = numpy.random.randint(0,10000)
            randNumEject = numpy.random.randint(0,10000)
            self.currentAction[2] = False
            self.currentAction[3] = False
            if randNumSplit > self.splitLikelihood:
                self.currentAction[2] = True
            if randNumEject > self.ejectLikelihood:
                self.currentAction[3] = True

        if self.player.getIsAlive():
            midPoint = self.player.getFovPos()
            size = self.player.getFovSize()
            x = int(midPoint[0])
            y = int(midPoint[1])
            left = x - int(size / 2)
            top = y - int(size / 2)
            size = int(size)
            xChoice = left + self.currentAction[0] * size
            yChoice = top + self.currentAction[1] * size
            splitChoice = True if self.currentAction[2] > 0.5 else False
            ejectChoice = True if self.currentAction[3] > 0.5 else False

            self.player.setCommands(xChoice, yChoice, splitChoice, ejectChoice)

    def getStateRepresentation(self):
        stateRepr = None
        if self.player.getIsAlive():
            if self.gridViewEnabled:
                stateRepr =  self.getGridStateRepresentation()
            else:
                stateRepr =  self.getSimpleStateRepresentation()
        return stateRepr

    def getSimpleStateRepresentation(self):
        # Get data about the field of view of the player
        size = self.player.getFovSize()
        midPoint = self.player.getFovPos()
        x = int(midPoint[0])
        y = int(midPoint[1])
        left = x - int(size / 2)
        top = y - int(size / 2)
        size = int(size)
        # At the moment we only care about the first cell of the current player, to be extended once we get this working
        firstPlayerCell = self.player.getCells()[0]

        # Adding all the state data to totalInfo
        totalInfo = []
        # Add data about player cells
        cellInfos = self.getCellDataOwnPlayer(left, top, size)
        for info in cellInfos:
            totalInfo += info
        # Add data about the closest enemy cell
        playerCellsInFov = self.field.getEnemyPlayerCellsInFov(self.player)
        closestEnemyCell = min(playerCellsInFov,
                               key=lambda p: p.squaredDistance(firstPlayerCell)) if playerCellsInFov else None
        totalInfo += self.isRelativeCellData(closestEnemyCell, left, top, size)
        # Add data about the closest pellet
        pelletsInFov = self.field.getPelletsInFov(midPoint, size)
        closestPellet = min(pelletsInFov, key=lambda p: p.squaredDistance(firstPlayerCell)) if pelletsInFov else None
        closestPelletPos = self.getRelativeCellPos(closestPellet, left, top, size)
        totalInfo += closestPelletPos
        # Add data about distances to the visible edges of the field
        width = self.field.getWidth()
        height = self.field.getHeight()
        distLeft = x / size if left <= 0 else 1
        distRight = (width - x) / size if left + size >= width else 1
        distTop = y / size if top <= 0 else 1
        distBottom = (height - y) / size if top + size >= height else 1
        totalInfo += [distLeft, distRight, distTop, distBottom]
        return totalInfo

    def getGridStateRepresentation(self):
        # Get Fov infomation
        fieldSize = self.field.getWidth()
        fovSize = self.player.getFovSize()
        fovPos = self.player.getFovPos()
        x = fovPos[0]
        y = fovPos[1]
        left = x - fovSize / 2
        top = y - fovSize / 2
        # Initialize spatial hash tables:
        gridSquaresPerFov = self.learningAlg.getNetwork().getGridSquaresPerFov()
        gsSize = fovSize / gridSquaresPerFov  # (gs = grid square)
        pelletSHT = spatialHashTable(fovSize, gsSize, left, top) #SHT = spatial hash table
        enemySHT =  spatialHashTable(fovSize, gsSize, left, top)
        virusSHT =  spatialHashTable(fovSize, gsSize, left, top)
        playerSHT = spatialHashTable(fovSize, gsSize, left, top)
        totalPellets = self.field.getPelletsInFov(fovPos, fovSize)
        pelletSHT.insertAllFloatingPointObjects(totalPellets)
        if __debug__ and self.player.getSelected():
            print("Total pellets: ",len(totalPellets))
            print("pellet view: ")
            buckets = pelletSHT.getBuckets()
            for idx in range(len(buckets)):
                print(len(buckets[idx]), end = " ")
                if idx != 0 and (idx + 1) % gridSquaresPerFov == 0:
                    print(" ")
        playerCells = self.field.getPortionOfCellsInFov(self.player.getCells(), fovPos, fovSize)
        playerSHT.insertAllFloatingPointObjects(playerCells)
        enemyCells = self.field.getEnemyPlayerCellsInFov(self.player)
        enemySHT.insertAllFloatingPointObjects(enemyCells)
        virusCells = self.field.getVirusesInFov(fovPos, fovSize)
        virusSHT.insertAllFloatingPointObjects(virusCells)

        # Mass vision grid related
        #enemyCellsCount = len(enemyCells)
        #allCellsInFov = playerCells + enemyCells + virusCells
        #biggestMassInFov = max(allCellsInFov, key = lambda cell: cell.getMass()).getMass() if allCellsInFov else None

        # Initialize grid squares with zeros:
        gridNumberSquared = gridSquaresPerFov * gridSquaresPerFov
        gsBiggestEnemyCellMassProportion = numpy.zeros(gridNumberSquared)
        gsBiggestOwnCellMassProportion = numpy.zeros(gridNumberSquared)
        gsWalls = numpy.zeros(gridNumberSquared)
        gsVirus = numpy.zeros(gridNumberSquared)
        gsPelletProportion = numpy.zeros(gridNumberSquared)
        # gsMidPoint is adjusted in the loops
        gsMidPoint = [left + gsSize / 2, top + gsSize/ 2]
        pelletcount = 0
        for c in range(gridSquaresPerFov):
            for r in range(gridSquaresPerFov):
                count = r + c * gridSquaresPerFov

                # Only check for cells if the grid square fov is within the playing field
                if not(gsMidPoint[0] + gsSize / 2 < 0 or gsMidPoint[0] - gsSize / 2 > fieldSize or
                        gsMidPoint[1] + gsSize / 2 < 0 or gsMidPoint[1] - gsSize / 2 > fieldSize):
                    # Create pellet representation
                    # Make the visionGrid's pellet count a percentage so that the network doesn't have to
                    # work on interpreting the number of pellets relative to the size (and Fov) of the player
                    pelletMassSum = 0
                    pelletsInGS = pelletSHT.getBucketContent(count)
                    if pelletsInGS:
                        for pellet in pelletsInGS:
                            pelletcount += 1
                            pelletMassSum += pellet.getMass()
                        gsPelletProportion[count] = pelletMassSum

                    # TODO: add relative fov pos of closest pellet to allow micro management

                    # Create Enemy Cell mass representation
                    # Make the visionGrid's enemy cell representation a percentage. The player's mass
                    # in proportion to the biggest enemy cell's mass in each grid square.
                    enemiesInGS = enemySHT.getBucketContent(count)
                    if enemiesInGS:
                        biggestEnemyInCell = max(enemiesInGS, key = lambda cell: cell.getMass())
                        gsBiggestEnemyCellMassProportion[count] = biggestEnemyInCell.getMass()

                    # Create Own Cell mass representation
                    playerCellsInGS = playerSHT.getBucketContent(count)
                    if playerCellsInGS:
                        biggestFriendInCell = max(playerCellsInGS, key=lambda cell: cell.getMass())
                        gsBiggestOwnCellMassProportion[count] = biggestFriendInCell.getMass()
                    # TODO: also add a count grid for own cells?

                    # Create Virus Cell representation
                    if self.field.getVirusEnabled():
                        virusesInGS = virusSHT.getBucketContent(count)
                        if virusesInGS:
                            biggestVirus = max(virusesInGS, key = lambda virus: virus.getRadius()).getMass()
                            gsVirus[count] = biggestVirus

                # Create Wall representation
                # 1s indicate a wall present in the grid square (regardless of amount of wall in square), else 0
                if gsMidPoint[0] - gsSize/2 < 0 or gsMidPoint[0] + gsSize/2 > fieldSize or \
                        gsMidPoint[1] - gsSize/2 < 0 or gsMidPoint[1] + gsSize/2 > fieldSize:
                    gsWalls[count] = 1
                # Increment grid square position horizontally
                gsMidPoint[0] += gsSize
            # Reset horizontal grid square, increment grid square position
            gsMidPoint[0] = left + gsSize / 2
            gsMidPoint[1] += gsSize
        if __debug__ and self.player.getSelected():
            print("counted pellets: ", pelletcount)
            print(" ")
        # Concatenate the basis data about the location of pellets, own cells and the walls:
        totalInfo = numpy.concatenate((gsPelletProportion, gsBiggestOwnCellMassProportion, gsWalls,
                                       gsBiggestEnemyCellMassProportion, gsVirus))
        # Add total Mass of player and field size:
        totalMass = self.player.getTotalMass()
        totalInfo = numpy.concatenate((totalInfo, [totalMass, fovSize]))

        return totalInfo

    def isCellData(self, cell):
        return [cell.getX(), cell.getY(), cell.getRadius()]

    def isRelativeCellData(self, cell, left, top, size):
        return self.getRelativeCellPos(cell, left, top, size) + \
               ([round(cell.getRadius() / size if cell.getRadius() <= size else 1, 5)] if cell != None else [0])

    def getQValues(self):
        return self.qValues

    def getMassOverTime(self):
        return self.totalMasses

    def getAvgReward(self):
        return self.rewardAvgOfEpisode

    def getLastReward(self):
        return self.lastReward

    def getRelativeCellPos(self, cell, left, top, size):
        if cell != None:
            return [round((cell.getX() - left) / size, 5), round((cell.getY() - top) / size, 5)]
        else:
            return [0, 0]

    def checkNan(self, value):
        if math.isnan(value):
            print("ERROR: predicted reward is nan")
            quit()

    def getCellDataOwnPlayer(self, left, top, size):
        cells = self.player.getCells()
        totalCells = len(cells)
        return [self.isRelativeCellData(cells[idx], left, top, size) if idx < totalCells else [0, 0, 0]
                     for idx in range(1)]

    def getTDError(self):
        return self.latestTDerror

    def getReward(self):
        if self.lastMass is None:
            return None
        if not self.player.getIsAlive():
            return -1 * self.lastMass
        currentMass = self.player.getTotalMass()
        reward = currentMass - self.lastMass
        #if abs(reward) < 0.1:
        #    reward -=  1
        return reward

    def getType(self):
        return self.type

    def getPlayer(self):
        return self.player

    def getTrainMode(self):
        return self.trainMode

    def getLearningAlg(self):
        return self.learningAlg

    def getOldState(self):
        return self.oldState

    def getCurrentActionIdx(self):
        return self.currentActionIdx

    def getCurrentAction(self):
        return self.currentAction

    def getCumulativeReward(self):
        return self.cumulativeReward

    def getMemories(self):
        return self.memories

    def getLastMemory(self):
        return self.lastMemory

    def getExpRepEnabled(self):
        return self.expRepEnabled