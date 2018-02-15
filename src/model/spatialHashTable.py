import numpy

class spatialHashTable(object):
    def __repr__(self):
        name = "Hash table: \n"
        name += "Rows: " + str(self.rows) + " Cols: " + str(self.cols) + " Cellsize: " + str(self.cellSize) + "\n"
        #name += "Hash table content: \n"
        total = 0
        for bucketId, content in self.buckets.items():
        #    name += "Bucket: " + str(bucketId) + " contains " + str(len(content)) + " items." + "\n"
            total += len(content)
        name += "Total objects in table: " + str(total)
        return name

    def __init__(self, width, height, cellSize):
        self.width = width
        self.height = height
        self.rows = int(numpy.ceil(height / cellSize))
        self.cols = int(numpy.ceil(width / cellSize))
        self.cellSize = cellSize
        self.buckets = {}
        self.clearBuckets()

    def getNearbyObjects(self, obj):
        cellIds = self.getIdsForObj(obj)
        return self.getObjectsFromBuckets(cellIds)

    def getNearbyObjectsInArea(self, pos, rad):
        cellIds = self.getIdsForArea(pos, rad)
        return self.getObjectsFromBuckets(cellIds)


    def getNearbyEnemyObjects(self, obj):
        cellIds = self.getIdsForObj(obj)
        nearbyObjects = self.getObjectsFromBuckets(cellIds)
        return [nearbyObject for nearbyObject in nearbyObjects if nearbyObject.getPlayer() is not obj.getPlayer()]


    def getObjectsFromBuckets(self, cellIds):
        nearbyObjects = set()
        for cellId in cellIds:
            for cell in self.buckets[cellId]:
                nearbyObjects.add(cell)
        return nearbyObjects

    def clearBuckets(self):
        for i in range(self.cols * self.rows):
            self.buckets[i] = []

    def insertObject(self, obj):
        cellIds = self.getIdsForObj(obj)
        for id in cellIds:
            self.buckets[id].append(obj)

    def insertAllObjects(self, objects):
        for obj in objects:
            self.insertObject(obj)

    # Deletes an object out of all the buckets it is in. Might not be needed as it might
    # be faster to clear all buckets and reinsert items than updating objects.
    def deleteObject(self, obj):
        cellIds = self.getIdsForObj(obj)
        for id in cellIds:
            self.buckets[id].remove(obj)

    def getIdsForObj(self, obj):
        pos = obj.getPos()
        radius = obj.getRadius()
        return self.getIdsForArea(pos, radius)

    def getIdsForArea(self, pos, radius):
        ids = set()
        topLeft = (max(0, pos[0] - radius), max(0, pos[1] - radius))
        limitX = radius + (min(radius, pos[0]))
        limitY = radius + (min(radius, pos[1]))
        #limitX = radius + ((radius))
        #limitY = radius + ((radius))
        stepSizeX = min(limitX, self.cellSize)
        stepSizeY = min(limitY, self.cellSize)
        i = 0
        #hashFunc = self.getHashId
        while i <= limitX:
            j = 0
            while j <= limitY:
                #x = i + topLeft[0]
                #y = j + topLeft[1]
                x = min(self.width - 1 , i + topLeft[0])
                y = min(self.height - 1, j + topLeft[1])
                hashId = self.getHashId((x, y))
                ids.add(hashId)
                j += stepSizeY
            i += stepSizeX
        return ids

    def getHashId(self, pos):
        return int(numpy.floor(pos[0] / self.cellSize) + numpy.floor(pos[1] / self.cellSize) * self.cols)
