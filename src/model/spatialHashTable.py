import numpy

class spatialHashTable(object):
    def __repr__(self):
        name = "Hash table: \n"
        name += "Rows: " + str(self.rows) + " Cols: " + str(self.cols) + " Cellsize: " + str(self.bucketSize) + "\n"
        #name += "Hash table content: \n"
        total = 0
        for bucketId, content in self.buckets.items():
        #    name += "Bucket: " + str(bucketId) + " contains " + str(len(content)) + " items." + "\n"
            total += len(content)
        name += "Total objects in table: " + str(total)
        return name

    def __init__(self, width, height, bucketSize):
        self.width = width
        self.height = height
        self.rows = int(numpy.ceil(height / bucketSize))
        self.cols = int(numpy.ceil(width / bucketSize))
        self.bucketSize = bucketSize
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
        hashFunc = self.getHashId
        topLeft = (max(0, pos[0] - radius), max(0, pos[1] - radius))
        bucketTopLeft = (topLeft[0] - topLeft[0] % self.bucketSize, topLeft[1] - topLeft[1] % self.bucketSize)
        stepSize = self.bucketSize
        limitX = min(self.width - 1, pos[0] + radius)
        limitY = min(self.height - 1, pos[1] + radius)

        x = bucketTopLeft[0]
        while x <= limitX:
            y = bucketTopLeft[1]
            while y <= limitY:
                ids.add(hashFunc(x, y))
                y += stepSize
            x += stepSize
        return ids


    def getHashId(self, x, y):
        return int(x / self.bucketSize) + int(y / self.bucketSize) * self.cols

    def getCols(self):
        return self.cols

    def getRows(self):
        return self.rows

    def getBuckets(self):
        return self.buckets
