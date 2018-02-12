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
        self.rows = int(height / cellSize + height % cellSize)
        self.cols = int(width / cellSize + width % cellSize)
        self.cellSize = cellSize
        self.buckets = {}
        self.clearBuckets()


    def getNearbyObjects(self, obj):
        cellIds = self.getIdsForObj(obj)
        nearbyObjects = set()
        for cellId in cellIds:
            for cell in self.buckets[cellId]:
                nearbyObjects.add(cell)
        return nearbyObjects

    def getNearbyEnemyObjects(self, obj):
        cellIds = self.getIdsForObj(obj)
        nearbyObjects = set()
        for cellId in cellIds:
            for cell in self.buckets[cellId]:
                if cell.getPlayer() != obj.getPlayer():
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
        #print("")
        #print("________")
        #print("Deleted from hashtable: ", obj)
        cellIds = self.getIdsForObj(obj)
        #print("ids for obj: ")
        for id in cellIds:
            #print("in Bucket ", id, ": ")
            #for bucket in self.buckets[id]:
            #    print(str(bucket))
            #while obj in self.buckets[id]:
            self.buckets[id].remove(obj)
        #print("________")
        #print("")




    def getIdsForObj(self, obj):
        ids = set()
        pos = obj.getPos()
        radius = obj.getRadius()
        topLeft = (max(0, pos[0] - radius), max(0, pos[1] - radius))
        cellWidth = obj.getRadius() * 2
        stepSize = min(cellWidth, self.cellSize)
        limit = cellWidth + 1
        i = 0
        while i <= limit:
            j = 0
            while j <= limit:
                x = min(self.width - 1, i + topLeft[0])
                y = min(self.height - 1, j + topLeft[1])
                hashId = self.getHashId((x, y))
                ids.add(hashId)
                j += stepSize
            i += stepSize
        return ids
    def getHashId(self, pos):
        return int(numpy.floor(pos[0] / self.cellSize) + numpy.floor(pos[1] / self.cellSize) * self.cols)
