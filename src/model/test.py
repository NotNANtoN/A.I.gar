import numpy
import time
import math

def calcHash1(x, y):
    return numpy.floor(x / 20) + numpy.floor(y / 20) * 5

def calcHash2(x, y):
    return int(x / 20) + int(y / 20) * 5

def calcHash3(x, y):
    return math.floor(x / 20) + math.floor(y / 20) * 5
# int() fastest!


def timing1(func, loops):
    xArray = [numpy.random.randint(0,numb) for numb in range(1, loops + 2)]
    yArray = [numpy.random.randint(0,numb) for numb in range(1, loops + 2)]
    timeStart = time.process_time()
    for i in range(loops):
        func(xArray[i], yArray[i])
    return (time.process_time() - timeStart)

loops = 50000
#print("Timing nump floor: ", timing1(calcHash1, loops))
#print("Timing int(): ", timing1(calcHash2, loops))
#print("Timing math floor: ", timing1(calcHash3, loops))

def timingNump(func, loops):
    xArray = [numpy.array([numpy.random.randint(0,numb),numpy.random.randint(0,numb)]) for numb in range(1, loops + 2)]
    yArray = [numpy.array([numpy.random.randint(0,numb),numpy.random.randint(0,numb)]) for numb in range(1, loops + 2)]
    timeStart = time.process_time()
    for i in range(loops):
        pos1 = xArray[i]
        pos2 = yArray[i]
        func(pos1, pos2)
    return (time.process_time() - timeStart)

def timingNorm(func, loops):
    xArray = [numpy.random.randint(0,numb) for numb in range(1, loops * 2 + 2)]
    yArray = [numpy.random.randint(0,numb) for numb in range(1, loops * 2 + 2)]
    timeStart = time.process_time()
    for i in range(loops):
        pos1 = xArray[i], xArray[i * 2]
        pos2 = yArray[i], yArray[i * 2]
        func(pos1, pos2)
    return (time.process_time() - timeStart)


def squaredDistance1(pos1, pos2):
        return numpy.sum(numpy.power(pos1 - pos2, 2))

def squaredDistance2(pos1, pos2):
	return (pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2

def squaredDistance3(pos1, pos2):
	return (pos1[0] - pos2[0]) * (pos1[0] - pos2[0])  + (pos1[1] - pos2[1]) *  (pos1[1] - pos2[1])


#print("Timing numpy: ", timingNump(squaredDistance1, loops))
#print("Timing native: ", timingNorm(squaredDistance2, loops))
#print("Timing native 2: ", timingNorm(squaredDistance3, loops))
# native 2 fastest!

def timingSqrt(func, loops):
    xArray = [numpy.random.randint(0,100) for numb in range(1, loops + 2)]
    timeStart = time.process_time()
    for i in range(loops):
        func(xArray[i])
    return (time.process_time() - timeStart)

#print("Timing math.sqrt: ", timingSqrt(math.sqrt, loops))
#print("Timing numpy.sqrt: ", timingSqrt(numpy.sqrt, loops))


bucketSize = 20
width = height = 100
cols = 5

def timingIds(func, loops):
    posArray = [numpy.array([numpy.random.randint(0,100),numpy.random.randint(0,100)]) for numb in range(1, loops + 2)]
    rArray = [numpy.random.randint(1,50) for numb in range(1, loops + 2)]
    timeStart = time.process_time()
    for i in range(loops):
        func(posArray[i], rArray[i])
    return (time.process_time() - timeStart)

def timingIdsNoNumpy(func, loops):
    posArray = [[numpy.random.randint(0,100),numpy.random.randint(0,100)] for numb in range(1, loops + 2)]
    rArray = [numpy.random.randint(1,50) for numb in range(1, loops + 2)]
    timeStart = time.process_time()
    for i in range(loops):
        func(posArray[i], rArray[i])
    return (time.process_time() - timeStart)



def getIdsForArea(pos, radius):
        ids = set()
        hashFunc = getHashId
        topLeft = (max(0, pos[0] - radius), max(0, pos[1] - radius))
        bucketTopLeft = (topLeft[0] - topLeft[0] % bucketSize, topLeft[1] - topLeft[1] % bucketSize)
        stepSize = bucketSize
        limitX = min(width - 1, pos[0] + radius)
        limitY = min(height - 1, pos[1] + radius)

        x = bucketTopLeft[0]
        while x <= limitX:
            y = bucketTopLeft[1]
            while y <= limitY:
                ids.add(hashFunc(x, y))
                y += stepSize
            x += stepSize
        return ids

def getIdsForAreaForLoop(pos, radius):
        ids = set()
        hashFunc = getHashId
        topLeft = (max(0, pos[0] - radius), max(0, pos[1] - radius))
        bucketLeft = int(topLeft[0] - topLeft[0] % bucketSize)
        bucketTop = int(topLeft[1] - topLeft[1] % bucketSize)
        limitX = int(min(width, pos[0] + radius + 1))
        limitY = int(min(height, pos[1] + radius + 1))

        for x in range(bucketLeft, limitX, bucketSize):
            for y in range(bucketTop, limitY, bucketSize):
                ids.add(hashFunc(x,y))
        return ids	

def getIdsForAreaArray(pos, radius):
        ids = []
        hashFunc = getHashId
        topLeft = (max(0, pos[0] - radius), max(0, pos[1] - radius))
        bucketLeft = int(topLeft[0] - topLeft[0] % bucketSize)
        bucketTop = int(topLeft[1] - topLeft[1] % bucketSize)
        limitX = int(min(width, pos[0] + radius + 1))
        limitY = int(min(height, pos[1] + radius + 1))

        for x in range(bucketLeft, limitX, bucketSize):
            for y in range(bucketTop, limitY, bucketSize):
                ids.append(hashFunc(x,y))
        return ids	


def getIdsForAreaNoNumpy(pos, radius):
        ids = set()
        hashFunc = getHashId
        topLeft = (max(0, pos[0] - radius), max(0, pos[1] - radius))
        bucketLeft = int(topLeft[0] - topLeft[0] % bucketSize)
        bucketTop = int(topLeft[1] - topLeft[1] % bucketSize)
        limitX = int(min(width, pos[0] + radius + 1))
        limitY = int(min(height, pos[1] + radius + 1))

        for x in range(bucketLeft, limitX, bucketSize):
            for y in range(bucketTop, limitY, bucketSize):
                ids.add(hashFunc(x,y))
        return ids

def getHashId( x, y):
    return int(x / bucketSize) + int(y / bucketSize) * cols


loops = 100000
#print("Timing while: ", timingIds(getIdsForArea, loops))
# while is 1.5 times slower
print("Timing for loop: ", timingIds(getIdsForAreaForLoop, loops))
#print("Timing Array: ", timingIds(getIdsForAreaArray, loops))
# Array might be a bit faster? But hard to measure, and if so then about 1-2%
print("Timing no numpy: ", timingIdsNoNumpy(getIdsForAreaNoNumpy, loops))
# no numpy is 30% faster













