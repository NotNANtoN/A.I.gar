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

print("Timing math.sqrt: ", timingSqrt(math.sqrt, loops))
print("Timing numpy.sqrt: ", timingSqrt(numpy.sqrt, loops))
#print("Timing math.sqrt: ", timingSqrt(math.sqrt, loops))









