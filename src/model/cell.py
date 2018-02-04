import numpy


class Cell(object):
    MOVESPEED = 1
    SPLITSPEED = 2 # Speed of just spawned cell

    def __init__(self, x, y, radius, color):
        self.radius = radius
        self.x = x
        self.y = y
        self.color = color
        self.vx = 0
        self.vy = 0

    def setMoveDirection(self, commandPoint):
        difference = numpy.subtract(commandPoint, [self.x, self.y])
        angle = numpy.arctan2(differece[1], difference[0])
        self.vx = MOVESPEED * numpy.sin(angle)
        self.vy = MOVESPEED * numpy.cos(angle)

    def split(self):
        pass

    def eject(self):
        pass

    def updateDirection(self, x, v, maxX):
        return min( maxX, max( 0, x + v))

 
    def updatePos(self, maxX, maxY):
        self.x += self.updateDirection(self.x, self.vx, maxX)
        self.y += self.updateDirection(self.y, self.vy, maxY)


    # Setters:
    def setPos(self, x , y):
        self.x = x
        self.y = y

    def setRadius(self, val):
        self.radius = val

    # Getters:
    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def getPos(self):
        return (x,y)

    def getColor(self):
        return self.color

    def getRadius(self):
        return self.radius




