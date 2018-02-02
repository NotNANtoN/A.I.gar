
class Cell(object):
    def __init__(self,x, y, radius, color):
        self.radius = radius
        self.x = x
        self.y = y
        self.color = color
        self.vx = 0
        self.vy = 0


    def split(self):
        pass

    def eject(self):
        pass

    def getRadius(self):
        return self.radius

    def setPos(self, x , y):
        self.x = x
        self.y = y



    def updateDirection(self, x, v, maxX):
        return min( maxX, max( 0, x + v))

 
    def updatePos(self, maxX, maxY):
        self.x += self.updateDirection(self.x, self.vx, maxX)
        self.y += self.updateDirection(self.y, self.vy, maxY)


    def changeRadiusBy(self, val):
        self.radius += val



