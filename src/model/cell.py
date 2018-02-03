
class Cell(object):
    def __init__(self,x, y, radius, color, player):
        self.player = player
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
    def getPos(self):
        return (x,y)

    def getColor(self):
        return self.color

    def getRadius(self):
        return self.radius




