from .field import Field
from .cell import Cell
from .player import Player
from .bot import Bot
import numpy

# The model class is the main wrapper for the game engine.
# It contains the field and the players.
# It links the actions of the players to consequences in the field and updates information.

class Model(object):
    def __init__(self, width, height):
        self.listeners = []

        self.players = []
        self.bots = []
        self.human = None
        self.players = []
        self.field = Field()

        self.screenWidth = width
        self.screenHeight = height

    def run(self):
        self.update()

    def update(self):
        # Get the decisions of the bots/human. Update the field accordingly.
        for bot in self.bots:
            bot.update()
        if( self.hasHuman() ):
            self.getHumanInput()

        self.field.update()
        self.notify(None)
        self.run()

    # Find the point where the player clicked, taking into account that he only sees the fov
    def isRelativeClick(self):
        mousePos = pygame.mouse.get_pos()
        fovPos = self.human.getFovPos()
        fovDims = self.human.getFovDims()
        difference = numpy.subtract(mousePos, [fovDims[0] / 2,fovDims[1] / 2])
        command = numpy.add(difference, fovPos[0], fovPos[1])

    def handleKeyInput(self):
        for event in pygame.event.get():
            if( event.type == KEY_DOWN ):
                if( event.key == pygame.K_SPACE  and human.canSplit() ):
                    human.setSplit(True)
                elif( event.key == pygame.K_w and human.canEject() ):
                    human.setEject(True)

    def updateHumanInput():
        handleKeyInput()
        command = isRelativeClick()
        return command

    # Setters:
    def createPlayer(self, name):
        newPlayer = Player(name, self.field)
        self.addPlayer(newPlayer)
        return newPlayer

    def createBot(self):
        name = "Bot " + str(len(self.bots))
        newPlayer = createPlayer(name)
        bot = Bot(newPlayer)
        self.addBot(bot)

    def createHuman(self, name):
        newPlayer = createPlayer(name)
        self.addHuman(newPlayer)

    def addPlayer(self, player):
        self.addPlayer(player)
        self.field.addPlayer(player)

    def addBot(self, bot):
        self.bots.append(bot)

    def addHuman(self, player): 
        self.human = player

    # Checks:
    def hasHuman(self):
        return human != Null

    # Getters:
    def getField(self):
        return self.field

    def getCollectibles(self):
        return self.field.getCollectibles()

    def getViruses(self):
        return self.field.getViruses()

    def getPlayerCells(self):
        return self.field.getPlayerCells()


    # MVC related method
    def register_listener(self, listener):
        self.listeners.append(listener)

    def notify(self, event_name):
        for listener in self.listeners:
            listener(event_name)