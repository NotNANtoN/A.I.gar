from .field import Field
from .cell import Cell
from .player import Player
# The model class is the main wrapper for the game engine.
# It contains the field and the players.
# It links the actions of the players to consequences in the field and updates information.

class Model(object):
	def __init__(self):
		self.listeners = []

		self.bots = []
		self.human = Null
		self.players = []
		self.field = Field()

	def addBot(self):
		self.bots.append( computerPlayer() )

	def addHuman(self): 
		self.human = humanPlayer

	def update(self):
		# Get the decisions of the bots/human. Update the field accordingly.
		for bot in self.bots:
			bot.update()
        if( self.hasHuman() ):
        	self.getHumanInput()

        self.field.update()
        self.notify(None)

    def updateHumanInput():
    	for event in pygame.event.get():
    		if( event.type == KEY_DOWN ):
    			if( event.key = pygame.K_SPACE  and human.canSplit() ):
    				human.split()
    			elif( event.key == pygame.K_w and human.canEject() ):
    				human.eject()
    	mousePos = pygame.mouse.get_pos()
    	difference = mousePos - human.getPos()


    def hasHuman(self):
    	return human != Null


    # MVC related method
    def register_listener(self, listener):
        self.listeners.append(listener)

    def notify(self, event_name):
        for listener in self.listeners:
            listener(event_name)