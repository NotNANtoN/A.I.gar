from .player import Player
import random

class Bot(object):
	"""docstring for Bot"""
	def __init__(self, player):
		self.player = player
		self.playerCommands = (-1, -1, 0, 0)


	def update(self):
		# This bot does random stuff
		fov = self.player.getFov()
		midPoint = fov[0]
		x = midPoint[0]
		y = midPoint[1]
		width = fov[1][0]
		height = fov[1][1]
		xChoice = randint(x - width, x + width)
		yChoice = randint(y - width, y + width)
		splitChoice = True if random.uniform(0, 1) > 0.95 else False
		ejectChoice = True if random.uniform(0, 1) > 0.99 else False
		self.player.setCommands(xChoice,yChoice,splitChoice,ejectChoice)
		

		