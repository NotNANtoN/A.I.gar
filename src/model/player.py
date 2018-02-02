
class Player(object):
	"""docstring for Player"""
	def __init__(self, name, x, y):
		startCell = Cell(x,y)
		self.cells = [startCell]
		self.totalsize = startCell.getSize()




	def addCell(self, cell):
		self.cells.append(cell)

	def removeCell(self, cell):
		self.cells.remove(cell)
		