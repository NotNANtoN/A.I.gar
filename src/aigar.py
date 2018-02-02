from model.model import *
from view.view import View


if __name__ == '__main__':
    SCREEN_WIDTH = 1200
    SCREEN_HEIGHT = 900



	model = Model()
    view = View(SCREEN_WIDTH, SCREEN_HEIGHT, model)
    controller = Controller(model, view)

	numberOfBots= int(input("Please enter the number of bots:"))
	for( i in range(0,numberOfBots) ):
		model.addBot()

	checkHuman = int(input("Do you want to play?  If so type 1."))
    if( checkHuman == 1 ):
    	model.addHuman()

    model.run()
	