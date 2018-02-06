from model.model import *
from view.view import View

if __name__ == '__main__':
    SCREEN_WIDTH = 1200
    SCREEN_HEIGHT = 900

    model = Model(SCREEN_WIDTH, SCREEN_HEIGHT)

    numberOfBots = int(input("Please enter the number of bots:"))
    for i in range(0, numberOfBots):
        model.createBot()

    checkHuman = int(input("Do you want to play? If so type 1."))
    if checkHuman == 1:
        name = input("What's your name?")
        model.createHuman(name)
        # controller = Controller(model, view)

    view = View(SCREEN_WIDTH, SCREEN_HEIGHT, model)

    model.initialize()
    model.run()
