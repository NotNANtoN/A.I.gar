from model.model import *
from view.view import View
from controller.controller import Controller
import sys

def quitGame():
    print("Exitting...")
    sys.exit()


if __name__ == '__main__':
    #This is used in case we want to use a freezing program to create an .exe 
    if getattr(sys, 'frozen', False):
        os.chdir(sys._MEIPASS)

    SCREEN_WIDTH = 1200
    SCREEN_HEIGHT = 900
    MAXBOTS = 1000
    MAXHUMANPLAYERS = 2

    model = Model(SCREEN_WIDTH, SCREEN_HEIGHT)
    

    numberOfBots= int(input("Please enter the number of bots:\n"))
    if not(numberOfBots > MAXBOTS):
        if not(numberOfBots < 0):
            for i in range(0,numberOfBots):
                model.createBot()
        else:
            print("Number of bots can't be negative.")
            self.quitGame()
    else:
        print("Too many bots.")
        self.quitGame()

    numberOfHumans = int(input("Please enter the number of human players: (" + str(MAXHUMANPLAYERS) + " max)\n"))
    if (numberOfHumans <= MAXHUMANPLAYERS):
        if (numberOfHumans >= 0):
            if(numberOfHumans > 0):
                for i in range(1,numberOfHumans+1):
                    name = input("Player" + str(i) +" name:\n")
                    model.createHuman(name)
            else:
                pass
        else:
            print("Number of humans can't be negative.")
            self.quitGame()
    else:
        print("Too many humans.")
        self.quitGame()


    view = View(SCREEN_WIDTH, SCREEN_HEIGHT, model)
    controller = Controller(model, view)

    model.initialize()
    view.draw()

    while controller.running:
        controller.process_input()
        model.update()


    