from model.model import *
from view.view import View
from controller.controller import Controller
from model.parameters import *
import sys
import os


def fitsLimitations(number, limit):
    if number < 0:
        print("Number can't be negative.")
        quit()
    if number > limit:
        print("Number can't be larger than ", limit, ".")
        quit()
    return True


def createHumans(model1):
    for i in range(numberOfHumans):
        name = input("Player" + str(i + 1) + " name:\n")
        model1.createHuman(name)


def createBots(number, model1):
    for i in range(number):
        model1.createBot()


if __name__ == '__main__':
    # This is used in case we want to use a freezing program to create an .exe
    if getattr(sys, 'frozen', False):
        os.chdir(sys._MEIPASS)

    debug = int(input("Display debug info?: (1 == yes)\n"))
    debug = (debug == 1)

    model = Model(SCREEN_WIDTH, SCREEN_HEIGHT, debug)

    numberOfBots = int(input("Please enter the number of bots:\n"))
    if fitsLimitations(numberOfBots, MAXBOTS):
        createBots(numberOfBots, model)

    numberOfHumans = int(input("Please enter the number of human players: (" + str(MAXHUMANPLAYERS) + " max)\n"))
    if fitsLimitations(numberOfHumans, MAXHUMANPLAYERS):
        createHumans(model)

    if not model.hasHuman():
        spectate = int(input("Do you want to spectate the bots then? (1 = yes)"))
        if spectate == 1:
            model.addSpectator()

    view = View(model, SCREEN_WIDTH, SCREEN_HEIGHT)
    controller = Controller(model, view)

    model.initialize()
    view.draw()

    while controller.running:
        controller.process_input()
        model.update()
