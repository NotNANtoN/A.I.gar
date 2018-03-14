from model.model import *
from view.view import View
from view.startScreen import StartScreen
from controller.controller import Controller
from model.parameters import *
from keras.models import load_model
import sys
import os

def modelMustHavePlayers():
    print("Model must have players")
    quit()

def fitsLimitations(number, limit):
    if number < 0:
        print("Number can't be negative.")
        quit()
    if number > limit:
        print("Number can't be larger than ", limit, ".")
        quit()
    return True

def defineScreenSize(humansNr):
    # Define screen size (to allow splitscreen)
    if humansNr == 2:
        return int(SCREEN_WIDTH * humansNr + humansNr -1), int(SCREEN_HEIGHT)
    if humansNr == 3:
        return int(SCREEN_WIDTH * humansNr * 2/3) + humansNr -1, int(SCREEN_HEIGHT * 2/3)

    return SCREEN_WIDTH, SCREEN_HEIGHT


def createHumans(numberOfHumans, model1):
    for i in range(numberOfHumans):
        name = input("Player" + str(i + 1) + " name:\n")
        model1.createHuman(name)


def createBots(number, model, type, expRep, gridView, modelName, explore = True):
    for i in range(number):
        model.createBot(type, expRep, gridView)
    # Load a stored model:
    if modelName is not None:
        for bot in model.getBots():
            if bot.getType() == type:
                Bot.valueNetwork = load_model(modelName + ".h5")
                break
    if explore == False:
        for bot in model.getBots():
            if bot.getType() == type:
                bot.setEpsilon(1)


if __name__ == '__main__':
    # This is used in case we want to use a freezing program to create an .exe
    if getattr(sys, 'frozen', False):
        os.chdir(sys._MEIPASS)

    viewEnabled = int(input("Display view?: (1 == yes)\n"))
    viewEnabled = (viewEnabled == 1)

    model = Model(viewEnabled)

    numberOfGreedyBots = int(input("Please enter the number of Greedy bots:\n"))
    numberOfBots = numberOfGreedyBots
    if fitsLimitations(numberOfBots, MAXBOTS):
        createBots(numberOfGreedyBots, model, "Greedy", False, False, None)

    numberOfNNBots = int(input("Please enter the number of NN bots:\n"))
    numberOfBots += numberOfNNBots
    if fitsLimitations(numberOfBots, MAXBOTS):
        modelName = None
        loadModel = int(input("Do you want to load a model? (1 == yes) (2=load model from last run)\n"))
        if loadModel == 1:
            while modelName == None:
                modelName = input("Enter the model name (without .h5): ")
        if loadModel == 2:
            modelName = "NN_latestModel"
        enableExpReplay = int(input("Do you want to enable experience replay? (1 == yes)\n"))
        enableGridView = int(input("Do you want to enable grid view state representation? (1 == yes)\n"))
        explore = int(input("Do you want to enable exploration? (1 == yes)\n"))
        createBots(numberOfNNBots, model, "NN", enableExpReplay == 1, enableGridView == 1, modelName, explore == 1)


    if numberOfBots == 0 and not viewEnabled:
        modelMustHavePlayers()

    numberOfHumans = 0
    if viewEnabled:
        numberOfHumans = int(input("Please enter the number of human players: (" + str(MAXHUMANPLAYERS) + " max)\n"))
        if fitsLimitations(numberOfHumans, MAXHUMANPLAYERS):
            createHumans(numberOfHumans, model)
        if numberOfBots + numberOfHumans == 0:
            modelMustHavePlayers()

        if not model.hasHuman():
            spectate = int(input("Do want to spectate an individual bot's FoV? (1 = yes)\n"))
            if spectate == 1:
                model.addPlayerSpectator()

    screenWidth, screenHeight = defineScreenSize(numberOfHumans)
    model.setScreenSize(screenWidth, screenHeight)
    startScreen = StartScreen(model)
    view = View(model, screenWidth, screenHeight)


    model.initialize()
    controller = Controller(model, viewEnabled, view)

    view.draw()

    while controller.running:
        controller.process_input()
        model.update()

    model.saveModels()
