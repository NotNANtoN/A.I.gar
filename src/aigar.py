import matplotlib
import sys

from keras.models import load_model

from controller.controller import Controller
from model.model import *
from model.parameters import *
from model.networkParameters import *
from view.startScreen import StartScreen
from view.view import View

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


def createBots(number, model, type, modelName, gridSquarePerFov = 0):
    if type == "NN":
        Bot.num_NNbots = number
        if not GRID_VIEW_ENABLED:
            Bot.stateReprLen = 12
        Bot.initializeNNs()
    elif type == "Greedy":
        Bot.num_Greedybots = number
    for i in range(number):
        model.createBot(type)
    # Load a stored model:
    if modelName is not None:
        for bot in model.getBots():
            if bot.getType() == type:
                Bot.loadedModelName = modelName
                Bot.valueNetwork = load_model(modelName + ".h5")
                Bot.targetNetwork = Bot.valueNetwork
                break



if __name__ == '__main__':
    # This is used in case we want to use a freezing program to create an .exe
    if getattr(sys, 'frozen', False):
        os.chdir(sys._MEIPASS)

    guiEnabled = int(input("Enable GUI?: (1 == yes)\n"))
    guiEnabled = (guiEnabled == 1)
    viewEnabled = False
    if guiEnabled:
        viewEnabled = int(input("Display view?: (1 == yes)\n"))
        viewEnabled = (viewEnabled == 1)
    else:
        maxSteps = int(input("For how many steps do you want to train?\n"))
    virusEnabled = int(input("Viruses enabled? (1==True)\n")) == 1
    resetSteps = int(input("Reset model after X steps (X==0 means no reset)\n"))
    model = Model(guiEnabled, viewEnabled, virusEnabled, resetSteps)

    numberOfGreedyBots = int(input("Please enter the number of Greedy bots:\n"))
    numberOfBots = numberOfGreedyBots
    if fitsLimitations(numberOfBots, MAXBOTS):
        createBots(numberOfGreedyBots, model, "Greedy", None)

    numberOfNNBots = int(input("Please enter the number of NN bots:\n"))
    numberOfBots += numberOfNNBots
    if fitsLimitations(numberOfBots, MAXBOTS) and numberOfNNBots > 0:
        modelName = None
        loadModel = int(input("Do you want to load a model? (1 == yes) (2=load model from last autosave)\n"))
        if loadModel == 1:
            while modelName == None:
                modelName = input("Enter the model name (without .h5): ")
        if loadModel == 2:
            modelName = "mostRecentAutosave"
        enableTrainMode = int(input("Do you want to train the network?: (1 == yes)\n"))
        model.setTrainingEnabled(enableTrainMode == 1)
        createBots(numberOfNNBots, model, "NN", modelName)


    if numberOfBots == 0 and not viewEnabled:
        modelMustHavePlayers()

    numberOfHumans = 0
    if guiEnabled and viewEnabled:
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

    model.initialize()
    if guiEnabled:
        view = View(model, screenWidth, screenHeight)
        controller = Controller(model, viewEnabled, view)
        view.draw()
        while controller.running:
            controller.process_input()
            model.update()
    else:
        endEpsilon = model.getEpsilon()
        startEpsilon = 1
        startEpsilon = 1
        smallPart = int(maxSteps / 200)
        for step in range(maxSteps):
            model.update()
            if step < maxSteps / 4:
                lr = startEpsilon - (1 - endEpsilon) * step / (maxSteps / 4)
            else:
                lr = endEpsilon
            model.setEpsilon(lr)
            if step % smallPart == 0 and step != 0:
                print("Trained: ", round(step / maxSteps * 100, 1), "%")

    if model.getTrainingEnabled():
        model.save()
