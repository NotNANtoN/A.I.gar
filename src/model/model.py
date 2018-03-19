import time
import matplotlib.pyplot as plt
import numpy

from .bot import Bot
from .field import Field
from .parameters import *
from .player import Player


# The model class is the main wrapper for the game engine.
# It contains the field and the players.
# It links the actions of the players to consequences in the field and updates information.

class Model(object):
    def __init__(self, guiEnabled, viewEnabled):
        self.listeners = []
        self.viewEnabled = viewEnabled
        self.guiEnabled = guiEnabled
        self.trainingEnabled = True

        self.players = []
        self.bots = []
        self.humans = []
        self.playerSpectator = None
        self.spectatedPlayer = None
        self.players = []
        self.field = Field()
        self.screenWidth = None
        self.screenHeight = None
        self.counter = 0
        self.timings = []
        self.rewards = []
        self.tdErrors = []
        self.meanErrors = []
        self.meanRewards = []

    def initialize(self):
        self.field.initialize()

    def printDebugInfo(self):
        if self.hasHuman():
            pass

    def update(self):
        timeStart = time.time()
        timeProcessStart = time.process_time()
        # Get the decisions of the bots/human. Update the field accordingly.
        for bot in self.bots:
            bot.update()
        self.field.update()
        if self.guiEnabled and self.viewEnabled:
            self.notify()
        if __debug__:
            self.printDebugInfo()
        self.visualize(timeProcessStart)
        if self.humans:
            time.sleep(max( (1/FPS) - (time.time() - timeStart),0))

        if self.trainingEnabled:
            for bot in self.bots:
                if bot.getType() != "Greedy":
                    if bot.currentActionIdx:
                        reward = bot.getReward()
                        tdError = bot.getTDError(reward)
                        self.rewards.append(reward)
                        self.tdErrors.append(abs(tdError))

            self.counter += 1

    def saveModels(self, path):
        savedTypes = []
        for bot in self.bots:
            botType = bot.getType()
            if botType != "Greedy" and botType not in savedTypes:
                bot.saveModel(path)
                savedTypes.append(botType)

    def saveSpecs(self, path):
        name_of_file = path + "model_specifications.txt"
        savedTypes = []
        with open(name_of_file, "w") as file:
             for bot in self.bots:
                botType = bot.getType()
                if botType != "Greedy" and botType not in savedTypes:
                    data = self.getRelevantModelData(bot)
                    file.write(data)
                    file.write("\n")
                    savedTypes.append(botType)

    def save(self, path):
        self.saveModels(path)
        self.saveSpecs(path)
        self.plotTDerror(path)

    def getRelevantModelData(self, bot):
        data = ""
        #Simulation:
        data += "Simulation:\n"
        data += "steps - " + str(self.counter) + "\n"
        data += "number of rl bots - " + str(Bot.num_NNbots) + "\n"
        data += "number of greedy bots - " + str(Bot.num_Greedybots) + "\n"
        data += "\n"
        # RL:
        data += "Reinforcement learning:\n"
        data += "Epsilon - " + str(Bot.epsilon) + "\n"
        data += "Discount factor - " + str(Bot.discount) + "\n"
        data += "Frame skip rate - " + str(Bot.frameSkipRate) + "\n"
        data += "State representation - " + ("Grid" if bot.gridViewEnabled else "Simple") + "\n"
        if bot.gridViewEnabled:
            data += "Grid - " + str(Bot.gridSquaresPerFov) + "x" +  str(Bot.gridSquaresPerFov)  + "\n"
        data += "Experience Replay - " + ("Enabled" if bot.expRepEnabled else "Disabled") + "\n"
        data += "Target Network steps until update - " + str(Bot.targetNetworkMaxSteps) + "\n"
        data += "Name of model that was loaded - " + (Bot.loadedModelName if Bot.loadedModelName else "None") + "\n"
        data += "\n"
        # ANN:
        data += "ANN:"
        data += "Input layer neurons(stateReprLen) - " + str(Bot.stateReprLen) + "\n"
        data += "First hidden layer neurons - " + str(Bot.hiddenLayer1) + "\n"
        data += "Second hidden layer neurons - " + str(Bot.hiddenLayer2) + "\n"
        data += "Third hidden layer neurons - " + str(Bot.hiddenLayer3) + "\n"
        data += "Output layer neurons(number of actions) - " + str(Bot.num_actions) + "\n"
        data += "Learning rate - " + str(Bot.learningRate) + "\n"
        data += "Activation function hidden layer(s) - " + Bot.activationFuncHidden + "\n"
        data += "Activation function output layer - " + Bot.activationFuncOutput + "\n"
        data += "Optimizer - " + str(Bot.optimizer) + "\n"
        return data

    def visualize(self, timeStart):
        stepsTillUpdate = 100
        numReward = len(self.rewards)
        self.timings.append(time.process_time() - timeStart)
        if self.counter % stepsTillUpdate == 0 and self.counter != 0:
            recentMeanReward = numpy.mean(self.rewards)
            recentMeanTDError = numpy.mean(self.tdErrors)
            self.meanErrors.append(recentMeanTDError)
            self.meanRewards.append(recentMeanReward)
            print(" ")
            print("Avg time since update start for the last ", stepsTillUpdate, " steps: ", str(round(numpy.mean(self.timings[len(self.timings) - stepsTillUpdate:]),3)))
            print("Avg reward   last 100 steps:", round(recentMeanReward, 4), " Min: ", round(min(self.rewards),4), " Max: ", round(max(self.rewards), 4))
            print("Avg abs TD-Error last 100 steps: ", round(recentMeanTDError, 4), " Min: ", round(min(self.tdErrors),4), " Max: ", round(max(self.tdErrors), 4))
            print("Step: ", self.counter)
            print(" ")
            self.tdErrors = []
            self.rewards = []

    def plotTDerror(self, path = None):
        res = 10 #running error step
        meanOfmeanError = numpy.convolve(self.meanErrors, numpy.ones((res,))/res, mode='valid')
        meanOfmeanRewards = numpy.convolve(self.meanRewards, numpy.ones((res,))/res, mode='valid')
        plt.plot(range(len(meanOfmeanError)), meanOfmeanError, label="Absolute TD-Error")
        plt.xlabel("Steps in hundreds")
        plt.ylabel("Running abs TD-Error avg of the last 100 steps")
        if path:
            plt.savefig(path + "TD-Errors.png")
        plt.plot(range(len(meanOfmeanRewards)), meanOfmeanRewards, label="Reward")
        plt.legend()
        plt.xlabel("Steps in hundreds")
        plt.ylabel("Running  avg of the last 100 steps")
        if path:
            plt.savefig(path + "Reward_and_TD-Error.png")
        else:
            plt.show()
        plt.close()

    # Setters:
    def createPlayer(self, name):
        newPlayer = Player(name)
        self.addPlayer(newPlayer)
        return newPlayer

    def createBot(self, type, expRep, gridView):
        name = type + " " + str(len(self.bots))
        newPlayer = self.createPlayer(name)
        bot = Bot(newPlayer, self.field, type, expRep, gridView, self.trainingEnabled)
        self.addBot(bot)

    def createHuman(self, name):
        newPlayer = self.createPlayer(name)
        self.addHuman(newPlayer)

    def addPlayer(self, player):
        self.players.append(player)
        self.field.addPlayer(player)

    def addBot(self, bot):
        self.bots.append(bot)

    def addHuman(self, human):
        self.humans.append(human)

    def addPlayerSpectator(self):
        self.playerSpectator = True
        self.setSpectatedPlayer(self.players[0])

    def setSpectatedPlayer(self, player):
        self.spectatedPlayer = player

    def setViewEnabled(self, boolean):
        self.viewEnabled = boolean

    def setScreenSize(self, width, height):
        self.screenWidth = width
        self.screenHeight = height

    def setTrainingEnabled(self, trainMode):
        self.trainingEnabled = trainMode

    # Checks:
    def hasHuman(self):
        return bool(self.humans)

    def hasPlayerSpectator(self):
        return self.playerSpectator is not None

    # Getters:
    def getTopTenPlayers(self):
        players = self.getPlayers()[:]
        players.sort(key=lambda p: p.getTotalMass(), reverse=True)
        return players[0:10]

    def getHumans(self):
        return self.humans

    def getFovPos(self, humanNr):
        if self.hasHuman():
            fovPos = numpy.array(self.humans[humanNr].getFovPos())
        elif self.hasPlayerSpectator():
            fovPos = numpy.array(self.spectatedPlayer.getFovPos())
        else:
            fovPos = numpy.array([self.field.getWidth() / 2, self.field.getHeight() / 2])
        return fovPos

    def getFovSize(self, humanNr):
        if self.hasHuman():
            fovSize = self.humans[humanNr].getFovSize()
        elif self.hasPlayerSpectator():
            fovSize = self.spectatedPlayer.getFovSize()
        else:
            fovSize = self.field.getWidth()
        return fovSize

    def getField(self):
        return self.field

    def getPellets(self):
        return self.field.getPellets()

    def getViruses(self):
        return self.field.getViruses()

    def getPlayers(self):
        return self.players

    def getBots(self):
        return self.bots

    def getPlayerCells(self):
        return self.field.getPlayerCells()

    def getSpectatedPlayer(self):
        if self.hasHuman():
            return self.humans
        if self.hasPlayerSpectator():
            return self.spectatedPlayer
        return None

    # MVC related method
    def register_listener(self, listener):
        self.listeners.append(listener)

    def notify(self):
        for listener in self.listeners:
            listener()
