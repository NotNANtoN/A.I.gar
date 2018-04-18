import time
import datetime
import matplotlib
matplotlib.use('Pdf')
import matplotlib.pyplot as plt
import numpy

from .bot import Bot
from .field import Field
from .parameters import *
from .player import Player

import linecache
import os
import sys
import shutil
import tracemalloc

# Useful function that displays the top 3 lines that use the most total memory so far
def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


# The model class is the main wrapper for the game engine.
# It contains the field and the players.
# It links the actions of the players to consequences in the field and updates information.

class Model(object):
    def __init__(self, guiEnabled, viewEnabled, virusEnabled, resetLimit):
        self.listeners = []
        self.viewEnabled = viewEnabled
        self.guiEnabled = guiEnabled
        self.virusEnabled = virusEnabled
        self.resetLimit = resetLimit
        self.trainingEnabled = True

        self.players = []
        self.bots = []
        self.humans = []
        self.playerSpectator = None
        self.spectatedPlayer = None
        self.players = []
        self.field = Field(virusEnabled)
        self.screenWidth = None
        self.screenHeight = None
        self.counter = 0
        self.timings = []
        self.rewards = []
        self.tdErrors = []
        self.meanErrors = []
        self.meanRewards = []

        tracemalloc.start()

    def initialize(self):
        if self.trainingEnabled:
            self.createPath()
            self.saveSpecs()
            self.copyParameters()
            for bot in self.bots:
                if bot.getType() == "NN":
                    data = self.getRelevantModelData(bot)
                    print(data)
                    break
        self.field.initialize()
        

    def resetModel(self):
        print("Resetting field and players!")
        for bot in self.bots:
            if bot.getType() == "NN":
                print("Average reward of ", bot.getPlayer(), " for this episode: ", bot.getAvgReward())
        self.field.reset()
        self.resetBots()

    def update(self):
        # Reset the model after self.resetLimit steps:
        if self.resetLimit > 0 and self.counter > 0 and  self.counter % self.resetLimit  == 0:
            self.resetModel()

        timeStart = time.time()
        timeProcessStart = time.process_time()
        # Get the decisions of the bots. Update the field accordingly.
        self.takeBotActions()
        self.field.update()
        # Update view if view is enabled
        if self.guiEnabled and self.viewEnabled:
            self.notify()
        # Slow down game to match FPS
        if self.humans:
            time.sleep(max( (1/FPS) - (time.time() - timeStart),0))

        # Save the models occasionally in case the program crashes at some point
        if self.trainingEnabled and self.counter != 0 and self.counter % 5000 == 0:
            self.saveSpecs()
            self.saveModels(self.path)
            if self.counter % 50000 == 0:
                self.save()

        # Store debug info and display progress
        if self.trainingEnabled and "NN" in [bot.getType() for bot in self.bots]:
            errors = []
            rewards = []
            for bot in self.bots:
                if bot.getType() == "NN":
                    if bot.getCurrentActionIdx() != None and bot.getLearningAlg().getTDError() != None:
                        reward = bot.getLastReward()
                        tdError = abs(bot.getLearningAlg().getTDError())
                        rewards.append(reward)
                        errors.append(tdError)
            # Save the mean td error and reward for the bots per update
            if len(rewards) > 0:
                self.rewards.append(numpy.mean(rewards))
                self.tdErrors.append(numpy.mean(errors))
            self.visualize(timeProcessStart)
        self.counter += 1

    def takeBotActions(self):
        for bot in self.bots:
            bot.makeMove()

    def resetBots(self):
        for bot in self.bots:
            bot.reset()

    def createPath(self):
        basePath = "savedModels"
        if not os.path.exists(basePath):
            os.makedirs(basePath)
        now = datetime.datetime.now()
        self.startTime = now
        nowStr = now.strftime("%b-%d_%H:%M")
        path = basePath + "/" + nowStr
        # Also display seconds in name if we already have a model this minute
        if os.path.exists(path):
            nowStr = now.strftime("%b-%d_%H:%M:%S")
            path = basePath + "/" + nowStr
        os.makedirs(path)
        path += "/"
        self.path = path

    def copyParameters(self):
        # Copy the simulation, NN and RL parameters so that we can load them later on
        shutil.copy("model/networkParameters.py", self.path)
        shutil.copy("model/parameters.py", self.path)

    def saveModels(self, path, end = False):
        savedTypes = []
        for bot in self.bots:
            botType = bot.getType()
            if botType != "Greedy" and botType not in savedTypes:
                bot.getLearningAlg().getNetwork().saveModel(path, bot)
                savedTypes.append(botType)
        if end:
            path = path[:-1]
            suffix = ""
            if self.counter >= 1000000000:
                suffix = "B"
                self.counter = int(self.counter / 1000000000)
            elif self.counter >= 1000000:
                suffix = "M"
                self.counter = int(self.counter / 1000000)
            elif self.counter >= 1000:
                suffix = "K"
                self.counter = int(self.counter / 1000)
            updatedPath = path + "_" + str(self.counter) + suffix
            os.rename(path, updatedPath)
            self.path = updatedPath + "/"

    def saveSpecs(self, end = False):
        # Save any additional info on this training
        name_of_file = self.path + "model_specifications.txt"
        savedTypes = []
        with open(name_of_file, "w") as file:
             for bot in self.bots:
                botType = bot.getType()
                if botType != "Greedy" and botType not in savedTypes:
                    data = self.getRelevantModelData(bot, end)
                    file.write(data)
                    file.write("\n")
                    savedTypes.append(botType)

    def save(self, end = False):
        self.saveModels(self.path, end)
        self.saveSpecs(end)
        self.plotTDError()
        self.plotMassesOverTime()
        self.plotQValuesOverTime()

    def getRelevantModelData(self, bot, end = False):
        network = bot.getLearningAlg().getNetwork()
        data = ""
        #Simulation:
        data += "Simulation:\n"
        data += "Start datetime - " + self.startTime.strftime("%b-%d_%H:%M:%S") + "\n"
        now = datetime.datetime.now()
        data += "Running for - " + str(now - self.startTime) + "\n"
        if end:
            data += "End datetime - " + now.strftime("%b-%d_%H:%M:%S") + "\n"
        data += "steps - " + str(self.counter) + "\n"
        data += "number of rl bots - " + str(Bot.num_NNbots) + "\n"
        data += "number of greedy bots - " + str(Bot.num_Greedybots) + "\n"
        data += "Virus Enabled - " + str(self.virusEnabled) + "\n"
        data += "Reset every x steps - " + str(self.resetLimit) + "\n"
        data += "\n"
        # RL:
        data += "Reinforcement learning:\n"
        data += "Epsilon - " + str(network.getEpsilon()) + "\n"
        data += "Discount factor - " + str(network.getDiscount()) + "\n"
        data += "Frame skip rate - " + str(network.getFrameSkipRate()) + "\n"
        data += "State representation - " + ("Grid" if bot.gridViewEnabled else "Simple") + "\n"
        if bot.gridViewEnabled:
            data += "Grid - " + str(network.getGridSquaresPerFov()) + "x" +  str(network.getGridSquaresPerFov())  + "\n"
        data += "Experience Replay - " + ("Enabled" if bot.expRepEnabled else "Disabled") + "\n"
        data += "Target Network steps until update - " + str(network.getTargetNetworkMaxSteps()) + "\n"
        data += "Name of model that was loaded - " + (network.getLoadedModelName() if network.getLoadedModelName() else "None") + "\n"
        data += "\n"
        # ANN:
        data += "ANN:\n"
        data += "Input layer neurons(stateReprLen) - " + str(network.getStateReprLen()) + "\n"
        data += "First hidden layer neurons - " + str(network.getHiddenLayer1()) + "\n"
        data += "Second hidden layer neurons - " + str(network.getHiddenLayer2()) + "\n"
        data += "Third hidden layer neurons - " + str(network.getHiddenLayer3()) + "\n"
        data += "Output layer neurons(number of actions) - " + str(network.getNumOfActions()) + "\n"
        data += "Learning rate - " + str(network.getLearningRate()) + "\n"
        data += "Activation function hidden layer(s) - " + network.getActivationFuncHidden() + "\n"
        data += "Activation function output layer - " + network.getActivationFuncOutput() + "\n"
        data += "Optimizer - " + str(network.getOptimizer()) + "\n"
        return data

    def printBotMasses(self):
        for bot in self.bots:
            mass = bot.getPlayer().getTotalMass()
            print("Mass of ", bot.getPlayer(), ":", end = " ")
            print(round(mass,1) if mass != None else (bot.getPlayer, " is dead!"))

    def visualize(self, timeStart):
        stepsTillUpdate = 100
        self.timings.append(time.process_time() - timeStart)
        if self.counter % stepsTillUpdate == 0 and self.counter != 0:
            avgOnset = -100 if len(self.rewards) >= 100 else -1 * len(self.rewards)
            recentMeanReward = numpy.mean(self.rewards[avgOnset:])
            recentMeanTDError = numpy.mean(self.tdErrors[avgOnset:])
            self.meanErrors.append(recentMeanTDError)
            self.meanRewards.append(recentMeanReward)
            print(" ")
            print("Avg time since update start for the last ", stepsTillUpdate, " steps: ", str(round(numpy.mean(self.timings[len(self.timings) - stepsTillUpdate:]),3)))
            print("Avg reward   last 100 steps:", round(recentMeanReward, 4), " Min: ", round(min(self.rewards),4), " Max: ", round(max(self.rewards), 4))
            print("Avg abs TD-Error last 100 steps: ", round(recentMeanTDError, 4), " Min: ", round(min(self.tdErrors),4), " Max: ", round(max(self.tdErrors), 4))
            print("Step: ", self.counter)
            print("Number of stored rewards: ", len(self.rewards))
            self.printBotMasses()
            print(" ")
   

        if self.counter != 0 and self.counter % 100000 == 0 and self.trainingEnabled:
            self.plotTDError()
            self.plotMassesOverTime()

    def runningAvg(self, array, numberOfPoints):
        res = len(array) // numberOfPoints  # res: running error step
        meanArray = []
        epoch = []
        for i in range(res):
            for j in range(numberOfPoints):
                if array[i * numberOfPoints + j] is not None:
                    epoch.append(array[i * numberOfPoints + j])
            meanArray.append(numpy.mean(epoch))
        return meanArray
        # return [numpy.mean(array[idx * numberOfPoints:(idx + 1) * numberOfPoints]) for idx in range(res)]

    def plotTDError(self):
        path = self.path
        numberOfPoints = 200
        meanOfmeanError   = self.runningAvg(self.tdErrors, numberOfPoints)
        meanOfmeanRewards = self.runningAvg(self.rewards,  numberOfPoints)
        plt.plot(range(len(meanOfmeanError)), meanOfmeanError, label="Absolute TD-Error")
        plt.xlabel("Time")
        plt.ylabel("TD error averaged")
        plt.savefig(path + "TD-Errors.pdf")
        plt.plot(range(len(meanOfmeanRewards)), meanOfmeanRewards, label="Reward")
        plt.legend()
        plt.ylabel("Running avg")
        plt.savefig(path + "Reward_and_TD-Error.pdf")
        plt.close()

    def plotMassesOverTime(self):
        for bot in self.bots:
            masses = bot.getMassOverTime()
            meanMass = round(numpy.mean(masses),1)
            medianMass = round(numpy.median(masses),1)
            varianceMass = round(numpy.std(masses), 1)
            maxMass = round(max(masses), 1)
            len_masses = len(masses)
            playerName = str(bot.getPlayer())
            plt.plot(range(len_masses), masses)
            plt.title("Mass of " + playerName + "- Mean: " + str(meanMass) + " Median: " + str(medianMass) + " Std: " +
                      str(varianceMass) + " Max: " + str(maxMass))
            plt.xlabel("Step")
            plt.ylabel("Total Player Mass")
            plt.savefig(self.path + "MassOverTime" + playerName + ".pdf")
            plt.close()

    def plotQValuesOverTime(self):
        for bot in self.bots:
            qValues = bot.getLearningAlg().getQValues()
            qValues = self.runningAvg(qValues, 200)
            meanQValue = round(numpy.mean(qValues), 1)
            playerName = str(bot.getPlayer())
            plt.plot(range(len(qValues)), qValues)
            plt.title("Q-Values of " + playerName + "- Mean: " + str(meanQValue))
            plt.xlabel("Step")
            plt.ylabel("Q-value of current action")
            plt.savefig(self.path + "qValuesOverTime" + playerName + ".pdf")
            plt.close()


    # Setters:
    def setEpsilon(self, val):
        Bot.epsilon = val

    def createPlayer(self, name):
        newPlayer = Player(name)
        self.addPlayer(newPlayer)
        return newPlayer

    def createBot(self, type, learningAlg):
        name = type + " " + str(len(self.bots))
        newPlayer = self.createPlayer(name)
        bot = Bot(newPlayer, self.field, type, self.trainingEnabled, learningAlg)
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
    def getPath(self):
        return self.path

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

    def getTrainingEnabled(self):
        return self.trainingEnabled

    # MVC related method
    def register_listener(self, listener):
        self.listeners.append(listener)

    def notify(self):
        for listener in self.listeners:
            listener()
