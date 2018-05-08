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
        self.path = None
        self.superPath = None
        self.startTime = None

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
        self.dataFiles = {}

        tracemalloc.start()

    def initialize(self, modelHasBeenLoaded):
        if self.trainingEnabled and not modelHasBeenLoaded:
            self.saveSpecs()
            for bot in self.bots:
                if bot.getType() == "NN":
                    data = self.getRelevantModelData(bot)
                    print(data)
                    break
        self.field.initialize()

    def loadModel(self, path):
        self.setPath(path)
        now = datetime.datetime.now()
        self.startTime = now

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
            if self.counter != 0 and self.counter % 50000 == 0:
                self.save()

        # Store debug info and display progress
        if self.trainingEnabled and "NN" in [bot.getType() for bot in self.bots]:
            self.storeRewardsAndTDError()
            self.visualize(timeProcessStart)

        self.counter += 1

    def storeRewardsAndTDError(self):
        errors = []
        rewards = []
        for bot in self.bots:
            if bot.getType() == "NN":
                if bot.getCurrentAction() is not None:
                    reward = bot.getLastReward()
                    rewards.append(reward)
                if bot.getLearningAlg().getTDError() is not None:
                    tdError = abs(bot.getLearningAlg().getTDError())
                    errors.append(tdError)
        # Save the mean td error and reward for the bots per update
        if len(rewards) > 0:
            self.rewards.append(numpy.mean(rewards))
        if len(errors) > 0:
            self.tdErrors.append(numpy.mean(errors))

    def takeBotActions(self):
        for bot in self.bots:
            bot.makeMove()

    def resetBots(self):
        for bot in self.bots:
            bot.reset()


    def initModelFolder(self, name = None):
        if name is None:
            self.createPath()
        else:
            self.createNamedPath(name)
        self.copyParameters()


    def createPath(self):
        basePath = "savedModels"
        if not os.path.exists(basePath):
            os.makedirs(basePath)
        now = datetime.datetime.now()
        self.startTime = now
        nowStr = now.strftime("%b-%d_%H:%M")
        path = basePath + "/" + "$" + nowStr + "$"
        # Also display seconds in name if we already have a model this minute
        if os.path.exists(path):
            nowStr = now.strftime("%b-%d_%H:%M:%S")
            path = basePath + "/" + "$" + nowStr + "$"
        os.makedirs(path)
        path += "/"
        self.path = path

    def createNamedPath(self, name):
        #Create savedModels folder
        basePath = "savedModels"
        if not os.path.exists(basePath):
            os.makedirs(basePath)
        #Create subFolder for given parameter tweaking
        subPath = name
        subName = os.getcwd() + "/" + subPath
        if not os.path.exists(subPath):
            os.makedirs(subPath)
        #Create folder based on name
        now = datetime.datetime.now()
        self.startTime = now
        nowStr = now.strftime("%b-%d_%H:%M")
        path = subPath + "/"  + "$" + nowStr + "$"
        # Also display seconds in name if we already have a model this minute
        if os.path.exists(path):
            nowStr = now.strftime("%b-%d_%H:%M:%S")
            path = subPath + "/" + "$" + nowStr + "$"
        os.makedirs(path)
        path += "/"
        self.path = path

    def copyParameters(self):
        # Copy the simulation, NN and RL parameters so that we can load them later on
        shutil.copy("model/networkParameters.py", self.path)
        shutil.copy("model/parameters.py", self.path)

    def addDataFilesToDictionary(self):
        self.dataFiles.update({"Error": "tdErrors.txt", "Reward": "rewards.txt"})
        # Add files for each bot
        for bot_idx, bot in enumerate([i for i in self.bots if str(i) == "NN"]):
            botName = str(bot) + str(bot_idx)
            massFileName = "meanMassOverTimeNN" + str(bot_idx) + ".txt"
            self.dataFiles.update({botName  + "_mass": massFileName})
            qvalueFileName = "meanQValuesOverTimeNN" + str(bot_idx) + ".txt"
            self.dataFiles.update({botName  + "_qValue": qvalueFileName})


    def saveModels(self, path, end = False):
        savedTypes = []
        for bot in self.bots:
            botType = bot.getType()
            if botType != "Greedy" and botType not in savedTypes:
                bot.saveModel(path)
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
        name_of_file = self.path + "/model_specifications.txt"
        savedTypes = []
        with open(name_of_file, "w") as file:
             for bot in self.bots:
                botType = bot.getType()
                if botType != "Greedy" and botType not in savedTypes:
                    data = self.getRelevantModelData(bot, end)
                    file.write(data)
                    file.write("\n")
                    savedTypes.append(botType)

        name_of_file = self.path + "networkParameters.py"
        text = "ALGORITHM = \"" + str(self.bots[0].getLearningAlg()) + "\"\n"
        lines = open(name_of_file, 'r').readlines()
        lines[1] = text
        out = open(name_of_file, 'w')
        out.writelines(lines)
        out.close()


    def save(self, end = False):
        self.exportData()
        self.plotTDError()
        self.plotMassesOverTime()
        self.plotQValuesOverTime()

    def exportData(self):
        path = self.path
        # Mean TD error export
        subPath = path + self.dataFiles["Error"]
        with open(subPath, "a") as f:
            for value in self.meanErrors:
                f.write("%s\n" % value)
        self.meanErrors = []
        # Mean Reward export
        subPath = path + self.dataFiles["Reward"]
        with open(subPath, "a") as f:
            for value in self.meanRewards:
                f.write("%s\n" % value)
        self.meanRewards = []
        # Bot exports
        for bot_idx, bot in enumerate([bot for bot in self.bots if str(bot) == "NN"]):
            # Masses export
            subPath = path + self.dataFiles[str(bot)+str(bot_idx)+"_mass"]
            with open(subPath, "a") as f:
                for value in bot.getMassOverTime():
                    f.write("%s\n" % value)
            bot.resetMassList()
            # Qvalues export
            subPath = path + self.dataFiles[str(bot)+str(bot_idx)+"_qValue"]
            learnAlg = bot.getLearningAlg()
            with open(subPath, "a") as f:
                for value in learnAlg.getQValues():
                    f.write("%s\n" % value)
            learnAlg.resetQValueList()


    def getRelevantModelData(self, bot, end = False):
        parameters = bot.parameters
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
        data += "Algorithm - " + str(bot.getLearningAlg()) + "\n"
        data += "Epsilon - " + str(parameters.EPSILON) + "\n"
        data += "Discount factor - " + str(parameters.DISCOUNT) + "\n"
        data += "Frame skip rate - " + str(parameters.FRAME_SKIP_RATE) + "\n"
        data += "State representation - " + ("Grid" if parameters.GRID_VIEW_ENABLED else "Simple") + "\n"
        if parameters.GRID_VIEW_ENABLED:
            data += "Grid - " + str(parameters.GRID_SQUARES_PER_FOV) + "x" +  str(parameters.GRID_SQUARES_PER_FOV)  + "\n"
        data += "Experience Replay - " + ("Enabled" if parameters.EXP_REPLAY_ENABLED else "Disabled") + "\n"
        data += "Target Network steps until update - " + str(parameters.TARGET_NETWORK_MAX_STEPS) + "\n"

        data += "Name of model that was loaded - " + (bot.modelName if bot.modelName else "None") + "\n"
        data += "\n"
        # ANN:
        data += "ANN:\n"
        data += "Input layer neurons(stateReprLen) - " + str(parameters.STATE_REPR_LEN) + "\n"
        data += "First hidden layer neurons - " + str(parameters.HIDDEN_LAYER_1) + "\n"
        data += "Second hidden layer neurons - " + str(parameters.HIDDEN_LAYER_2) + "\n"
        data += "Third hidden layer neurons - " + str(parameters.HIDDEN_LAYER_3) + "\n"
        data += "Output layer neurons(number of actions) - " + str() + "\n"
        data += "Learning rate - " + str(parameters.ALPHA) + "\n"
        data += "Activation function hidden layer(s) - " + parameters.ACTIVATION_FUNC_HIDDEN + "\n"
        data += "Activation function output layer - " + parameters.ACTIVATION_FUNC_OUTPUT + "\n"
        data += "Optimizer - " + str(parameters.OPTIMIZER) + "\n"
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
            if len(self.rewards) > 0:
                 print("Avg reward   last 100 steps:", round(recentMeanReward, 4), " Min: ", round(min(self.rewards),4), " Max: ", round(max(self.rewards), 4))
            if len(self.tdErrors) > 0:
                 print("Avg abs TD-Error last 100 steps: ", round(recentMeanTDError, 4), " Min: ", round(min(self.tdErrors),4), " Max: ", round(max(self.tdErrors), 4))
            print("Step: ", self.counter)
            print("Number of stored rewards: ", len(self.rewards))
            self.printBotStds()
            self.printBotMasses()
            print(" ")
            self.rewards = []
            self.tdErrors = []

    def printBotStds(self):
        for bot in self.bots:
            if bot.getType() == "NN" and str(bot.getLearningAlg()) == "AC":
                print("Std dev: " , bot.getLearningAlg().std)
                break

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
        numberOfPoints = 2000
        errorListPath = self.path + self.dataFiles["Error"]
        with open(errorListPath, 'r') as f:
            errorList = list(map(float, f))

        rewardListPath = self.path + self.dataFiles["Reward"]
        with open(rewardListPath, 'r') as f:
            rewardList = list(map(float, f))
        timeAxis = list(range(0, len(errorList)*numberOfPoints, numberOfPoints))
        plt.plot(timeAxis, errorList, label="Absolute TD-Error")
        plt.xlabel("Time")
        plt.ylabel("TD error averaged")
        plt.savefig(path + "TD-Errors.pdf")
        plt.plot(timeAxis, rewardList, label="Reward")
        plt.legend()
        plt.ylabel("Running avg")
        plt.savefig(path + "Reward_and_TD-Error.pdf")
        plt.close()

    def plotMassesOverTime(self):
        for bot_idx, bot in enumerate([bot for bot in self.bots]):
            massListPath = self.path + self.dataFiles[str(bot) + str(bot_idx) + "_mass"]
            with open(massListPath, 'r') as f:
                massList = list(map(float, f))
            meanMass = round(numpy.mean(massList),1)
            medianMass = round(numpy.median(massList),1)
            varianceMass = round(numpy.std(massList), 1)
            maxMass = round(max(massList), 1)
            len_masses = len(massList)
            playerName = str(bot.getPlayer())
            plt.plot(range(len_masses), massList)
            plt.title("Mass of " + playerName + "- Mean: " + str(meanMass) + " Median: " + str(medianMass) + " Std: " +
                      str(varianceMass) + " Max: " + str(maxMass))
            plt.xlabel("Step")
            plt.ylabel("Total Player Mass")
            plt.savefig(self.path + "MassOverTime" + playerName + ".pdf")
            plt.close()

    def plotQValuesOverTime(self):
        for bot_idx, bot in enumerate([bot for bot in self.bots if bot.getType() == "NN"]):
            qValueListPath = self.path + self.dataFiles["NN" + str(bot_idx) + "_qValue"]
            f = open(qValueListPath, 'r')
            q = f.readlines()
            qValueList = []
            timeAxis = []
            for i in range(len(q)):
                if q[i] != "nan\n":
                    qValueList.append(float(q[i]))
                    timeAxis.append(i)
            f.close()
            # qValueList_filtered = [i for i in qValueList if i != float("NaN")]
            meanQValue = round(numpy.mean(qValueList), 1)
            playerName = str(self.bots[bot_idx].getPlayer())
            plt.plot(timeAxis, qValueList)
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

    def createBot(self, type, learningAlg = None, parameters = None, modelName = None):
        name = type + str(len(self.bots))
        newPlayer = self.createPlayer(name)
        bot = Bot(newPlayer, self.field, type, self.trainingEnabled, learningAlg, parameters, modelName)
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

    def setPath(self, path):
        self.path = path

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

    def getDataFiles(self):
        return self.dataFiles

    # MVC related method
    def register_listener(self, listener):
        self.listeners.append(listener)

    def notify(self):
        for listener in self.listeners:
            listener()
