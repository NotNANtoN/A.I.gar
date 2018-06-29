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
from model.rgbGenerator import *

import linecache
import os
import shutil
import tracemalloc
import pickle as pkl

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
    def __init__(self, guiEnabled, viewEnabled, parameters, trainingEnabled, testingModel = False):
        self.listeners = []
        self.viewEnabled = viewEnabled
        self.guiEnabled = guiEnabled
        self.parameters = parameters
        self.virusEnabled = parameters.VIRUS_SPAWN
        self.resetLimit = parameters.RESET_LIMIT
        self.trainingEnabled = trainingEnabled
        self.path = None
        self.superPath = None
        self.startTime = None
        self.isTestingModel = testingModel

        self.players = []
        self.bots = []
        self.humans = []
        self.playerSpectator = None
        self.spectatedPlayer = None
        self.field = Field(self.virusEnabled)
        self.screenWidth = None
        self.screenHeight = None
        self.counter = 0
        self.timings = []
        self.rewards = []
        self.tdErrors = []
        self.dataFiles = {}
        self.pointAveraging = parameters.EXPORT_POINT_AVERAGING

        if __debug__:
            tracemalloc.start()


    def modifySettings(self, reset_time):
        self.resetLimit = reset_time

    def initialize(self, modelHasBeenLoaded):
        print("Initializing model...")
        if self.trainingEnabled:
            for bot in self.bots:
                bot.saveInitialModels(self.path)
            if not modelHasBeenLoaded:
                self.saveSpecs()
                for bot in self.bots:
                    if bot.getType() == "NN":
                        data = self.getRelevantModelData(bot)
                        print(data)
                        break
        self.field.initialize()
        self.resetBots()


    def loadModel(self, path):
        self.setPath(path)
        now = datetime.datetime.now()
        self.startTime = now

    def resetModel(self):
        print("Resetting field and players!")
        self.field.reset()
        self.resetBots()

    def update(self):
        self.counter += 1
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

        # Store debug info and display progress
        if self.trainingEnabled and "NN" in [bot.getType() for bot in self.bots]:
            self.storeRewardsAndTDError()
            self.timings.append(time.process_time() - timeProcessStart)

        if self.trainingEnabled and self.counter % 1000 == 0:
            self.visualize()

        # Save the models occasionally in case the program crashes at some point
        if self.trainingEnabled and self.counter != 0 and self.counter % max(self.resetLimit, 2500) == 0:
            self.saveSpecs()
            self.saveModels()
            if self.counter != 0 and self.counter % max(self.resetLimit*5, 12500) == 0:
                self.save()


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
        if len(rewards) > 0 and len(errors) > 0:
            self.rewards.append(numpy.mean(rewards))
            self.tdErrors.append(numpy.mean(errors))


    def resetStoredValues(self):
        self.rewards = []
        self.tdErrors = []
        for bot in self.bots:
            bot.resetMassList()
            if bot.type == "NN":
                learnAlg = bot.getLearningAlg()
                learnAlg.resetQValueList()


    def takeBotActions(self):
        for bot in self.bots:
            bot.makeMove()

    def resetBots(self):
        for bot in self.bots:
            bot.reset()


    def initModelFolder(self, name = None, loadedModelName = None, model_in_subfolder = None):
        if name is None:
            self.createPath()
        else:
            if loadedModelName is None:
                self.createNamedPath(name)
            else:
                if model_in_subfolder:
                    self.createNamedLoadPath(name, loadedModelName)
                else:
                    self.createLoadPath(loadedModelName)
        self.copyParameters(loadedModelName)


    def createPath(self):
        basePath = "savedModels/"
        if not os.path.exists(basePath):
            os.makedirs(basePath)
        now = datetime.datetime.now()
        self.startTime = now
        nowStr = now.strftime("%b-%d_%H:%M")
        path = basePath + "$" + nowStr + "$"
        # Also display seconds in name if we already have a model this minute
        if os.path.exists(path):
            nowStr = now.strftime("%b-%d_%H:%M:%S")
            path = basePath + "$" + nowStr + "$"
        os.makedirs(path)
        path += "/"
        print("Path: ", path)
        self.path = path

    def countLoadDepth(self, loadedModelName):
        if loadedModelName[-3] == ")" and loadedModelName[-6:-4] == "(l":
            loadDepth = int(loadedModelName[-4]) + 1
        else:
            loadDepth = 1
        loadString = "_(l" + str(loadDepth) + ")"
        return loadString

    def createLoadPath(self, loadedModelName):
        loadDepth = self.countLoadDepth(loadedModelName)
        basePath = "savedModels/"
        if not os.path.exists(basePath):
            os.makedirs(basePath)
        now = datetime.datetime.now()
        self.startTime = now
        nowStr = now.strftime("%b-%d_%H:%M")
        path = basePath + "$" + nowStr + loadDepth + "$"
        # Also display seconds in name if we already have a model this minute
        if os.path.exists(path):
            nowStr = now.strftime("%b-%d_%H:%M:%S")
            path = basePath + "$" + nowStr + loadDepth + "$"
        os.makedirs(path)
        path += "/"
        print("Path: ", path)
        self.path = path

    def countNamedLoadDepth(self, superName, loadedModelName):
        char = -3
        while loadedModelName[char] != "/":
            char -= 1
        if loadedModelName[char-1] == ")" and loadedModelName[char-4:char-2] == "(l":
            loadDepth = int(loadedModelName[char-2]) + 1
        else:
            loadDepth = 1
        loadString = "_(l" + str(loadDepth) + ")"
        superName = superName[0:len(superName)-1] + loadString + "/"
        return superName

    def createNamedLoadPath(self, superName, loadedModelName):
        superName = self.countNamedLoadDepth(superName, loadedModelName)
        basePath = "savedModels/"
        if not os.path.exists(basePath):
            os.makedirs(basePath)
        # Create subFolder for given parameter tweaking
        osPath = os.getcwd() + "/" + superName
        time.sleep(numpy.random.rand())
        if not os.path.exists(osPath):
            os.makedirs(osPath)
        # Create folder based on name
        now = datetime.datetime.now()
        self.startTime = now
        nowStr = now.strftime("%b-%d_%H:%M:%S:%f")
        path = superName + "$" + nowStr + "$"
        time.sleep(numpy.random.rand())
        if os.path.exists(path):
            randNum = numpy.random.randint(100000)
            path = superName + "$" + nowStr + "-" + str(randNum) + "$"
        os.makedirs(path)
        path += "/"
        print("Super Path: ", superName)
        print("Path: ", path)
        self.path = path

    def createNamedPath(self, superName):
        #Create savedModels folder
        basePath = "savedModels/"
        if not os.path.exists(basePath):
            os.makedirs(basePath)
        #Create subFolder for given parameter tweaking
        osPath = os.getcwd() + "/" + superName
        time.sleep(numpy.random.rand())
        if not os.path.exists(osPath):
            os.makedirs(osPath)
        #Create folder based on name
        now = datetime.datetime.now()
        self.startTime = now
        nowStr = now.strftime("%b-%d_%H:%M:%S:%f")
        path = superName  + "$" + nowStr + "$"
        time.sleep(numpy.random.rand())
        if os.path.exists(path):
            randNum = numpy.random.randint(100000)
            path = superName + "$" + nowStr + "-" + str(randNum) + "$"
        os.makedirs(path)
        path += "/"
        print("Super Path: ", superName)
        print("Path: ", path)
        self.path = path

    def copyParameters(self, loadedModelName = None):
        # Copy the simulation, NN and RL parameters so that we can load them later on
        if loadedModelName is None:
            shutil.copy("model/networkParameters.py", self.path)
            shutil.copy("model/parameters.py", self.path)
        else:
            shutil.copy(loadedModelName + "networkParameters.py", self.path)
            shutil.copy(loadedModelName + "parameters.py", self.path)
            if os.path.exists(loadedModelName + "model.h5"):
                shutil.copy(loadedModelName + "model.h5", self.path)
            if os.path.exists(loadedModelName + "actor_model.h5"):
                shutil.copy(loadedModelName + "actor_model.h5", self.path)
            if os.path.exists(loadedModelName + "value_model.h5"):
                shutil.copy(loadedModelName + "value_model.h5", self.path)


    def addDataFilesToDictionary(self):
        self.dataFiles.update({"Error": "tdErrors.txt", "Reward": "rewards.txt"})
        # Add files for each bot
        for bot_idx, bot in enumerate(self.bots):
            botName = str(bot)
            massFileName = "meanMassOverTime" + botName + ".txt"
            self.dataFiles.update({botName  + "_mass": massFileName})
            qvalueFileName = "meanQValuesOverTime" + botName + ".txt"
            self.dataFiles.update({botName  + "_qValue": qvalueFileName})


    def saveModels(self, end = False):
        savedTypes = []
        for bot in self.bots:
            botType = bot.getType()
            if botType == "NN" and botType not in savedTypes:
                bot.saveModel(self.path)
                savedTypes.append(botType)
        if end:
            path = self.path[:-1]
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
        print("SAVING MODEL")
        self.exportData()
        self.resetStoredValues()
        self.plotTDError()
        self.plotMassesOverTime()
        if self.resetLimit != 0:
            self.plotMassesOverTimeClean()
        self.plotQValuesOverTime()


    def exportData(self):
        path = self.path + "data/"
        if not os.path.exists(path):
            os.mkdir(path)
        # Mean TD error export
        subPath = path + self.dataFiles["Error"]
        with open(subPath, "a") as f:
            for value in range(0,len(self.tdErrors),self.pointAveraging):
                meanError = numpy.mean(self.tdErrors[value:(value + self.pointAveraging)])
                f.write("%s\n" % meanError)
        # self.meanErrors = []
        # Mean Reward export
        subPath = path + self.dataFiles["Reward"]
        with open(subPath, "a") as f:
            for value in range(0,len(self.rewards),self.pointAveraging):
                meanReward = numpy.mean(self.rewards[value:(value + self.pointAveraging)])
                f.write("%s\n" % meanReward)
        # self.meanRewards = []
        # Bot exports
        for bot_idx, bot in enumerate(self.bots):
            # Masses export
            subPath = path + self.dataFiles[str(bot)+"_mass"]
            massList = bot.getMassOverTime()
            with open(subPath, "a") as f:
                for value in range(0, len(massList), self.pointAveraging):
                    meanMass = numpy.mean(massList[value:(value + self.pointAveraging)])
                    f.write("%s\n" % meanMass)
            # Qvalues export
            if bot.type == "NN":
                subPath = path + self.dataFiles[str(bot)+"_qValue"]
                qList = bot.getLearningAlg().getQValues()
                interval = int(self.pointAveraging / (bot.getFrameSkipRate() + 1))
                with open(subPath, "a") as f:
                    for value in range(0, len(qList), interval):
                        qValuesInRange = [i for i in qList[value:(value + interval)] if not math.isnan(i)]
                        meanQ = numpy.mean(qValuesInRange) if len(qValuesInRange) > 0 else float("NaN")

                        f.write("%s\n" % meanQ)


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
        data += "CNN representation - " + ("True" if parameters.CNN_REPR else "False") + "\n"

        if parameters.GRID_VIEW_ENABLED:
            if parameters.CNN_REPR:
                if parameters.CNN_USE_L1:
                    data += "Grid - " + str(parameters.CNN_INPUT_DIM_1) + "x" + str(
                        parameters.CNN_INPUT_DIM_1) + "\n"
                elif parameters.CNN_USE_L2:
                    data += "Grid - " + str(parameters.CNN_INPUT_DIM_2) + "x" + str(
                        parameters.CNN_INPUT_DIM_2) + "\n"
                else:
                    data += "Grid - " + str(parameters.CNN_INPUT_DIM_3) + "x" + str(
                        parameters.CNN_INPUT_DIM_3) + "\n"
            else:
                data += "Grid - " + str(parameters.GRID_SQUARES_PER_FOV) + "x" +  str(parameters.GRID_SQUARES_PER_FOV)  + "\n"
        data += "Experience Replay - " + ("Enabled" if parameters.EXP_REPLAY_ENABLED else "Disabled") + "\n"
        data += "Target Network steps until update - " + str(parameters.TARGET_NETWORK_STEPS) + "\n"
        data += "\n"
        # ANN:
        data += "ANN:\n"
        data += "Input layer neurons(stateReprLen) - " + str(parameters.STATE_REPR_LEN) + "\n"
        data += "Output layer neurons(number of actions) - " + str() + "\n"
        data += "Learning rate - " + str(parameters.ALPHA) + "\n"
        data += "Activation function hidden layer(s) - " + parameters.ACTIVATION_FUNC_HIDDEN + "\n"
        data += "Activation function output layer - " + parameters.ACTIVATION_FUNC_OUTPUT + "\n"
        data += "Optimizer - " + str(parameters.OPTIMIZER) + "\n"
        return data

    def printBotMasses(self):
        for bot in self.bots:
            mass = bot.getPlayer().getTotalMass()
            print("Mass of ", bot.getPlayer(), ": ", round(mass, 1) if mass is not None else "Dead")


    def visualize(self):
        print(" ")
        print("Avg time since update start for the last ", self.pointAveraging, " steps: ", str(round(numpy.mean(self.timings[len(self.timings) - self.pointAveraging:]), 3)))
        #if len(self.rewards) > 0:
        #     print("Avg reward   last 100 steps:", round(recentMeanReward, 4), " Min: ", round(min(self.rewards),4), " Max: ", round(max(self.rewards), 4))
        #if len(self.tdErrors) > 0:
        #     print("Avg abs TD-Error last 100 steps: ", round(recentMeanTDError, 4), " Min: ", round(min(self.tdErrors),4), " Max: ", round(max(self.tdErrors), 4))
        print("Step: ", self.counter)
        if self.trainingEnabled:
            print("Noise level: ", round(self.getCurrentNoise(),4))
        self.printBotMasses()
        print(" ")

    def getCurrentNoise(self):
        for bot in self.bots:
            if bot.getType() == "NN":
                return bot.getLearningAlg().getNoise()
        return None

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
        errorListPath = self.path + "data/" + self.dataFiles["Error"]
        with open(errorListPath, 'r') as f:
            errorList = list(map(float, f))

        rewardListPath = self.path + "data/" + self.dataFiles["Reward"]
        with open(rewardListPath, 'r') as f:
            rewardList = list(map(float, f))
        timeAxis = list(range(0, len(errorList) * self.pointAveraging, self.pointAveraging))
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
        for bot_idx, bot in enumerate(self.bots):
            massListPath = self.path + "data/" +  self.dataFiles[str(bot) + "_mass"]
            with open(massListPath, 'r') as f:
                massList = list(map(float, f))
            meanMass = round(numpy.mean(massList),1)
            medianMass = round(numpy.median(massList),1)
            varianceMass = round(numpy.std(massList), 1)
            maxMass = round(max(massList), 1)
            len_masses = len(massList)
            playerName = str(bot.getPlayer())
            timeAxis = list(range(0, len_masses * self.pointAveraging, self.pointAveraging))

            plt.plot(timeAxis, massList)
            plt.title("Mass of " + playerName + "- Mean: " + str(meanMass) + " Median: " + str(medianMass) + " Std: " +
                      str(varianceMass) + " Max: " + str(maxMass))
            plt.xlabel("Step")
            plt.ylabel("Total Player Mass")
            plt.savefig(self.path + "MassOverTime" + playerName + ".pdf")
            plt.close()


    def plotMassesOverTimeClean(self):
        for bot_idx, bot in enumerate(self.bots):
            massListPath = self.path + "/data/" + self.dataFiles[str(bot) + "_mass"]
            with open(massListPath, 'r') as f:
                massList = list(map(float, f))
            avg_step = int(self.resetLimit / self.pointAveraging)
            if avg_step == 0:
                avg_step = int(len(massList) / 10) + 1
            massList = [numpy.mean(massList[idx:idx+avg_step]) for idx in range(0, len(massList), avg_step)]
            meanMass = round(numpy.mean(massList),1)
            medianMass = round(numpy.median(massList),1)
            varianceMass = round(numpy.std(massList), 1)
            maxMass = round(max(massList), 1)
            playerName = str(bot.getPlayer())
            #timeAxis = list(range(0, len_masses * self.pointAveraging, self.pointAveraging))
            plt.plot(massList)
            plt.title("Mass of " + playerName + "- Mean: " + str(meanMass) + " Median: " + str(medianMass) + " Std: " +
                      str(varianceMass) + " Max: " + str(maxMass))
            plt.xlabel("Episode")
            plt.ylabel("Total Player Mass")
            plt.savefig(self.path + "CleanMassOverTime" + playerName + ".pdf")
            plt.close()


    def plotQValuesOverTime(self):
        for bot_idx, bot in enumerate([bot for bot in self.bots if bot.getType() == "NN"]):
            qValueListPath = self.path + "/data/" +  self.dataFiles[str(bot) + "_qValue"]
            f = open(qValueListPath, 'r')
            q = f.readlines()
            qValueList = []
            timeAxis = []
            for i in range(len(q)):
                if q[i] != "nan\n":
                    qValueList.append(float(q[i]))
                    timeAxis.append(i * self.pointAveraging)
            f.close()
            # qValueList_filtered = [i for i in qValueList if i != float("NaN")]
            meanQValue = round(numpy.mean(qValueList), 1)
            playerName = str(self.bots[bot_idx].getPlayer())
            # timeAxis = list(range(0, len(qValueList) * self.pointAveraging, self.pointAveraging))
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

    def createBot(self, botType, learningAlg = None, parameters = None):
        name = botType + str(len(self.bots))
        newPlayer = self.createPlayer(name)
        rgbGenerator = None
        if parameters is not None:
            if parameters.CNN_REPR and parameters.CNN_P_REPR:
                rgbGenerator = RGBGenerator(self.field, parameters)
        bot = Bot(newPlayer, self.field, botType, self.trainingEnabled, learningAlg, parameters, rgbGenerator)
        self.addBot(bot)

    def createHuman(self, name):
        newPlayer = self.createPlayer(name)
        self.addHuman(newPlayer)

    def addPlayer(self, player):
        self.players.append(player)
        self.field.addPlayer(player)

    def addBot(self, bot):
        self.bots.append(bot)
        player = bot.getPlayer()
        if player not in self.players:
            self.addPlayer(player)

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
    def getNNBot(self):
        for bot in self.bots:
            if bot.getType() == "NN":
                return bot

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

    def getPointAveraging(self):
        return self.pointAveraging

    def getParameters(self):
        return self.parameters

    def getVirusEnabled(self):
        return self.virusEnabled

    # MVC related method
    def register_listener(self, listener):
        self.listeners.append(listener)

    def notify(self):
        for listener in self.listeners:
            listener()
