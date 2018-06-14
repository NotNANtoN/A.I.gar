import os
import sys
import importlib
from controller.controller import Controller
from model.model import *
from model.network import Network
from model.qLearning import *
from model.nsSarsa import *
from model.expectedSarsa import *
from model.treeBackup import *
from model.parameters import *
from model.actorCritic import *
from view.view import View
from modelCombiner import createCombinedModelGraphs, plot, getMeanAndStDev

import numpy as np
import tensorflow as tf
import random as rn


def fix_seeds(seedNum):
    # The below is necessary in Python 3.2.3 onwards to
    # have reproducible behavior for certain hash-based operations.
    # See these references for further details:
    # https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
    # https://github.com/keras-team/keras/issues/2280#issuecomment-306959926

    # import os
    # os.environ['PYTHONHASHSEED'] = '0'

    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.

    if seedNum is not None:
        np.random.seed(42)
    else:
        np.random.seed()

    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.

    # rn.seed(12345)

    # Force TensorFlow to use single thread.
    # Multiple threads are a potential source of
    # non-reproducible results.
    # For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res

    if seedNum is not None:

        session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

        from keras import backend as K

        # The below tf.set_random_seed() will make random number generation
        # in the TensorFlow backend have a well-defined initial state.
        # For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed

        tf.set_random_seed(seedNum)

        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(sess)
    else:
        session_conf = tf.ConfigProto()

        from keras import backend as K

        # The below tf.set_random_seed() will make random number generation
        # in the TensorFlow backend have a well-defined initial state.
        # For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed

        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(sess)


def setSeedAccordingToFolderNumber(model_in_subfolder, loadModel, modelPath, enableTrainMode):
    seedNumber = None
    if model_in_subfolder and not loadModel:
        folders = [i for i in os.listdir(modelPath) if os.path.isdir(modelPath + "/" + i)]
        seedNumber = len(folders)
    if seedNumber and enableTrainMode:
        fix_seeds(seedNumber)


def algorithmNumberToName(val):
    if val == 0:
        return "Q-Learning"
    elif val == 1:
        return "n-step Sarsa"
    elif val == 2:
        return "CACLA"
    elif val == 3:
        return "Discrete ACLA"
    elif val == 4:
        return "Expected Sarsa"
    elif val == 5:
        return "Tree Backup"
    else:
        print("Wrong algorithm selected...")
        quit()


def algorithmNameToNumber(name):
    if name == "Q-learning":
        return 0
    elif name == "n-step Sarsa":
        return 1
    elif name == "AC":
        return 2
    elif name == "Discrete ACLA":
        return 3
    elif name == "Expected Sarsa":
        return 4
    elif name == "Tree Backup":
        return 5
    else:
        print("ALGORITHM in networkParameters not found.\n")
        quit()


def checkValidParameter(param):
    name_of_file = "model/networkParameters.py"
    lines = open(name_of_file, 'r').readlines()
    for n in range(len(lines)):
        name = ""
        for char in lines[n]:
            if char == " ":
                break
            name += char
        if param == name:
            print("FOUND")
            return n
    print("Parameter with name " + tweakedParameter + "not found.")
    quit()


def modifyParameterValue(tweaked, model):
    name_of_file = model.getPath() + "networkParameters.py"
    lines = open(name_of_file, 'r').readlines()
    for i in range(len(tweaked)):
        text = ""
        for char in lines[tweaked[i][2]]:
            text += char
            if char == "=":
                break
        text += " " + str(tweaked[i][1]) + "\n"
        lines[tweaked[i][2]] = text
    out = open(name_of_file, 'w')
    out.writelines(lines)
    out.close()


def nameSavedModelFolder(array):
    name = ""
    for i in range(len(array)):
        if i != 0:
            name += "&"
        name += array[i][0] + "=" + str(array[i][1]).replace('.', '_')
    name += '/'
    return name


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
        return int(SCREEN_WIDTH * humansNr + humansNr - 1), int(SCREEN_HEIGHT)
    if humansNr == 3:
        return int(SCREEN_WIDTH * humansNr * 2 / 3) + humansNr - 1, int(SCREEN_HEIGHT * 2 / 3)

    return SCREEN_WIDTH, SCREEN_HEIGHT


def createHumans(numberOfHumans, model1):
    for i in range(numberOfHumans):
        name = input("Player" + str(i + 1) + " name:\n")
        model1.createHuman(name)


def createBots(number, model, botType, parameters, algorithm=None, loadModel=None):
    learningAlg = None
    loadPath = model.getPath() if loadModel else None
    if botType == "NN":
        Bot.num_NNbots = number
        networks = {}
        for i in range(number):
            # Create algorithm instance
            if algorithm == 0:
                learningAlg = QLearn(numberOfNNBots, numberOfHumans, parameters)
            elif algorithm == 1:
                learningAlg = nsSarsa(numberOfNNBots, numberOfHumans, parameters)
            elif algorithm == 2:
                learningAlg = ActorCritic(parameters)
            elif algorithm == 3:
                learningAlg = ExpectedSarsa(numberOfNNBots, numberOfHumans, parameters)
            elif algorithm == 4:
                learningAlg = TreeBackup(numberOfNNBots, numberOfHumans, parameters)
            else:
                print("Please enter a valid algorithm.\n")
                quit()
            networks = learningAlg.initializeNetwork(loadPath, networks)
            model.createBot(botType, learningAlg, parameters)
    elif botType == "Greedy":
        Bot.num_Greedybots = number
        for i in range(number):
            model.createBot(botType, None, parameters)


def testModel(testingModel, n_training, reset_time, modelPath, name, plotting=True):
    masses = []
    meanMasses = []
    maxMasses = []
    bot = testingModel.getNNBot()
    print("Testing ", name, "...")
    testingModel.initialize(False)
    # viewTestModel = View(testModel, screenWidth, screenHeight)
    # viewTestModel.draw()
    plotPath = modelPath
    modelPath += "data/"
    if not os.path.exists(modelPath):
        os.mkdir(modelPath)
    for test in range(n_training):
        bot.resetMassList()
        testingModel.resetModel()
        for updateStep in range(reset_time):
            testingModel.update()

        massOverTime = bot.getMassOverTime()
        meanMass = numpy.mean(massOverTime)
        maxMass = numpy.max(massOverTime)
        masses.append(massOverTime)
        meanMasses.append(meanMass)
        maxMasses.append(maxMass)
        print("Mean mass for run ", test + 1, ": ", meanMass)
        if plotting:
            exportTestResults(massOverTime, modelPath, "Run_" + str(test + 1) + "_Mean_Mass_" + name)
        #else:
        #    exportTestResults(massOverTime, modelPath, name + "_run_" + str(test + 1))

    meanScore = numpy.mean(meanMasses)
    stdMean = numpy.std(meanMasses)
    meanMaxScore = numpy.mean(maxMasses)
    stdMax = numpy.std(maxMasses)
    maxScore = numpy.max(maxMasses)
    if plotting:
        meanMassPerTimeStep = []
        for timeIdx in range(reset_time):
            val = 0
            for listIdx in range(n_training):
                val += masses[listIdx][timeIdx]
            meanVal = val / n_training
            meanMassPerTimeStep.append(meanVal)

        #exportTestResults(meanMassPerTimeStep, modelPath, "Mean_Mass_" + name)
        labels = {"meanLabel": "Mean Mass", "sigmaLabel": '$\sigma$ range', "xLabel": "Step number",
                  "yLabel": "Mass mean value", "title": "Mass plot test phase", "path": plotPath,
                  "subPath": "Mean_Mass_" + name}
        plot(masses, reset_time, 1, labels)
    return name, maxScore, meanScore, stdMean, meanMaxScore, stdMax


def cloneModel(model):
    clone = Model(False, False, model.getParameters(), False)
    for bot in model.getBots():
        clone.createBot(bot.getType(), bot.getLearningAlg(), bot.parameters)
    return clone


def updateTestResults(testResults, model, percentage):
    originalNoise = model.getNNBot().getLearningAlg().getNoise()
    clonedModel = cloneModel(model)
    model.getNNBot().getLearningAlg().setNoise(0)

    if str(clonedModel.getNNBot().getLearningAlg()) != "AC":
        originalTemp = clonedModel.getNNBot().getLearningAlg().getTemperature()
        clonedModel.getNNBot().getLearningAlg().setTemperature(0)

    currentEval = testModel(clonedModel, 5, 10000 if not clonedModel.resetLimit else clonedModel.resetLimit,
                            model.getPath(), "test", False)
    clonedModel.getNNBot().getLearningAlg().setNoise(originalNoise)

    if str(clonedModel.getNNBot().getLearningAlg()) != "AC":
        clonedModel.getNNBot().getLearningAlg().setTemperature(originalTemp)

    meanScore = currentEval[2]
    stdDev = currentEval[3]
    testResults.append((meanScore, stdDev))
    return testResults


def exportTestResults(testResults, path, name):
    filePath = path + name + ".txt"
    with open(filePath, "a") as f:
        for val in testResults:
            # write as: "mean\n"
            line = str(val) + "\n"
            f.write(line)


def plotTesting(testResults, path, timeBetween, end):
    x = range(0, end + timeBetween, timeBetween)
    y = [x[0] for x in testResults]
    ysigma = [x[1] for x in testResults]

    y_lower_bound = [y[i] - ysigma[i] for i in range(len(y))]
    y_upper_bound = [y[i] + ysigma[i] for i in range(len(y))]

    plt.ticklabel_format(axis='x', style='sci', scilimits=(1, 4))
    plt.clf()
    fig = plt.gcf()
    ax = plt.gca()
    # fig, ax = plt.subplots(1)
    ax.plot(x, y, lw=2, label="testing mass", color='blue')
    ax.fill_between(x, y_lower_bound, y_upper_bound, facecolor='blue', alpha=0.5,
                    label="+/- sigma")
    ax.set_xlabel("Time")
    yLabel = "Mass"
    title = "Testing mass over time"

    meanY = numpy.mean(y)
    ax.legend(loc='upper left')
    ax.set_ylabel(yLabel)
    ax.set_title(title + " mean value (" + str(round(meanY, 1)) + ") $\pm$ $\sigma$ interval")
    ax.grid()
    print("PATH: ", path)
    fig.savefig(path + title + ".pdf")

    plt.close()


def runTests(model):
    np.random.seed()

    print("Testing...")
    # Set Parameters:
    resetPellet = 15000
    resetGreedy = 30000
    resetVirus = 15000
    n_test_runs = 10
    trainedBot = model.getNNBot()
    trainedAlg = trainedBot.getLearningAlg()
    evaluations = []
    # Pellet testing:
    pelletModel = Model(False, False, model.getParameters(), False)
    pelletModel.createBot("NN", trainedAlg, parameters)
    pelletEvaluation = testModel(pelletModel, n_test_runs, resetPellet, model.getPath(), "pellet_collection")
    evaluations.append(pelletEvaluation)
    # Greedy Testing:
    if len(model.getBots()) > 1:
        greedyModel = Model(False, False, model.getParameters(), False)
        greedyModel.createBot("NN", trainedAlg, parameters)
        greedyModel.createBot("Greedy")
        greedyEvaluation = testModel(greedyModel, n_test_runs, resetGreedy, model.getPath(), "vs_1_greedy")
        evaluations.append(greedyEvaluation)
    # Virus Testing:
    if model.getVirusEnabled():
        virusModel = Model(False, False, model.getParameters(), False)
        virusModel.createBot("NN", trainedAlg, parameters)
        virusEvaluation = testModel(virusModel, n_test_runs, resetVirus, model.getPath(), "virus")
        evaluations.append(virusEvaluation)

    # TODO: add more test scenarios for multiple greedy bots and full model check
    print("Testing completed.")

    name_of_file = model.getPath() + "/final_results.txt"
    with open(name_of_file, "w") as file:
        data = "Avg run time(s): " + str(round(numpy.mean(model.timings), 5)) + "\n"
        data += "Number of runs per testing: " + str(n_test_runs) + "\n"
        for evaluation in evaluations:
            name = evaluation[0]
            maxScore = str(round(evaluation[1], 1))
            meanScore = str(round(evaluation[2], 1))
            stdMean = str(round(evaluation[3], 1))
            meanMaxScore = str(round(evaluation[4], 1))
            stdMax = str(round(evaluation[5], 1))
            data += name + " Highscore: " + maxScore + " Mean: " + meanScore + " StdMean: " + stdMean \
                    + " Mean_Max_Score: " + meanMaxScore + " Std_Max_Score: " + stdMax + "\n"
        file.write(data)


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

    modelName = None
    modelPath = None
    loadedModelName = None
    algorithm = None
    packageName = None
    parameters = None
    model_in_subfolder = False
    loadModel = int(input("Do you want to load a model? (1 == yes)\n"))
    loadModel = (loadModel == 1)
    if loadModel:
        while packageName is None:
            packageName = None
            print("#########################################")
            print("Saved Models: \n")
            for folder in [i for i in os.listdir("savedModels/")]:
                print(folder)
            modelName = input("Enter the model name (name of directory in savedModels): (Empty string == break)\n")
            # If user presses enter, quit model loading
            if str(modelName) == "":
                loadModel = False
                modelName = None
                break
            # If user inputs wrong model name, ask for input again
            modelPath = "savedModels/" + modelName + "/"
            if not os.path.exists(modelPath):
                print("Invalid model name, no model found under ", modelPath)
                continue
            # CHECK FOR SUBFOLDERS
            if str(modelName)[0] != "$":
                while packageName is None:
                    print("------------------------------------")
                    print("Folder Submodels: \n")
                    for folder in [i for i in os.listdir(modelPath) if os.path.isdir(modelPath + "/" + i)]:
                        print(folder)
                    subModelName = input("Enter the submodel name: (Empty string == break)\n")
                    # If user presses enter, leave model
                    if str(subModelName) == "":
                        break
                    subPath = modelPath + subModelName + "/"
                    if not os.path.exists(subPath):
                        print("Invalid model name, no model found under ", subPath)
                        continue
                    packageName = "savedModels." + modelName + "." + subModelName
                    loadedModelName = subPath
                    # modelName = path
                    model_in_subfolder = True
                if packageName is None:
                    continue

            if packageName is None:
                packageName = "savedModels." + modelName
                loadedModelName = modelPath
                # ModelName = None will autogenereate a name
                modelName = None
        if packageName is not None:
            parameters = importlib.import_module('.networkParameters', package=packageName)
            algorithm = algorithmNameToNumber(parameters.ALGORITHM)
            # model.setPath(modelName)

    if not loadModel:
        parameters = importlib.import_module('.networkParameters', package="model")

        algorithm = int(input("What learning algorithm do you want to use?\n" + \
                              "'Q-Learning' == 0, 'n-step Sarsa' == 1, 'Actor-Critic' == 2,\n" + \
                              " 'Tree Backup' == 3, 'Expected Sarsa' == 4\n"))
    tweaking = int(input("Do you want to tweak parameters? (1 == yes)\n"))
    tweakedTotal = []
    if tweaking == 1:
        while True:
            tweakedParameter = str(input("Enter name of parameter to be tweaked:\n"))
            paramLineNumber = checkValidParameter(tweakedParameter)
            if paramLineNumber is not None:
                paramValue = str(input("Enter parameter value:\n"))
                tweakedTotal.append([tweakedParameter, paramValue, paramLineNumber])
            if 1 != int(input("Tweak another parameter? (1 == yes)\n")):
                break
        modelPath = "savedModels/" + nameSavedModelFolder(tweakedTotal)
        model_in_subfolder = True

    if int(input("Give saveModel folder a custom name? (1 == yes)\n")) == 1:
        modelPath = "savedModels/" + str(input("Input folder name:\n"))


    model = Model(guiEnabled, viewEnabled, parameters, True)

    model.initModelFolder(modelPath, loadedModelName, model_in_subfolder)

    numberOfHumans = 0
    mouseEnabled = True
    humanTraining = False
    if guiEnabled and viewEnabled:
        numberOfHumans = int(input("Please enter the number of human players: (" + str(MAXHUMANPLAYERS) + " max)\n"))
        if fitsLimitations(numberOfHumans, MAXHUMANPLAYERS):
            createHumans(numberOfHumans, model)
            if 2 >= numberOfHumans > 0:
                humanTraining = int(input("Do you want to train the network using human input? (1 == yes)\n"))
                mouseEnabled = not humanTraining
            if numberOfHumans > 0 and not humanTraining:
                mouseEnabled = int(input("Do you want control Player1 using the mouse? (1 == yes)\n"))

        if not model.hasHuman():
            spectate = int(input("Do want to spectate an individual bot's FoV? (1 = yes)\n"))
            if spectate == 1:
                model.addPlayerSpectator()


    if tweakedTotal:
        modifyParameterValue(tweakedTotal, model)

    enableTrainMode = humanTraining if humanTraining is not None else False
    if not humanTraining:
        enableTrainMode = int(input("Do you want to train the network?: (1 == yes)\n"))
    model.setTrainingEnabled(enableTrainMode == 1)
    # if enableTrainMode:
    #    maxTrainSteps = str(input("For how many steps do you want to train?\n"))
    #    paramLineNumber = checkValidParameter("MAX_TRAINING_STEPS")
    #    modifyParameterValue([["MAX_TRAINING_STEPS", maxTrainSteps, paramLineNumber]], model)

    parameters = importlib.import_module('.networkParameters', package=model.getPath().replace("/", ".")[:-1])
    numberOfNNBots = parameters.NUM_NN_BOTS
    numberOfGreedyBots = parameters.NUM_GREEDY_BOTS
    numberOfBots = numberOfNNBots + numberOfGreedyBots

    Bot.init_exp_replayer(parameters)

    setSeedAccordingToFolderNumber(model_in_subfolder, loadModel, modelPath, enableTrainMode)

    createBots(numberOfNNBots, model, "NN", parameters, algorithm, loadModel)

    if fitsLimitations(numberOfGreedyBots, 1000):
        createBots(numberOfGreedyBots, model, "Greedy", parameters)

    model.addDataFilesToDictionary()

    if numberOfNNBots == 0:
        model.setTrainingEnabled(False)

    if numberOfBots == 0 and not viewEnabled:
        modelMustHavePlayers()

    model.initialize(loadModel)

    screenWidth, screenHeight = defineScreenSize(numberOfHumans)
    # model.setScreenSize(screenWidth, screenHeight)

    if guiEnabled:
        view = View(model, screenWidth, screenHeight, parameters)
        controller = Controller(model, viewEnabled, view, mouseEnabled)
        view.draw()
        while controller.running:
            controller.process_input()
            model.update()
    else:
        maxSteps = parameters.MAX_SIMULATION_STEPS
        smallPart = max(int(maxSteps / 100), 1)
        testResults = []
        for step in range(maxSteps):
            model.update()
            if step % smallPart == 0 and step != 0:
                print("Trained: ", round(step / maxSteps * 100, 1), "%")
                # Test every 10% of training
            if step % (smallPart * 5) == 0:
                testResults = updateTestResults(testResults, model, round(step / maxSteps * 100, 1))
        testResults = updateTestResults(testResults, model, 100)
        meanMassesOfTestResults = [val[0] for val in testResults]
        exportTestResults(meanMassesOfTestResults, model.getPath() + "/data/", "testMassOverTime")
        plotTesting(testResults, model.getPath(), smallPart * 5, maxSteps)
        print("Training done.")
        print("")

    if model.getTrainingEnabled():
        model.save(True)
        model.saveModels()

        runTests(model)
        if model_in_subfolder:
            print(os.path.join(modelPath))
            createCombinedModelGraphs(os.path.join(modelPath))

        print("Total average time per update: ", round(numpy.mean(model.timings), 5))

        bots = model.getBots()
        for bot_idx, bot in enumerate([bot for bot in model.getBots() if bot.getType() == "NN"]):
            player = bot.getPlayer()
            print("")
            print("Network parameters for ", player, ":")
            attributes = dir(parameters)
            for attribute in attributes:
                if not attribute.startswith('__'):
                    print(attribute, " = ", getattr(parameters, attribute))
            print("")
            print("Mass Info for ", player, ":")
            massListPath = model.getPath() + "/data/" +  model.getDataFiles()["NN" + str(bot_idx) + "_mass"]
            with open(massListPath, 'r') as f:
                massList = list(map(float, f))
            mean = numpy.mean(massList)
            median = numpy.median(massList)
            variance = numpy.std(massList)
            print("Median = ", median, " Mean = ", mean, " Std = ", variance)
            print("")
