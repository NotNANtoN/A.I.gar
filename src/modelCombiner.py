import os
import matplotlib
matplotlib.use('Pdf')
import matplotlib.pyplot as plt
import numpy as np
import fnmatch
import math
import numpy
import scipy.stats

POINT_AVERAGING = 500
RESET_INTERVAL = 10000


def createCombinedModelGraphs(path):
    print("###############################")
    print("Generating average plots:\n")
    if path[-1] != "/":
        path = path + "/"
    print("Path: ", path)
    plotTDErrorAndMean(path)
    plotMassesOverTime(path)
    plotQValuesOverTime(path)
    plotTestingMassOverTime(path)
    plotFinalTests(path)
    print("Combining test results..")
    combineTestResults(path)
    print("###############################")

def combineTestResults(path):
    modelList = [i for i in os.listdir(path) if os.path.isdir(path + i)]

    evaluations = {}
    keyList = []
    for model in modelList:
        modelPath = path + model + "/"
        for file in os.listdir(modelPath):
            if not fnmatch.fnmatch(file, 'final_results*'):
                continue
            resultsPath = modelPath + file
            with open(resultsPath, 'r') as f:
                for line in f.readlines():
                    words = line.split(sep=" ")
                    # Only check for evaluation lines
                    if len(words) == 11:
                        name = words[0]
                        maxScore = float(words[2])
                        meanScore = float(words[4])
                        stdMean = float(words[6])
                        meanMaxScore = float(words[8])
                        stdMax = float(words[10])
                        evaluation = (name, maxScore, meanScore, stdMean, meanMaxScore, stdMax)
                        # Create entry if it is not there yet
                        try:
                            evaluations[name]
                        except KeyError:
                            evaluations[name] = []
                            keyList.append(name)
                        evaluations[name].append(evaluation)

    name_of_file = path + "/combined_final_results.txt"
    with open(name_of_file, "w") as file:
        data = ""
        for key in keyList:
            evalList = evaluations[key]
            name = key
            maxScore = str(round(max([evaluation[1] for evaluation in evalList]), 1))
            meanScore = str(round(numpy.mean([evaluation[2] for evaluation in evalList]), 1))
            stdMean = str(round(numpy.mean([evaluation[3] for evaluation in evalList]), 1))
            meanMaxScore = str(round(numpy.mean([evaluation[4] for evaluation in evalList]), 1))
            stdMax = str(round(numpy.mean([evaluation[5] for evaluation in evalList]), 1))
            data += name + " Highscore: " + maxScore + " Mean: " + meanScore + " StdMean: " + stdMean \
                    + " Mean_Max_Score: " + meanMaxScore + " Std_Max_Score: " + stdMax + "\n"
        file.write(data)


def plotTDErrorAndMean(path):
    modelList = [i for i in os.listdir(path) if os.path.isdir(path + i)]

    allErrorLists = []
    allRewardLists = []
    maxLength = 0
    for model in modelList:
        print(model)
        modelPath = path + model + "/data/"
        errorListPath = modelPath + "tdErrors.txt"
        if not os.path.exists(errorListPath):
            print("-- Model does not have tdError.txt --")
            continue
        with open(errorListPath, 'r') as f:
            errorList = list(map(float, f))
            if len(errorList) > maxLength:
                maxLength = len(errorList)
            #errorList = np.array(errorList)
            allErrorLists.append(errorList)

        rewardListPath = modelPath + "rewards.txt"
        if not os.path.exists(rewardListPath):
            print("-- Model does not have rewards.txt --")
            continue
        with open(rewardListPath, 'r') as f:
            rewardList = list(map(float, f))
            if len(rewardList) > maxLength:
                maxLength = len(rewardList)
            #rewardList = np.array(rewardList)
            allRewardLists.append(rewardList)
        if len(allErrorLists) != len(allRewardLists):
            print("Error lists and reward lists don't match.")
            quit()

    errorLabels = {"meanLabel": "Mean Error", "sigmaLabel": '$\sigma$ range', "xLabel": "Step number",
              "yLabel": "TD error mean value", "title": "TD Error", "path": path, "subPath": "Mean_TD-Error"}
    rewardLabels = {"meanLabel": "Mean Reward", "sigmaLabel": '$\sigma$ range', "xLabel": "Step number",
                   "yLabel": "Reward mean value", "title": "Reward", "path": path, "subPath": "Mean_Reward"}

    # Combined plot
    plot(allErrorLists, maxLength, POINT_AVERAGING, errorLabels, allRewardLists, rewardLabels)

    # Single plot Error
    plot(allErrorLists, maxLength, POINT_AVERAGING, errorLabels)

    # Single plot Reward
    plot(allRewardLists, maxLength, POINT_AVERAGING, rewardLabels)


def plotFinalTests(path):
    modelList = [i for i in os.listdir(path) if os.path.isdir(path + i)]

    allMassesOverTime = {}
    allMaxLengths = {}

    for model in modelList:
        modelPath = path + model + "/data/"
        if not os.path.exists(modelPath):
            print("-- ", modelPath, " does not exist. --")
            continue
        for file in os.listdir(modelPath):
            if not fnmatch.fnmatch(file, "*Mean_Mass_*") or not fnmatch.fnmatch(file, "*.txt"):
                continue

            testingMassPath = modelPath + file

            meanMassIdx = file.find("Mean_Mass")
            file = file[meanMassIdx:]
            # Create empty list if it is not yet there
            try:
                allMassesOverTime[file]
            except KeyError:
                allMassesOverTime[file] = []

            print(file)

            with open(testingMassPath, 'r') as f:
                meanVals = list(map(float, f))
                allMaxLengths[file] = len(meanVals)
                allMassesOverTime[file].append(meanVals)


    for test in allMassesOverTime:
        print("Keyword: ", test)
        masses = allMassesOverTime[test]
        maxLength = allMaxLengths[test]

        labels = {"meanLabel": "Mean Reward", "sigmaLabel": '$\sigma$ range', "xLabel": "Training steps",
                  "yLabel": "Mass Mean Value", "title": test[:-4], "path": path,
                  "subPath": "Final_testing_" + test[:-4]}
        plot(masses, maxLength, 1, labels, showConfInt = True)


def plotTestingMassOverTime(path):
    modelList = [i for i in os.listdir(path) if os.path.isdir(path + i)]

    allMeans = []
    maxLength = 0
    for model in modelList:
        modelPath = path + model + "/data/"
        if not os.path.exists(modelPath):
            print("-- ", modelPath, " does not exist. --")
            continue
        for file in os.listdir(modelPath):
            if not fnmatch.fnmatch(file, "testMassOverTime.txt"):
                continue
            testingMassPath = modelPath + file

            with open(testingMassPath, 'r') as f:
                meanVals = list(map(float, f))
                maxLength = len(meanVals)
                allMeans.append(meanVals)

    labels = {"meanLabel": "Mean Reward", "sigmaLabel": '$\sigma$ range', "xLabel": "Training Time (%)",
              "yLabel": "Testing Mass Mean Value", "title": "Mass", "path": path, "subPath": "Mean_Testing_Mass_During_training"}
    plot(allMeans, maxLength, 10, labels, showConfInt=True)


def plotMassesOverTime(path):
    modelList = [i for i in os.listdir(path) if os.path.isdir(path + i)]

    allMassList = []
    maxLength = 0
    massListPresent = False
    for model in modelList:
        print(model)
        modelPath = path + model + "/data/"
        if not os.path.exists(modelPath):
            print("-- ", modelPath, " does not exist. --")
            continue
        for file in os.listdir(modelPath):
            if not fnmatch.fnmatch(file, 'meanMassOverTimeNN*'):
                continue
            print(file)
            massListPath = modelPath + file
            massListPresent = True

            with open(massListPath, 'r') as f:
                massList = list(map(float, f))
                if len(massList) > maxLength:
                    maxLength = len(massList)
                allMassList.append(massList)

    if not massListPresent:
        print("-- Model does not have any meanMassOverTimeNN(i).txt --")
        return

    labels = {"meanLabel": "Mean Reward", "sigmaLabel": '$\sigma$ range', "xLabel": "Step number",
              "yLabel": "Mass mean value", "title": "Mass", "path": path, "subPath": "Mean_Mass"}
    plot(allMassList, maxLength, POINT_AVERAGING, labels)

    episodeSize = int(numpy.ceil(RESET_INTERVAL / POINT_AVERAGING))
    cleanAllMassList = []
    for i in range(len(allMassList)):
        cleanMassList = []
        for t in range(0,maxLength,episodeSize):
            if len(allMassList[i]) <= t:
                continue
            episodeMean = np.mean(allMassList[i][t:(t + episodeSize)])
            cleanMassList.append(episodeMean)
        cleanAllMassList.append(cleanMassList)

    labels = {"meanLabel": "Mean Reward", "sigmaLabel": '$\sigma$ range', "xLabel": "Step number",
              "yLabel": "Mass mean value", "title": "Mass", "path": path, "subPath": "Clean_Mean_Mass"}
    plot(cleanAllMassList, int(numpy.floor(maxLength/episodeSize)), RESET_INTERVAL, labels)



def plotQValuesOverTime(path):
    modelList = [i for i in os.listdir(path) if os.path.isdir(path + i)]

    allQValueList = []
    maxLength = 0
    massListPresent = False
    for model in modelList:
        print(model)
        modelPath = path + model + "/data/"
        if not os.path.exists(modelPath):
            print("-- ", modelPath, " does not exist. --")
            continue
        for file in os.listdir(modelPath):
            if not fnmatch.fnmatch(file, 'meanQValuesOverTimeNN*'):
                continue
            print(file)
            massListPath = modelPath + file
            massListPresent = True

            with open(massListPath, 'r') as f:
                massList = list(map(float, f))
                if len(massList) > maxLength:
                    maxLength = len(massList)
                allQValueList.append(massList)

    if not massListPresent:
        print("-- Model does not have any meanMassOverTimeNN(i).txt --")
        return

    labels = {"meanLabel": "Mean Q-value", "sigmaLabel": '$\sigma$ range', "xLabel": "Step number",
              "yLabel": "Q-value mean value", "title": "Q-value", "path": path, "subPath": "Mean_QValue" }
    plot(allQValueList, maxLength, POINT_AVERAGING, labels)

def getTimeAxis(maxLength, avgLength):
    return np.array(list(range(0, maxLength * avgLength, avgLength)))


def getMeanAndStDev(allList, maxLength):
    print("Shape of allList: ", numpy.shape(allList))
    mean_list = []
    stDev_list = []
    for i in range(maxLength):
        summed = 0
        num_lists = 0
        for j in range(len(allList)):
            if i >= len(allList[j]):
                continue
            summed += allList[j][i]
            num_lists += 1
        if num_lists == 0:
            print("Number of lists is 0 for idx ", i, " and max length ", maxLength)
            quit()
        m = summed / num_lists
        mean_list.append(m)
        sqe = 0
        for j in range(len(allList)):
            if i >= len(allList[j]):
                continue
            sqe += (allList[j][i] - m)**2
        sd = math.sqrt(sqe / num_lists)
        stDev_list.append(sd)
    mean_list = np.array(mean_list)
    stDev_list = np.array(stDev_list)
    return mean_list, stDev_list


def plot(ylist, maxLength, avgLength, labels, y2list=None, labels2=None, showConfInt = True):
    print("Plotting ", labels["title"], "...")
    x = getTimeAxis(maxLength, avgLength)
    y, ysigma = getMeanAndStDev(ylist, maxLength)
    if y is None or ysigma is None:
        print(labels["title"] + " list is None.")
        return
    y_lower_bound = y - ysigma
    y_upper_bound = y + ysigma

    matplotlib.pyplot.ticklabel_format(axis='x', style='sci', scilimits=(1, 4))
    plt.clf()
    fig, ax = plt.subplots(1)
    ax.plot(x, y, lw=2, label=labels["meanLabel"], color='blue')
    ax.fill_between(x, y_lower_bound, y_upper_bound, facecolor='blue', alpha=0.5,
                    label=labels["sigmaLabel"])
    ax.set_xlabel(labels["xLabel"])
    yLabel = labels["yLabel"]
    title = labels["title"].replace("_", " ")
    path = labels["path"] + "/" + labels["subPath"]
    if y2list is not None:
        y2, y2sigma = getMeanAndStDev(y2list, maxLength)
        y2_lower_bound = y2 - y2sigma
        y2_upper_bound = y2 + y2sigma
        ax.plot(x, y2, lw=2, label=labels2["meanLabel"], color='red')
        ax.fill_between(x, y2_lower_bound, y2_upper_bound, facecolor='red', alpha=0.5,
                        label=labels2["sigmaLabel"])
        yLabel += "\n" + labels2["yLabel"]
        title += " & " + labels2["title"]
        path += "_and_" + labels2["subPath"]

    meanY = numpy.mean(y)


    ax.legend(loc='upper left')
    ax.set_ylabel(yLabel)
    if showConfInt:
        meanOfRuns = [numpy.mean(run) for run in ylist]
        meanVal = np.mean(meanOfRuns)
        stdVal = scipy.stats.sem(meanOfRuns)
        confInt = scipy.stats.t.interval(0.95, len(meanOfRuns) - 1, loc=meanVal,
                                         scale=stdVal)
        title = title + " averaged over " + str(len(meanOfRuns)) +  " runs. Mean: " + str(round(meanY,1)) +\
                " Std: " + str(round(stdVal, 1)) + "\nConfInt95%: "+ str(np.round(confInt, 1))


    else:
        title = title + " mean value (" + str(round(meanY, 1)) + ") $\pm$ $\sigma$ interval"
    ax.set_title(title)
    ax.grid()
    fig.savefig(path + ".pdf")
    plt.close()



if __name__ == '__main__':
    basePath = "savedModels"
    if not os.path.exists(basePath):
        print("savedModels folder does not exist. Quitting...")
        quit()

    for folder in [i for i in os.listdir(basePath)if str(i)[0] != "$"]:
        subPath = basePath + "/" + folder
        if str(folder)[0] != "$":
            print("\n")
            print(folder)
            createCombinedModelGraphs(subPath)


