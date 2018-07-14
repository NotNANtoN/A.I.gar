import os
import numpy

from scipy.stats import ttest_ind

def getAllDirectories():
    allDirs = []
    path = "savedModels/"
    fileList = os.listdir(path)

    for directory in fileList:
        if os.path.isdir(path + directory):
            allDirs.append(directory)

    return allDirs


def findDefaults():
    defaults = [directory for directory in  getAllDirectories() if "Default" in directory]

    if len(defaults) == 0:
        print("")
        print("No default models found!")
        quit()

    print("Number of defaults: ", len(defaults))

    return defaults


def normalizeArray(arr):
    if arr is None or len(arr) < 1:
        return arr
    maxNum = max(arr)
    if maxNum == 0:
        return arr
    for idx in range(len(arr)):
        arr[idx] /= maxNum
    return arr


def addMeanToDictionary(dictionary, name, path, takeMean):
    f = open(path)
    massList = list(map(float, f))
    if takeMean:
        if name == "test":
            massList = normalizeArray(massList)
        meanVal = numpy.mean(massList)
    else:
        meanVal = massList
    try:
        dictionary[name]
    except KeyError:
        dictionary[name] = []
    dictionary[name].append(meanVal)
    f.close()


def translateToSuperDict(subDict, superDict, takeMean):
    for key in subDict:
        values = subDict[key]
        if takeMean:
            meanVal = numpy.mean(values)
        else:
            meanVal = values
        try:
            superDict[key]
        except KeyError:
            superDict[key] = []
        superDict[key].append(meanVal)


def getDistributions(directories, takeMean = True):
    distributions = []

    for directory in directories:
        directory_distributions = {"name":directory}
        path = "savedModels/" + directory + "/"
        print("Directory: ", path)
        subfolders = os.listdir(path)
        for subfolder in subfolders:
            if not os.path.isdir(path + subfolder):
                continue
            dataPath = path + subfolder + "/data/"
            localDict = {}
            if os.path.exists(dataPath):
                data = os.listdir(dataPath)
            else:
                data = []
            for dataFile in data:
                if "MassOverTime.txt" in dataFile:
                    massIdx = dataFile.find("Mass")
                    name = dataFile[:massIdx]
                    addMeanToDictionary(directory_distributions, name, dataPath + dataFile, takeMean)
                if "Mean_Mass_" in dataFile:
                    meanMassIdx = dataFile.find("Mean_Mass_")
                    name = dataFile[meanMassIdx + 10:-4]
                    addMeanToDictionary(localDict, name, dataPath + dataFile, takeMean)
            translateToSuperDict(localDict, directory_distributions, takeMean)
        distributions.append(directory_distributions)

    return distributions


def createEvaluationFiles(all_dists, default_dists, compareAll):
    basePath = "savedModels/"

    for distribution_set in all_dists:
        data = ""
        currentName = distribution_set["name"]
        path = basePath + currentName
        print("path: ", path)

        for key in distribution_set:
            if key == "name":
                continue
            meanValList = distribution_set[key]
            overallMeanVal = numpy.mean(meanValList)
            overallStd = numpy.std(meanValList)
            meanString = str(int(overallMeanVal)) if not key == "test" else str(round(overallMeanVal,2))
            stdString = str(int(overallStd)) if not key == "test" else str(round(overallStd,2))
            data += key + " mean value " + meanString + " and std " + stdString + "\n"

            for default_dist_set in default_dists:
                default_name = default_dist_set["name"]
                if currentName == default_name:
                    continue
                defaultIdx = default_name.find("Default")
                startDefault = default_name[:defaultIdx]
                print("default start: ", startDefault)
                print("name start: ", currentName[:defaultIdx])
                if startDefault != "" and currentName[:defaultIdx] != startDefault:
                    if not compareAll:
                        continue


                try:
                    defaultMeanValList = default_dist_set[key]
                    shortenedDefault = default_name[defaultIdx:]
                    if "&" in shortenedDefault:
                        andIdx = shortenedDefault.find("&")
                        shortenedDefault = shortenedDefault[8:andIdx]
                    else:
                        shortenedDefault = shortenedDefault[8:]
                    overallDefaultMeanVal = numpy.mean(defaultMeanValList)
                    tVal, pVal = ttest_ind(meanValList, defaultMeanValList)
                    shortMeanVal = str(int(overallDefaultMeanVal)) if not key == "test" else str(round(overallDefaultMeanVal,2))
                    comparison = " better " if overallMeanVal > overallDefaultMeanVal else " worse "
                    data += "Has a " + str(round((1 - pVal) * 100, 1)) + "%\t probability to be" + comparison + "than " \
                            + shortenedDefault + "'s mean of " + shortMeanVal + "\n"
                except KeyError:
                    continue
                except RuntimeWarning:
                    pass

            data += "\n"
        evalFile = open(path + "/eval.txt", "w+")
        evalFile.write(data)
        evalFile.close()


if __name__ == '__main__':
    compareAll = int(input("Compare all defaults? (1==yes)"))

    defaults = findDefaults()

    default_distributions = getDistributions(defaults) # in the form of dictionaries

    allDirectories = getAllDirectories()

    all_distributions = getDistributions(allDirectories)

    createEvaluationFiles(all_distributions, default_distributions, compareAll)

