import os
import subprocess
import math
import time

def filterLines(lines):
    for idx, line in enumerate(lines[:]):
        words = line.split(" ")
        if len(words) < 3 or (len(words) > 0 and len(words[0]) > 0 and words[0][0] == "#"):
            lines.remove(line)
    return lines

def checkCorrectParamNames(lines):
    netParamFile = open("model/networkParameters.py")
    netParamLines = netParamFile.readlines()
    netParamNames = [line.split(" ")[0] for line in netParamLines]
    for idx, line in enumerate(lines):
        words = line.split(" ")
        for word in words:
            if len(word) > 0:
                paramName = word
                break

        print("Param name: ", paramName)
        if not (paramName == "Default" or paramName in netParamNames):
            print("")
            print("A parameter name in the file does not exist!")
            print("Line: ", line)
            print()
            netParamFile.close()
            quit()
    netParamFile.close()


def getJobs(lines):
    paramTestNum = None
    jobs = []
    paramStack = []
    valueStack = []
    for idx, line in enumerate(lines):
        words = line.split(" ")
        print(words)

        otherWordAppeared = False
        depthCount = 0
        for word in words[:]:
            if word == "":
                if not otherWordAppeared:
                    depthCount += 1
                words.remove(word)
            else:
                otherWordAppeared = True

        if depthCount > len(paramStack):
            print("Something went wrong with line: ", line)
            print("The depth (number of spaces at start) is bigger than the depth")
            quit()

        while depthCount < len(paramStack):
            del paramStack[-1]
            del valueStack[-1]

        if len(words) != 3:
            print("Line is not correctly formatted: ", line)
            quit()

        paramName =  words[0]
        paramVal =  words[1]
        if words[2][:-1] == "&":
            paramTestNum = None
        elif words[2][-3:] == "x:\n":
            paramTestNum =  int(words[2][:-3]) #remove the "x:\n"
        else:
            print("Error in reading line: ", line)
            print("Third word is neither a & nor does it indicate the number of tests.")
            quit()

        if paramTestNum is not None:
            jobs.append((paramStack + [paramName],valueStack + [paramVal], paramTestNum))
        else:
            paramStack.append(paramName)
            valueStack.append(paramVal)

    return jobs


def displayJobs(jobs):
    for idx, job in enumerate(jobs):
        print("")
        print("Job ", idx + 1, ":")
        for paramIdx in range(len(job[0])):
            print(job[0][paramIdx], "=", job[1][paramIdx])
        print("Will be run ", job[2], " times.")
    print("")


def runJobs(jobs):
    sampleJobScriptFile = open("anton.sh", "r")
    sampleLines = sampleJobScriptFile.readlines()

    timeLineBase = sampleLines[1][:15]
    outputNameLineBase = sampleLines[6][:17]
    sampleJobScriptFile.close()

    standardTime = 7.5  # hours for 500k steps for standard Q-learning without other bots

    for idx, job in enumerate(jobs):
        paramData = ""
        outputName = ""
        timeBotFactor = 1
        timeStepFactor = 1
        timeOtherFactor = 1
        resetTime = 15000
        algorithmType = 0
        memoryLimit = 20000
        for paramIdx in range(len(job[0])):
            paramName = job[0][paramIdx]
            paramVal = job[1][paramIdx]
            paramData += "1\n"
            paramData += paramName + "\n"
            paramData += paramVal + "\n"

            outputName += paramName + "-" + paramVal.replace(".", "_") + "_"

            if paramName == "NUM_NN_BOTS":
                timeBotFactor *= int(paramVal * 2)
                resetTime = 30000
            elif paramName == "NUM_GREEDY_BOTS" and int(paramVal) > 0:
                resetTime = 30000
                timeBotFactor *= (1 + 0.2 * int(paramVal))
            elif paramName == "MAX_TRAINING_STEPS":
                timeStepFactor *= int(paramVal) / 500000
            elif paramName == "ACTOR_CRITIC_TYPE":
                algorithmType = 2
            elif paramName == "USE_ACTION_AS_INPUT":
                timeOtherFactor *= 4
            elif paramName == "ACTOR_CRITIC_TYPE":
                if paramVal == "\"DPG\"":
                    timeOtherFactor *= 1.25
                elif paramVal == "\"CACLA\"":
                    timeOtherFactor *= 1.1
            elif paramName == "CNN_REPRESENTATION":
            	memoryLimit = 120000

        jobTime = math.ceil(standardTime * timeBotFactor * timeStepFactor * timeOtherFactor)
        days = jobTime // 24
        hours = jobTime % 24

        timeLine = timeLineBase + str(days) + "-"
        timeLine += str(hours) if hours >= 10 else "0" + str(hours)
        timeLine += ":00:00\n"
        outputNameLine = outputNameLineBase + outputName + "%j.out\n"
        resetLine = str(resetTime) + "\n"
        algorithmLine = str(algorithmType) + "\n"
        fileName = outputName[:-1] + ".sh"
        script = open(fileName, "w+")

        data = "#!/bin/bash\n"\
               + timeLine \
               + "#SBATCH --mem=" + str(memoryLimit) + "\n#SBATCH --nodes=1\n#SBATCH --mail-type=ALL\n#SBATCH --mail-user=antonwiehe@gmail.com\n"\
               + outputNameLine\
               + "module load matplotlib/2.1.2-foss-2018a-Python-3.6.4\nmodule load TensorFlow/1.6.0-foss-2018a-Python-3.6.4\n" \
               + "module load h5py/2.7.1-foss-2018a-Python-3.6.4\npython -O ./aigar.py <<EOF\n" \
               + "0\n0\n" + resetLine + "0\n" +algorithmLine + paramData +"0\n0\n1\nEOF\n"
        script.write(data)
        script.close()

        for jobNum in range(job[2]):
            try:
                subprocess.call(["sbatch" , fileName])
            except FileNotFoundError:
                script.close()
                print("Command sbatch not found or filename invalid!")
                print("Filename: ", fileName)

        os.remove(fileName)

        print("Submitted job: ", fileName)
        time.sleep(0.2)

if __name__ == '__main__':
    path = "ParameterTuningFiles/"
    print("Parameter files:")
    fileList = os.listdir(path)
    for file in fileList:
        print(file)
    name = ""
    while name  not in fileList:
        name = input("What file do you want to open?\n")
    f = open(path + name, "r")
    lines = f.readlines()
    f.close()
    lines = filterLines(lines)
    checkCorrectParamNames(lines)

    jobs = getJobs(lines)

    displayJobs(jobs)
    jobSum = 0
    for job in jobs:
        jobSum += job[2]
    confirm = input("Do you really want to submit " + str(len(jobs)) +  " parameter tunings to create " +  str(jobSum) + " jobs? (y==yes)\n")
    if confirm == "y":
        runJobs(jobs)
        try:
            subprocess.call(["squeue", "-u", "s2972301"])
        except FileNotFoundError:
            print("I tried to give you an overview of the jobs, but it did not work :(")
