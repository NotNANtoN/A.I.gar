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


def runJobs(jobs, email):
    sampleJobScriptFile = open("anton.sh", "r")
    sampleLines = sampleJobScriptFile.readlines()

    timeLineBase = sampleLines[1][:15]
    outputNameLineBase = sampleLines[6][:17]
    sampleJobScriptFile.close()

    standardTime = 13  # hours for 500k steps for standard Q-learning without other bots

    for idx, job in enumerate(jobs):
        paramData = ""
        outputName = ""
        timeBotFactor = 1
        timeStepFactor = 1
        timeOtherFactor = 1
        timeSPGFactor = 1
        algorithmType = 0
        memoryLimit = 20000
        cnn = False
        for paramIdx in range(len(job[0])):
            paramName = job[0][paramIdx]
            paramVal = job[1][paramIdx]
            paramData += "1\n"
            paramData += paramName + "\n"
            paramData += paramVal + "\n"

            outputName += paramName + "-" + paramVal.replace(".", "_") + "_"

            if paramName == "NUM_NN_BOTS":
                timeBotFactor *= (int(paramVal) / 6) + 1
            elif paramName == "NUM_GREEDY_BOTS" and int(paramVal) > 0:
                timeBotFactor *= (1 + 0.2 * int(paramVal))
            elif paramName == "MAX_TRAINING_STEPS":
                timeStepFactor *= int(paramVal) / 500000
            elif paramName == "USE_ACTION_AS_INPUT":
                timeOtherFactor *= 5
            elif paramName == "ACTOR_CRITIC_TYPE":
                algorithmType = 2
                if paramVal == "\"DPG\"":
                    timeOtherFactor *= 2
                elif paramVal == "\"CACLA\"":
                    timeOtherFactor *= 1.2
            elif paramName == "CNN_REPR":
                cnn = True
            elif paramName == "MEMORY_CAPACITY":
                memoryLimit *= int(int(paramVal) / 75000) + 1
            elif paramName == "MEMORY_BATCH_LEN":
                timeOtherFactor *= int(int(paramVal) / 32) + 1
            elif paramName == "GRID_SQUARES_PER_FOV":
                timeOtherFactor *= ((int(paramVal) - 11) / 10) + 1
            elif paramName == "ENABLE_SPLIT":
                timeOtherFactor *= 1.3
            elif "Layers" in paramName:
                timeOtherFactor *= 1.3
            elif paramName == "NUM_ACTIONS":
                timeOtherFactor *= 1.5
            elif paramName == "OCACLA_ENABLED":
                timeSPGFactor = 8
            elif paramName == "OCACLA_EXPL_SAMPLES":
                timeOtherFactor *= int(paramVal) / 4
            elif paramName == "OCACLA_ONLINE_SAMPLES":
                timeOtherFactor *= 3

        jobTime = math.ceil(standardTime * timeBotFactor * timeStepFactor * timeOtherFactor * timeSPGFactor)
        if jobTime > 240:
            jobTime = 240
        days = jobTime // 24
        hours = jobTime % 24

        if cnn:
            timeLine = timeLineBase + "3-08:00:00\n"
            memoryLimit = 40000

        else:
            timeLine = timeLineBase + str(days) + "-"
            timeLine += str(hours) if hours >= 10 else "0" + str(hours)
            timeLine += ":00:00\n"

        outputNameLine = outputNameLineBase + outputName + "%j.out\n"
        algorithmLine = str(algorithmType) + "\n"
        fileName = outputName[:-1] + ".sh"
        script = open(fileName, "w+")

        data = "#!/bin/bash\n"\
               + timeLine \
               + "#SBATCH --mem=" + str(memoryLimit) + "\n#SBATCH --nodes=1\n#SBATCH --mail-type=ALL\n#SBATCH --mail-user=" + email + "\n"\
               + outputNameLine\
               + "module load matplotlib/2.1.2-foss-2018a-Python-3.6.4\nmodule load TensorFlow/1.6.0-foss-2018a-Python-3.6.4\n" \
               + "module load h5py/2.7.1-foss-2018a-Python-3.6.4\npython -O ./aigar.py <<EOF\n" \
               + "0\n0\n" +algorithmLine + paramData +"0\n0\n1\nEOF\n"
        script.write(data)
        script.close()

        print("Job: ", fileName)
        print("Job hours: ", jobTime)
        for jobNum in range(job[2]):
            try:
                subprocess.call(["sbatch" , fileName])
            except FileNotFoundError:
                script.close()
                print("Command sbatch not found or filename invalid!")
                print("Filename: ", fileName)
        
        os.remove(fileName)

        print("Submitted job: ", fileName)
        print()
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

    email = int(input("Emails:\n0 == antonwiehe@gmail.com\n1 == n.stolt.anso@student.rug.nl\nWhat email do you want to use?\n"))
    if email == 0:
        email = "antonwiehe@gmail.com"
    else:
        email = "n.stolt.anso@student.rug.nl"
    print("EMAIL: ", email)
    jobSum = 0
    for job in jobs:
        jobSum += job[2]
    confirm = input("Do you really want to submit " + str(len(jobs)) +  " parameter tunings to create " +  str(jobSum) + " jobs? (y==yes)\n")
    if confirm == "y":
        runJobs(jobs, email)
        try:
            subprocess.call(["squeue", "-u", "s2972301"])
        except FileNotFoundError:
            print("I tried to give you an overview of the jobs, but it did not work :(")
