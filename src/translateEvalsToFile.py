
def addEvals(lines):
    defaults =[]
    lastSpaces = 0
    for line in lines:
        if len(line) < 3 or (line[0] != "" and line[0][0] == "#"):
            continue
        words = line.split(" ")
        spaces = 0
        otherWordAppeared = False
        for idx, word in enumerate(words):
            if word == "" and not otherWordAppeared:
                spaces += 1
                continue
            elif  word == "Default":
                defaults.append((words[idx] + "=" + words[idx + 1], spaces))
            otherWordAppeared = True



        if spaces < lastSpaces:
            for default in defaults[:]:
                if default[2] > spaces:
                    defaults.remove(default)
        lastSpaces = spaces

        line = getEvaluatedLine(line, defaults)


if __name__ == '__main__':
    path = "ParameterTuningFiles/"
    print("Parameter files:")
    fileList = os.listdir(path)
    for file in fileList:
        print(file)
    name = ""
    while name  not in fileList:
        name = input("In which file do you want to copy the evaluations?\n")
    paramFileName = path + name
    f = open(paramFileName, "r")
    lines = f.readlines()
    f.close()
    addEvals(lines)