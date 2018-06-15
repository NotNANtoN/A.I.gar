from jobEvaluater import getAllDirectories

def getDesiredDirectories(desiredDirectories):
    return [directory for directory in getAllDirectories() if directory in desiredDirectories]

if __name__ == '__main__':
    directoriesToPlot = []
    name = input("Type the name of the directory you want to include in the plot: ")
    directoriesToPlot.append(name)
    while input("More? (y==more)\n") == "y":
        directoriesToPlot.append(input("Type in the name: "))

    dirs = getDesiredDirectories(directoriesToPlot)















