import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':

    df = pd.read_csv("test.txt", delimiter=";")

    print(df.head())

    processes = [0, 1]
    dataFrames = []
    colors = ["k", "r"]
    processIndices = []

    for proc in processes:
        indexList = df[df["process"] == proc].index.tolist()
        processIndices.append((indexList[0], indexList[-1]))
        copiedDataFrame = df[indexList[0]: indexList[-1]].copy()
        dataFrames.append(copiedDataFrame.sort_values("key"))
        #dataFrames.append(copiedDataFrame.loc[indexList[0], indexList[-1]])

    print(processIndices)

    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #for i, processIndex in enumerate(processIndices):
    #    ax.scatter(df["x"][processIndex[0]:processIndex[1]],
    #               df["y"][processIndex[0]:processIndex[1]],
    #               df["z"][processIndex[0]:processIndex[1]],
    #               c=colors[i])
    #plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i, dataFrame in enumerate(dataFrames):
        ax.scatter(dataFrame["x"],
                   dataFrame["y"],
                   dataFrame["z"],
                   c=colors[i])
    plt.show()



