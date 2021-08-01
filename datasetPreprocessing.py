import pandas as pd
import numpy as np


def calculateZScore(arr):
    # Z-score
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    x_train = (arr - mean) / (std + 1e-7)
    return x_train

def getPimaDataset():
    filename = "dataset/diabetes.csv"
    data = pd.read_csv(filename)
    #print(data.columns)

    x = data.values
    x = np.delete(x,8,1)
    x = calculateZScore(x)

    y = data['Outcome'].tolist()
    y = np.array(y)

    count=[0,0]
    for i in y:
        if y[i]==0:
            count[0]+=1
        else:
            count[1] += 1

    print(count)


    return x,y


def getIonosphereDataset():
    filename = "dataset/ionosphere.data"
    f = open(filename,"r")
    x = []
    y = []

    for i in f.readlines():
        arr = i[:-1]
        arr = arr.split(sep=",")

        temp = arr[0:34]
        temp = [float(j) for j in temp]
        x.append(temp)

        if(arr[34]=="g"):
            y.append(1)
        else:
            y.append(0)


    x = np.array(x)
    x = calculateZScore(x)
    y = np.array(y)

    count = [0, 0]
    for i in y:
        if y[i] == 0:
            count[0] += 1
        else:
            count[1] += 1

    print(count)

    return x,y


"""
x,y = getIonosphereDataset()

print(x)
print(y)
print(x.shape)
print(y.shape)
"""