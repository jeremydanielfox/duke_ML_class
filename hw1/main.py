# Code written by Jeremy Fox for ML HW 1
# Professor: Cynthia Rudin
# Duke University

import numpy as np
import math

def readFile(filename):
    file = open(FILE_PREFIX + filename, 'r')
    return file.readlines()

def predict(x,w):
    x = np.array(x)
    w = np.array(w)
    return np.sign(x.dot(w))

def preProcess(labels, data):
    # Strip out new lines characters
    labels = map(lambda label: label.rstrip(),labels)
    data = map(lambda d: d.rstrip(), data)
    # Find and mark all places that are not a 0 or 1
    for idx, val in enumerate(labels):
        if not (val == "0" or val == "1"):
            labels[idx] = INVALID
            data[idx] = INVALID
    # Filter out marked values
    labels = filter(lambda a: a != INVALID, labels)
    data = filter(lambda a: a != INVALID, data)
    # Convert labels to numbers
    labels = map(lambda num: int(num), labels)
    # Split data from strings into vectors
    data = map(lambda a: np.array(map(lambda b: float(b), a.split())), data)
    return (data, labels)

def perceptron(data, n):
    w = np.zeros(n)
    while(True):
        gotThrough = True
        for idx, val in enumerate(data):
            x = val[0]
            y = val[1]
            guess = predict(x,w)
            if not ((guess == 1 and y == 1) or (guess == -1 and y == 0)):
                y = 1 if y == 1 else -1
                w = w + y*x
                gotThrough = False
                break
        numCorrect = 0
        if (gotThrough):
            break
    return w

def setupBias(data):
    return map(lambda a: (np.append(a[0],1),a[1]),data)

def reduceTopBottom(datapoint,start,end):
    return datapoint[start:end]

def reduceByFrequency(datapoint,frequencies,k,j):
    first = frequencies[:k]
    last = frequencies[len(frequencies)-j:]
    for val in first:
        idx = val[0]
        datapoint[idx] = -1.0
    for val in last:
        idx = val[0]
        datapoint[idx] = -1.0
    out = np.array(filter(lambda a: a != -1.0, datapoint))
    return out


def dataTransform(data, func):
    return map(lambda a: (func(a[0]),a[1])   ,data)

def processColumnFrequency(data):
    frequencies = [0]*784
    for val in data:
        img = val[0]
        for idx, col in enumerate(img):
            if col != 0:
                frequencies[int(idx)] = frequencies[int(idx)] + 1
    out = list()
    for idx, val in enumerate(frequencies):
        out.append((idx,val))
    return sorted(out, key=lambda val: val[1])

def calculateAccuracy(dataList, w):
    data = list()
    labels = list()
    for val in dataList:
        data.append(val[0])
        labels.append(val[1])
    total = 0.0
    correct = 0.0
    for idx, x in enumerate(data):
        actual = labels[idx]
        guess = predict(x,w)
        total = total + 1
        if (actual == 1 and guess == 1 or actual == 0 and guess == -1):
            correct = correct+1
    accuracy = correct/total
    print "accuracy: " + str(accuracy)

def makeTupleList(inData,inLabels):
    data = list()
    for idx, val in enumerate(inLabels):
        data.append((inData[idx],val))
    return data

def test(trainData,testData,func):
    trainData = dataTransform(trainData, func)
    trainData = setupBias(trainData)
    n = trainData[0][0].size
    print "n=" + str(n)
    w = perceptron(trainData,n)
    testData = dataTransform(testData, func)
    testData = setupBias(testData)
    calculateAccuracy(testData, w)




FILE_PREFIX = "mnist/"
INVALID = "-1"
# Import the data
trainLabels = readFile("mnist_train_labels.txt")
trainData = readFile("mnist_train.txt")
processed = preProcess(trainLabels,trainData)
trainData = processed[0]
trainLabels = processed[1]

# Package the data up into a list of tuples

data = makeTupleList(trainData,trainLabels)

testLabels = readFile('mnist_test_labels.txt')
testData = readFile('mnist_test.txt')
processed = preProcess(testLabels, testData)
testData = makeTupleList(processed[0],processed[1])
# Calculate accuracy
test(data, testData, lambda a: reduceTopBottom(a,3*28,784-13*28))
for i in range(11):
    print "*******************************"
    print "i= " + str(i)
    test(data,testData, lambda a: reduceTopBottom(a,i*28,783-i*28))
print "************************************"
freq = processColumnFrequency(data)
test(data,testData, lambda a: reduceByFrequency(a,freq,625,40))

