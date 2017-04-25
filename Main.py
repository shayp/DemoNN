__author__ = 'shay-macbook'
from Network import *
from numpy import *
from random import *
import matplotlib.pyplot as plt
import csv

def main():
    testSpiralClassification('DATA_TRAIN.csv', 'DATA_valid.csv')

def testSpiralClassification(trainFileName, testFileName):
    activationFunction = 'tanh'
    eta = 0.02
    numOfLayers = 2
    maxNeuronsInLayer = 10
    inputSize = 2
    outputSize = 1
    learningMethod = 'decent'
    numOfEpoch = 2000
    numOfMiniBatch = 5
    trainInput, trainOutput, testInput, testOutput = getTrainAndTest(trainFileName, testFileName)
    net = Network(activationFunction, eta, learningMethod, numOfEpoch, numOfMiniBatch, inputSize, outputSize, maxNeuronsInLayer, numOfLayers)
    plt.figure(1)
    plt.subplot(311)

    net.train(trainInput, trainOutput)
    testValues = net.test(testInput, testOutput)

    for i in range(0,len(testValues)):
        if testValues[i] >= 0.5:
           testValues[i] = 1
        else:
            testValues[i] = 0

    for i in range(0, len(testInput)):
        if testOutput[i] == 1:
            plt.plot(testInput[i][0],testInput[i][1], 'r.')
        else:
            plt.plot(testInput[i][0],testInput[i][1], 'b.')

        if testValues[i] == 1:
            plt.plot(testInput[i][0],testInput[i][1], 'r+')
        else:
            plt.plot(testInput[i][0],testInput[i][1], 'b+')

    plt.show()
def testCosFunction():
    activationFunction = 'tanh'
    eta = 0.02
    numOfLayers = 1
    maxNeuronsInLayer = 10
    inputSize = 1
    outputSize = 1
    learningMethod = 'decent'
    numOfEpoch = 2000
    numOfMiniBatch = 5

    inputs  = []
    outputs = []
    testinputs  = []
    testoutputs = []
    net = Network(activationFunction, eta, learningMethod, numOfEpoch, numOfMiniBatch, inputSize, outputSize, maxNeuronsInLayer, numOfLayers)
    for i in range(0,5000):
        inputs.append(2 * math.pi * uniform(0, 1))
        outputs.append(cos(inputs[i]))
        testinputs.append(2 * math.pi * uniform(0, 1))
        testoutputs.append(cos(testinputs[i]))

    net.train(inputs,outputs)
    testValues = net.test(testinputs, testoutputs)
    net.getNetwork()
    plt.figure(1)
    plt.subplot(311)
    plt.plot(asarray(testinputs), asarray(testoutputs), 'b.', asarray(testinputs), asarray(testValues), 'ro')
    plt.subplot(312)
    plt.plot(asarray(testoutputs), asarray(testValues), 'b.')
    plt.show()

def getTrainAndTest(csvTrain, csvTest):
    trainInput = []
    trainOutput = []
    testInput = []
    testOutput = []
    f = open(csvTrain, 'rb')
    reader = csv.reader(f)
    for row in reader:
        trainInput.append([float(row[0]), float(row[1])])
        trainOutput.append(float(row[2]))

    f = open(csvTest, 'rb')
    reader = csv.reader(f)
    for row in reader:
        testInput.append([float(row[0]),float(row[1])])
        testOutput.append(float(row[2]))

    return trainInput, trainOutput, testInput, testOutput

if __name__ == "__main__":
    main()