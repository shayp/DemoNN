__author__ = 'shay-macbook'
from Network import *
from numpy import *
from random import *
import matplotlib.pyplot as plt
import csv
import pickle


def main():
    #testSpiralClassification('DATA_TRAIN.csv', 'DATA_valid.csv', 'Netowrk.bin')
    RunSerializedNetwork('DATA_TRAIN.csv', 'DATA_valid.csv', 'Netowrk.bin')


def RunSerializedNetwork(trainPath, testPath, networkPath):
    pkl_file = open(networkPath, 'rb')
    net = pickle.load(pkl_file)
    # Get the data from CSV
    trainData, testData = getTrainAndTest(trainPath, testPath)

    # Test the network on the train data(for getting train error)
    testTrainingValue = net.test(trainData)

    print ' ************** Train results ******************'
    PrintResults(trainData, testTrainingValue)

    # Test the network on new data
    testValues = net.test(testData)

    print ' ************** Test results ******************'
    PrintResults(testData, testValues)


def PrintResults(realData, predictedData):
    # Define train parameters
    TP = 0.0
    FP = 0.0
    FN = 0.0
    TN = 0.0

    for i in range(0, len(realData)):

        # Set threshold for classification
        if predictedData[i] >= 0.5:
           predictedData[i] = 1
        else:
            predictedData[i] = 0

        # Set error/success for each example
        if (realData[i][2] == 1 and predictedData[i] == 1):
            TP = TP + 1
        elif (realData[i][2] == 1 and predictedData[i] == 0):
            FP = FP + 1
        if (realData[i][2] == 0 and predictedData[i] == 1):
            FN = FN + 1
        else:
            TN = TN + 1

    # Calculate metrics
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    ACC = (TP + TN) / (TP + TN + FP + FN)
    F1 = (2 * TP) / (2 * TP + FP + FN)

    # Print metrics
    print 'True positive rate: ' + repr(TPR * 100.0) + '%'
    print 'False positive rate: ' + repr(FPR* 100.0) + '%'
    print 'accuracy: ' + repr(ACC * 100.0) + '%'
    print 'F1: ' + repr(F1 * 100.0) + '%'

def testSpiralClassification(trainFileName, testFileName, networkPath):

    # Define network parameters
    activationFunction = 'tanh'
    eta = 0.01
    numOfLayers = 2
    maxNeuronsInLayer = 11
    inputSize = 2
    outputSize = 1
    learningMethod = 'decent'
    numOfEpoch = 40000
    numOfMiniBatch = 5
    L2regularizationFactor = 0.0000001
    L1regularizationFactor = 0.0

    momentumFactor = 0.7

    # Get the data from CSV
    trainData, testData = getTrainAndTest(trainFileName, testFileName)

    # Define the network
    net = Network(activationFunction, eta, learningMethod, numOfEpoch, numOfMiniBatch, inputSize, outputSize, maxNeuronsInLayer, numOfLayers,L2regularizationFactor,L1regularizationFactor, momentumFactor)

    # Train the network
    net.train(trainData)

    # Print network weights after training
    net.getNetwork()

    # Test the network on the train data(for getting train error)
    testTrainingValue = net.test(trainData)

    print ' ************** Train results ******************'
    PrintResults(trainData, testTrainingValue)

    # Test the network on new data
    testValues = net.test(testData)

    print ' ************** Test results ******************'
    PrintResults(testData, testValues)

    plt.figure(1)
    plt.subplot(211)

    for i in range(0, len(testData)):
        # plot the real spiral values in the top box
        if testData[i][2] == 1:
            plt.plot(testData[i][0], testData[i][1], 'r.')
        else:
            plt.plot(testData[i][0], testData[i][1], 'b.')

    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title('Real spiral')

    plt.subplot(212)

    # plot the classification of the network for the spiral
    for i in range(0, len(testData)):
        if testValues[i] == 1:
            plt.plot(testData[i][0], testData[i][1], 'r.')
        else:
            plt.plot(testData[i][0], testData[i][1], 'b.')

    plt.show()

    fileObject = open(networkPath, 'wb')
    pickle.dump(net, fileObject)



def getTrainAndTest(csvTrain, csvTest):
    train = []
    test = []
    f = open(csvTrain, 'rb')

    reader = csv.reader(f)
    for row in reader:
        train.append([float(row[0]), float(row[1]), float(row[2])])

    f = open(csvTest, 'rb')
    reader = csv.reader(f)
    for row in reader:
        test.append([float(row[0]),float(row[1]), float(row[2])])

    return train, test

if __name__ == "__main__":
    main()