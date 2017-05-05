__author__ = 'shay-macbook'
from Network import *
from numpy import *
from random import *
import matplotlib.pyplot as plt
import csv

def main():
    testSpiralClassification('DATA_TRAIN.csv', 'DATA_valid.csv')

def testSpiralClassification(trainFileName, testFileName):

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
    regularizationFactor = 0.0000001
    momentumFactor = 0.7

    # Get the data from CSV
    trainData, testData = getTrainAndTest(trainFileName, testFileName)

    # Define the network
    net = Network(activationFunction, eta, learningMethod, numOfEpoch, numOfMiniBatch, inputSize, outputSize, maxNeuronsInLayer, numOfLayers,regularizationFactor, momentumFactor)

    # Train the network
    net.train(trainData)

    # Print network weights after training
    net.getNetwork()

    # Test the network on the train data(for getting train error)
    testTrainingValue = net.test(trainData)

    # Define train parameters
    TrainTP = 0.0
    TrainFP = 0.0
    TrainFN = 0.0
    TrainTN = 0.0

    for i in range(0, len(testTrainingValue)):

        # Set threshold for classification
        if testTrainingValue[i] >= 0.5:
           testTrainingValue[i] = 1
        else:
            testTrainingValue[i] = 0

        # Set error/success for each example
        if (trainData[i][2] == 1 and testTrainingValue[i] == 1):
            TrainTP = TrainTP + 1
        elif (trainData[i][2] == 1 and testTrainingValue[i] == 0):
            TrainFP = TrainFP + 1
        if (trainData[i][2] == 0 and testTrainingValue[i] == 1):
            TrainFN = TrainFN + 1
        else:
            TrainTN = TrainTN + 1

    # Calculate metrics
    TrainTPR = TrainTP / (TrainTP + TrainFN)
    TrainFPR = TrainFP / (TrainFP + TrainTN)
    TrainACC = (TrainTP + TrainTN) / (TrainTP + TrainTN + TrainFP + TrainFN)
    TrainF1 = (2 * TrainTP) / (2 * TrainTP + TrainFP + TrainFN)

    # Print metrics
    print '#######    Train Results    #################'
    print 'True positive rate: ' + repr(TrainTPR * 100.0) + '%'
    print 'False positive rate: ' + repr(TrainFPR* 100.0) + '%'
    print 'accuracy: ' + repr(TrainACC * 100.0) + '%'
    print 'F1: ' + repr(TrainF1 * 100.0) + '%'

    # Test the network on new data
    testValues = net.test(testData)

    # Set threshold
    for i in range(0,len(testValues)):
        if testValues[i] >= 0.5:
           testValues[i] = 1
        else:
            testValues[i] = 0

    # Define test parameters
    TestTP = 0.0
    TestFP = 0.0
    TestFN = 0.0
    TestTN = 0.0

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

        # Set error/success for each example
        if (testData[i][2] == 1 and testValues[i] == 1):
            TestTP = TestTP + 1
        elif (testData[i][2] == 1 and testValues[i]== 0):
            TestFP = TestFP + 1
        if (testData[i][2] == 0 and testValues[i] == 1):
            TestFN = TestFN + 1
        else:
            TestTN = TestTN + 1

    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title('Spiral after classification')

    # Calculate metrics
    TestTPR = TestTP / (TestTP + TestFN)
    TestFPR = TestFP / (TestFP + TestTN)
    TestACC = (TestTP + TestTN) / (TestTP + TestTN + TestFP + TestFN)
    TestF1 = (2 * TestTP) / (2 * TestTP + TestFP + TestFN)

    # Print metrics
    print '#######    Test Results    #################'
    print 'True positive rate: ' + repr(TestTPR * 100.0) + '%'
    print 'False positive rate: ' + repr(TestFPR * 100.0) + '%'
    print 'accuracy: ' + repr(TestACC * 100.0) + '%'
    print 'F1: ' + repr(TestF1 * 100.0) + '%'

    plt.show()

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