__author__ = 'shay-macbook'
from numpy import *
from Layer import *
import matplotlib.pyplot as plt
from random import *
class Network:
    def __init__(self, activationFunction, eta, learningMethod, numOfEpoch, numOfMiniBatch, inputSize, outputSize, maxNeuronsInLayer, numOfLayers,L2regularizationFactor,L1regularizationFactor, momentumFactor):

        self.activation = activationFunction
        self.eta = eta
        self.learningMethod = learningMethod
        self.numOfEpoch = numOfEpoch
        self.numOfMiniBatch = numOfMiniBatch
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.maxNeuronsInLayer = maxNeuronsInLayer
        self.numOfLayers = numOfLayers
        self.L2regularizationFactor = L2regularizationFactor
        self.L1regularizationFactor = L1regularizationFactor
        self.network = []
        self.momentumFactor = momentumFactor
        self.buildNetwork()

    def buildNetwork(self):

        # set first layer
        firstLayer = Layer(self.activation,self.eta, self.inputSize, self.maxNeuronsInLayer, self.L2regularizationFactor,self.L1regularizationFactor, self.momentumFactor)
        self.network.append(firstLayer)

        # insert all the layers to the network object
        for i in range(1,self.numOfLayers):
            layerX = Layer(self.activation,self.eta, self.maxNeuronsInLayer, self.maxNeuronsInLayer, self.L2regularizationFactor,self.L1regularizationFactor, self.momentumFactor)
            self.network.append(layerX)

        # insert the output layer
        outputLayer = Layer(self.activation,self.eta, self.maxNeuronsInLayer, self.outputSize, self.L2regularizationFactor,self.L1regularizationFactor, self.momentumFactor)
        self.network.append(outputLayer)

    def train(self,trainData):

        localTrainedData = trainData
        # Run for each epoch
        for i in range(0,self.numOfEpoch):

            # We shuffle the training data each epoch in order to make the learning unbiased
            shuffle(localTrainedData)

            print 'epoch num is ' + repr(i)

            # Run for each input in the train data
            for j in range(0, len(localTrainedData)):
                nextlayerInput = localTrainedData[j][0], localTrainedData[j][1]

                # feed forward the input
                for xlayer in self.network:
                    nextlayerInput = xlayer.feedForward(nextlayerInput)

                # get output layer delta vector
                deltaVector = self.network[len(self.network) - 1].computeOutputDeltaVector(localTrainedData[j][2])

                # get last layer weights
                layerWeight  = self.network[len(self.network) - 1].getWeights()

                # run back propogation algorithm for last layer
                self.network[len(self.network) - 1].backProp()

                # if it is mini batch modulo, update the weights
                if j % self.numOfMiniBatch == 0:
                    self.network[len(self.network) - 1].update()

                for k in range(0, len(self.network) - 1):
                    # The index of the wanted layer to update
                    m = len(self.network) - 2 - k

                    # Calculating delta vector of layer m
                    deltaVector = self.network[m].computeDeltaVector(deltaVector, layerWeight)

                    # getting weights  of layer m
                    layerWeight = self.network[m].getWeights()

                    # run backpropogation algorithm for layer m
                    self.network[m].backProp()

                    # if it is mini batch modulo, update the weights
                    if j % self.numOfMiniBatch == 0:
                        self.network[m].update()


    def test(self,testData):

        testOutputs = []

        # Run on all test data set
        for j in range(0,len(testData)):
            # Get input
            nextlayerInput = testData[j][0],testData[j][1]

            # Feed feadforward algorithm for all layers
            for xlayer in self.network:
                nextlayerInput = xlayer.feedForward(nextlayerInput)

            # Add the result of last layer feedforward to the outputs array
            testOutputs.append(nextlayerInput[0])

        return testOutputs

    def getNetwork(self):
        i = 0
        for xlayer in self.network:
            i += 1
            print 'Layer ' + repr(i) + ' weights ' + repr(xlayer.getWeights())