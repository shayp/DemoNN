__author__ = 'shay-macbook'
from numpy import *
from Layer import *
import matplotlib.pyplot as plt
class Network:
    def __init__(self, activationFunction, eta, learningMethod, numOfEpoch, numOfMiniBatch, inputSize, outputSize, maxNeuronsInLayer, numOfLayers):

        self.activation = activationFunction
        self.eta = eta
        self.learningMethod = learningMethod
        self.numOfEpoch = numOfEpoch
        self.numOfMiniBatch = numOfMiniBatch
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.maxNeuronsInLayer = maxNeuronsInLayer
        self.numOfLayers = numOfLayers
        self.network = []
        self.buildNetwork()

    def buildNetwork(self):
        inputLayer = Layer(self.activation,self.eta, self.inputSize, self.maxNeuronsInLayer)
        self.network.append(inputLayer)
        for i in range(1,self.numOfLayers):
            layerX = Layer(self.activation,self.eta, self.maxNeuronsInLayer, self.maxNeuronsInLayer)
            self.network.append(layerX)

        outputLayer = Layer(self.activation,self.eta, self.maxNeuronsInLayer, self.outputSize)
        self.network.append(outputLayer)

    def train(self,inputs, outputs):
        for i in range(0,self.numOfEpoch):
            print 'epoch num is ' + repr(i)
            for j in range(0, len(inputs)):
                nextlayerInput = inputs[j]
                for xlayer in self.network:
                    nextlayerInput = xlayer.feedForward(nextlayerInput)

                deltaVector = self.network[len(self.network) - 1].computeOutputDeltaVector(outputs[j])
                layerWeight  = self.network[len(self.network) - 1].getWeights()
                self.network[len(self.network) - 1].backProp(deltaVector)
                if j % self.numOfMiniBatch == 0:
                    self.network[len(self.network) - 1].update()

                for k in range(0, len(self.network) - 2):
                    m = len(self.network) - 2 - k
                    deltaVector = self.network[m].computeDeltaVector(deltaVector, layerWeight)
                    layerWeight = self.network[m].getWeights()
                    self.network[m].backProp(deltaVector)
                    if j % self.numOfMiniBatch == 0:
                        self.network[m].update()


    def test(self,inputs, outputs):
        testOutputs = []
        for j in range(0,len(inputs)):
            nextlayerInput = inputs[j]
            for xlayer in self.network:
                nextlayerInput = xlayer.feedForward(nextlayerInput)
            testOutputs.append(nextlayerInput[0])
            #print (outputs[j] - nextlayerInput[0]) * (outputs[j] - nextlayerInput[0])

        return testOutputs

    def getNetwork(self):
        i = 0
        for xlayer in self.network:
            i += 1
            print 'Layer ' + repr(i) + ' weights ' + repr(xlayer.getWeights())