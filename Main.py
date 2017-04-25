__author__ = 'shay-macbook'
from Network import *
from numpy import *
from random import *
import matplotlib.pyplot as plt

def main():

    activationFunction = 'tanh'
    eta = 0.02
    numOfLayers = 1
    maxNeuronsInLayer = 5
    inputSize = 1
    outputSize = 1
    learningMethod = 'decent'
    numOfEpoch = 4000
    numOfMiniBatch = 1

    inputs  = []
    outputs = []
    testinputs  = []
    testoutputs = []
    net = Network(activationFunction, eta, learningMethod, numOfEpoch, numOfMiniBatch, inputSize, outputSize, maxNeuronsInLayer, numOfLayers)
    for i in range(0,5000):
        inputs.append( 2 * math.pi * random())
        outputs.append(sin(inputs[i]))
        testinputs.append(2 * math.pi * random())
        testoutputs.append(sin(testinputs[i]))

    net.train(inputs,outputs)
    testValues = net.test(testinputs, testoutputs)
    net.getNetwork()
    plt.figure(1)
    plt.subplot(311)
    plt.plot(asarray(testinputs),asarray(testoutputs),'bo')
    plt.subplot(312)
    plt.plot(asarray(testinputs),asarray(testValues),'bo')
    plt.subplot(313)
    plt.plot(asarray(testoutputs),asarray(testValues),'bo')
    plt.show()

if __name__ == "__main__":
    main()