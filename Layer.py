__author__ = 'shay-macbook'
from numpy import *
from activationFunction import *
class Layer:
    def __init__(self, activationFunction, eta, numOfneurons, numOfNeuronsInNextLayer):

        self.activationFunctionName  = activationFunction
        self.activationFunction = ActivationFunction(activationFunction)
        self.eta = eta
        self.numOfneurons = numOfneurons
        self.numOfNeuronsInNextLayer = numOfNeuronsInNextLayer;
        self.WMatrix = random.rand(numOfNeuronsInNextLayer, numOfneurons)
        self.input = 0
        self.output = 0
        self.weigthedinput = 0
        self.currentderevative = 0
        self.gradientW = zeros((numOfNeuronsInNextLayer, numOfneurons))


    def feedForward(self,input):
        self.input = input
        if type(input) is ndarray:
            self.weigthedinput = matmul(self.WMatrix, input)
        else:
            self.weigthedinput = self.WMatrix * input
        self.output = self.activate(self.weigthedinput)

        #print self.output
        return self.output


    def activate(self, value):
        output = self.activationFunction.activate(value)
        #print output
        return output

    def getResult(self):
        return self.output

    def derevative(self, value):
        self.currentderevative = self.activationFunction.dervative(value)
        return self.currentderevative

    def backProp(self, deltaVector):
        #print 's input for backprop: ' + repr(self.input)
        self.gradientW += self.eta * deltaVector * transpose(self.input)

    def computeDeltaVector(self,deltaVector, nextLayerW):
        Trm1 = matmul(transpose(deltaVector),nextLayerW)
        Trm2 = self.derevative(self.weigthedinput)
        self.deltaVector = multiply(Trm1,Trm2)

        return self.deltaVector

    def computeOutputDeltaVector(self,teacheranswer):
        #print 'Expected: ' + repr(teacheranswer) + ' Actual: ' + repr(self.output[0])
        Trm1 = teacheranswer - self.output
        Trm2 = self.derevative(self.weigthedinput)
        self.deltaVector = multiply(Trm1,Trm2)
        return self.deltaVector

    def update(self):
        self.WMatrix += self.gradientW
        #print 'delta weights Change: ' + repr(self.gradientW)
        self.gradientW = zeros((self.numOfNeuronsInNextLayer, self.numOfneurons))

    def getWeights(self):
        return self.WMatrix
    def getLayerOutput(self):
        return self.output