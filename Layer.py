__author__ = 'shay-macbook'
from numpy import *
from activationFunction import *
class Layer:
    def __init__(self, activationFunction, eta, numOfneuronsInPrevLayer, numOfNeuronsInLayer):

        self.activationFunctionName = activationFunction
        self.activationFunction = ActivationFunction(activationFunction)
        self.eta = eta
        self.numOfneuronsInPrevLayer = numOfneuronsInPrevLayer
        self.numOfNeuronsInLayer = numOfNeuronsInLayer
        self.WMatrix = random.rand(numOfNeuronsInLayer, numOfneuronsInPrevLayer + 1)
        self.input = 0
        self.output = 0
        self.weigthedinput = 0
        self.currentderevative = 0
        self.gradientW = zeros((numOfNeuronsInLayer, numOfneuronsInPrevLayer + 1))


    def feedForward(self,input):
        #print 'feedForward input: ' + repr(input)
        self.input = input
        input = append(asarray(input), 1)
        #self.input = delete(input, len(input) - 1)
        if type(input) is ndarray:
            self.weigthedinput = matmul(self.WMatrix, input)
        else:
            self.weigthedinput = self.WMatrix * input
        self.output = self.activate(self.weigthedinput)
        #print 'feedForward output: ' + repr(append(self.output, 1))
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
        #print 'backprop input: ' + repr(transpose(self.input))
        #print 'backprop deltaVector:' + repr(deltaVector)
        self.gradientW += self.eta * deltaVector * transpose(append(asarray(self.input), 0))
        #print 'delta change values' + repr(self.gradientW)
    def computeDeltaVector(self,deltaVector, nextLayerW):
        Trm1 = matmul(transpose(deltaVector),nextLayerW)
        Trm2 = self.derevative(self.weigthedinput)
        self.deltaVector = multiply(Trm1,Trm2)

        return self.deltaVector

    def computeOutputDeltaVector(self,teacheranswer):
        #print 'Expected: ' + repr(teacheranswer) + ' Actual: ' + repr(self.output[0])
        Trm1 = teacheranswer - self.output
        #print Trm1
        Trm2 = self.derevative(self.weigthedinput)
        self.deltaVector = multiply(Trm1,Trm2)
        return self.deltaVector

    def update(self):
        self.WMatrix += self.gradientW
        #print 'delta weights Change: ' + repr(self.gradientW)
        self.gradientW = zeros((self.numOfNeuronsInLayer, self.numOfneuronsInPrevLayer + 1))

    def getWeights(self):
        return self.WMatrix
    def getLayerOutput(self):
        return self.output