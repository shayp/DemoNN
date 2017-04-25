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
        self.WMatrix = 0.01 * random.rand(numOfNeuronsInLayer, numOfneuronsInPrevLayer + 1)
        self.input = 0
        self.output = 0
        self.weigthedinput = 0
        self.currentderevative = 0
        self.gradientW = zeros((numOfNeuronsInLayer, numOfneuronsInPrevLayer + 1))


    def feedForward(self,input):
        #print 'feedForward input: ' + repr(input)
        self.input = input
        input = append(asarray(input), 1)
        #print '[feedForward] input size: ' + repr(input.size) + ' input val: ' + repr(input)
        if type(input) is ndarray:
            self.weigthedinput = matmul(self.WMatrix, input)
        else:
            print 'Bug??? '
            #self.weigthedinput = self.WMatrix * input

        #print '[feedForward] weigthedinput size: ' + repr(self.weigthedinput.size)
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
        #print 'computeDeltaVector start, inputs: '
        #print 'deltaVector ' + repr(deltaVector)
        #print 'nextLayerW ' + repr(nextLayerW)
        Trm1 = matmul(transpose(deltaVector),nextLayerW)
        #print 'Trm 1 ' + repr(Trm1)
        #print 'self.weigthedinput size: ' + repr(self.weigthedinput.size)
        Trm2 = self.derevative(append(self.weigthedinput, 1))
        #print 'Trm 2 ' + repr(Trm2)
        self.deltaVector = multiply(Trm1, Trm2)

        return self.deltaVector

    def computeOutputDeltaVector(self,teacheranswer):
        #print 'Expected: ' + repr(teacheranswer) + ' Actual: ' + repr(self.output[0])
        Trm1 = teacheranswer - self.output
        #print 'Trm 1 ' + repr(Trm1)
        Trm2 = self.derevative(self.weigthedinput)
        #print 'Trm 2 ' + repr(Trm2)

        self.deltaVector = multiply(Trm1, Trm2)
        return self.deltaVector

    def update(self):
        self.WMatrix += self.gradientW
        #print 'delta weights Change: ' + repr(self.gradientW)
        self.gradientW = zeros((self.numOfNeuronsInLayer, self.numOfneuronsInPrevLayer + 1))

    def getWeights(self):
        return self.WMatrix
    def getLayerOutput(self):
        return self.output