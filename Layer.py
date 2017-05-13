__author__ = 'shay-macbook'
from numpy import *
from activationFunction import *
class Layer:
    def __init__(self, activationFunction, eta, numOfneuronsInPrevLayer, numOfNeuronsInLayer, L2regularizationFactor, L1regularizationFactor, momentumFactor):

        self.activationFunctionName = activationFunction
        self.activationFunction = ActivationFunction(activationFunction)
        self.eta = eta
        self.numOfneuronsInPrevLayer = numOfneuronsInPrevLayer
        self.numOfNeuronsInLayer = numOfNeuronsInLayer
        self.WMatrix = random.rand(numOfNeuronsInLayer, numOfneuronsInPrevLayer + 1)
        self.input = 0
        self.output = 0
        self.weigthedinput = 0
        self.currentDerivative = 0
        self.momentumFactor = momentumFactor
        self.L2regularizationFactor = L2regularizationFactor
        self.L1regularizationFactor = L1regularizationFactor

        self.deltaW = zeros((numOfNeuronsInLayer, numOfneuronsInPrevLayer + 1))
        self.momentumChange = 0


    def feedForward(self, input):
        # Adding one neuron with value one for bias
        input = append(asarray(input), 1)

        # Update input to include bias
        self.input = input

        if type(input) is ndarray:
            # compute W*X
            self.weigthedinput = matmul(self.WMatrix, input)
        else:
            print 'Bug??? '

        # Run activation function, add the result to the layer output val
        self.output = self.activate(self.weigthedinput)

        return self.output


    def activate(self, value):

        # Call the initialized activation function
        output = self.activationFunction.activate(value)

        return output

    def getResult(self):

        return self.output

    def derivative(self, value):

        # Call the initialized derivative activation function
        self.currentDerivative = self.activationFunction.derivative(value)
        return self.currentDerivative

    def backProp(self):

        # In case of a paramater and not a vector, do classic mul
        if self.input.size  == 1 or self.deltaVector.size == 1:
            currentChange = self.eta * self.deltaVector * transpose(self.input) - self.L2regularizationFactor * self.eta * self.getWeights() + self.momentumFactor * self.momentumChange - self.L1regularizationFactor * self.eta * sign(self.getWeights())
        else:

            # We chanhe the vector to a ndarry struct in order to creat a matrix
            inputToMul = reshape(self.input, (1,self.input.size))
            deltaVectorToMul = reshape(self.deltaVector, (self.deltaVector.size,1))

            # We calculate the deltaW value for change
            currentChange = self.eta * matmul(deltaVectorToMul, inputToMul) - self.L2regularizationFactor * self.eta * self.getWeights() + self.momentumFactor * self.momentumChange - self.L1regularizationFactor * self.eta * sign(self.getWeights())

        # We update the deltaW with current calculation
        self.deltaW += currentChange

        # We save current deltaW for future mumentum
        self.momentumChange = currentChange

    def computeDeltaVector(self,deltaVector, nextLayerW):

        # Trm1 = we multiply  deltaVector(L + 1) * W(L + 1)
        Trm1 = matmul(transpose(deltaVector), nextLayerW)

        # Trm 2 = the derevative of( W * X) we append zero for the bias neuron
        Trm2 = self.derivative(self.weigthedinput)
        Trm2 = append(Trm2, 0)

        # the layer delta Vector is - deltaVector(L + 1) * W(L + 1) (*) g'(w*x)
        self.deltaVector = multiply(Trm1, Trm2)

        # We remove the extra neuron
        self.deltaVector = delete(self.deltaVector, Trm2.size - 1)

        return self.deltaVector

    def computeOutputDeltaVector(self,teacheranswer):

        # Trm1 - We calculate the difference between teacher answer to the layer output
        Trm1 = teacheranswer - self.output

        # Trm 2 - We calculate g'(W * X)
        Trm2 = self.derivative(self.weigthedinput)

        # delta(L) - (teacher answer - network answer) * g'(W * X)
        self.deltaVector = multiply(Trm1, Trm2)

        return self.deltaVector

    def update(self):

        # We update the weights
        self.WMatrix += self.deltaW

        # We initialize the weights
        self.deltaW = zeros((self.numOfNeuronsInLayer, self.numOfneuronsInPrevLayer + 1))

    def getWeights(self):
        return self.WMatrix

    def getLayerOutput(self):
        return self.output