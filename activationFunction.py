__author__ = 'shay-macbook'

from numpy import *
class ActivationFunction:
    def __init__(self, activationFunction):

        self.activationFunction = activationFunction
        if activationFunction != 'tanh' and activationFunction != 'sigmoid' and activationFunction != 'relu' and activationFunction != 'linear':
            print 'Errror! no activation function found'

    def activate(self,input):
        if (self.activationFunction == 'tanh'):
            output = tanh(input)
        elif (self.activationFunction == 'sigmoid'):
            output =  1 / (1 + exp(-input))
        elif (self.activationFunction == 'linear'):
            output = input
        elif (self.activationFunction == 'relu'):
            print 'relu not implemented yet'
        else:
            print 'error'
        return output

    def derivative(self, input):
        if (self.activationFunction == 'tanh'):
            x = tanh(input)
            output = 1 - x*x;
        elif (self.activationFunction == 'sigmoid'):
            x = 1 / (1 + exp(-input))
            output =  x * (1 - x)
        elif (self.activationFunction == 'linear'):
            output = input
        elif (self.activationFunction == 'relu'):
            print 'relu not implemented yet'
        else:
            print 'error'
        return output