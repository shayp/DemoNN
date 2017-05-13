# DemoNN

A neural network for spiral classification in python using numpy
Includes:
* L1 & L2 regularization
* Different activation functions
* SGD with momentum 
* Mini batch implementation

## Main.py
* Read coirdinates input files from CSV, build neural network with constant configuration.
*  can read a serialized network class from saved file

## Netowrk.py 
A neural network class
Perform train and test functionality.

## Layer.py
Includes:
* FeedForward algorithm
* BackPropogation algorithm 
* Calculation of delta vector for weights change in backprop
* Bias term

## activationfunction.py
Enable different activation functions for the network

## Incstructions
* If you want to train the netwrok call testSpiralClassification
* If you want to test serialized network call RunSerializedNetwork