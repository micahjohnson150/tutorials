# NeuralNetwork Notes

Most of this almost verbatim from https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6

A Neural network is simple something that maps an input to an output.
Consists of:

* Input layer
* Arbitrary number if hidden layers
* Weights and biases between each layer
* A choice of activation functions (we are using Sigmoid)


Sigmoid is used because the differientation is convenient during backPropagatation
