import numpy as np


def sigmoid(x):
    """
    Performs the sigmoid calc which is simply a log curve forcing stuff to
    1 or 0.

    """
    return 1/(1+np.exp(-1*x))

def sigmoid_derivative(x):
    """
    Calculates the derivative of the sigmoid
    """
    return x * (1-x)

def loss(y, y_hat):
    """
    Calculates the l2 norm of y -y^2
    """
    residual = 0
    for i,yy in enumerate(y_hat):
        residual += (y[i] - yy)**2

    return residual

class NeuralNetwork(object):
    def __init__(self,x,y):
        """
        A two layer model output Y_hat is represented by :

        Y_hat = sigma * (W_2* sigma * (W_1 + b_1) + b_2)


        """

        # Input using to map to the output
        self.input = x

        # Layer one.
        self.weights1   = np.random.rand(self.input.shape[1],4)

        # Layer 2.
        self.weights2   = np.random.rand(4,1)

        self.y = y

        self.output = np.zeros(y.shape)

    def feedForward(self):
        """
        Propagate through to the output to evaluate our current weights and
        bias
        """
        # Calculate the current values of the first layer
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))

        # Calculate the sigmoid of the second layer which is the output
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backPropagate(self):
        """
        Evaluate the loss function after feeding forward.
        This uses the gradient decent for finding a local minima in the loss
        function.
        (sounds like a newton solver to me)
        """

                # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2


if __name__ == "__main__":
    x = np.array([[0,0,1],
                  [0,1,1],
                  [1,0,1],
                  [1,1,1]])

    y = np.array([[0],
                  [1],
                  [1],
                  [0]])

    n = NeuralNetwork(x,y)
    z = 0

    while loss(n.y,n.output) > 0.00001 or z <10000:
        n.feedForward()
        n.backPropagate()
        z+=1
    print("Number of iterations: {}".format(z))
    print(n.output)
