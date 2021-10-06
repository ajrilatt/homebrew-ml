# Adam Rilatt
# 13 April 2021
# Linear Regression Implementation

import math
import random

class LinearRegressor:

    '''
    Initializes a new model.
    '''
    def __init__(self, random_state = None, learning_rate = 0.01, cutoff = 0.001):

        self.rng = random.Random(random_state)

        # The learning rate for this model, which controls how quickly
        # the model can optimize via gradient descent and how sensitive the
        # model is to local / absolute minima.
        self.lr = learning_rate

        # The marginal difference between adjacent rounds of training which the
        # model considers small enough to cease fitting.
        self.cutoff = cutoff

        # The weight vector for this model, which has a width equal to the
        # number of input features.
        self.w_v = [self.rng.uniform(-1, 1)]

        # Loss function that will be used to evaluate this model. MSE is
        # mean squared error. Other options can include mean absolute error,
        # root mean squared error, etc.
        self.loss = self.mse
        self.loss_d = self.mse_d

    '''
    Generates an output vector based on the current internal state of the model and
    the given input vector X. If weights are provided, they will be used. Otherwise
    this model's current weights will be used.
    '''
    def predict(self, x_v, weights = None):
        if weights is None:
            return sum( self.w_v[i] * x_v[i] for i in range(len(x_v)) )
        else:
            return sum( weights[i] * x_v[i] for i in range(len(x_v)) )

    '''
    Generates a loss / cost measurement for this model based on the given
    output prediction 2d vector and true target value vector. This particular
    loss function is mean squared error. If weights is not set, uses the weights
    of this current model. If it is, uses the weights provided.
    '''
    def mse(self, x_2v, y_v, weights = None):
        return sum( (self.predict(x_2v[i], weights = weights) - y_v[i]) ** 2 for i in range(len(y_v)) ) / (2 * len(y_v))

    '''
    Defines the partial derivative of the mean squared error function given
    an output prediction 2d vector and a true target vector. This is calculated
    on a per-weight basis, so j is included to specify which weight.
    '''
    def mse_d(self, x_2v, y_v, j):
        return sum( (self.predict(x_2v[i]) - y_v[i]) * x_2v[i][j] for i in range(len(y_v)) ) / len(y_v)

    '''
    Updates the model's weight vector by stochastic gradient descent. This
    also factors in the model's learning rate.
    '''
    def update_weights(self, x_2v, y_v):
        for i in range(len(self.w_v)):
            # Whatever the partial derivative of the loss function is with
            # respect to the current element of the model's weight vector,
            # we traverse in the opposite direction (i.e. we travel with the
            # negative of the slope).
            self.w_v[i] -= (self.lr / len(y_v)) * self.loss_d(x_2v, y_v, i)

    '''
    Train the model on input and outputs x and y until the difference between
    iterations is less than this model's cutoff parameter. x must be a
    2-dimensional list with rows being data and internal rows (columns) being
    features, while y must be a 1-dimensional list representing the output column.
    '''
    def fit(self, x_train, y_train):

        # update the weight vector to fit the number of input features
        self.w_v = [self.rng.uniform(-1, 1) for n in range(len(x_train[0]))]

        while True:

            wv1 = self.w_v[:]

            self.update_weights(x_train, y_train)

            if self.loss(x_train, y_train, weights = wv1) - self.loss(x_train, y_train) < self.cutoff:
                break


if __name__ == "__main__":

    with open('swedish_jeeps.txt', 'r') as f:
        lines = f.readlines()

    for i in range(len(lines)):
        lines[i] = list(map(float, lines[i].strip().split('\t')))

    x = [[line[0]] for line in lines]
    y = [line[1] for line in lines]

    #x = [[n, n + 2] for n in range(1, 101)]
    #y = [n + 1 for n in range(1, 101)]

    brain = LinearRegressor(random_state = 1337, learning_rate = 1e-2, cutoff = 1e-5)
    brain.fit(x, y)

    print(math.sqrt(brain.loss(x, y)))

    preds = [brain.predict(n) for n in x]

    import matplotlib.pyplot as plt
    plt.scatter(x, y)
    plt.plot(x, preds)
    plt.show()
