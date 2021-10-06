# Adam Rilatt
# 14 April 2021
# Logistic Classifier Implementation

import math
import random

class LogisticClassifier:

    '''
    Initializes a new model.
    '''
    def __init__(self, threshold = 0.5, random_state = None, learning_rate = 0.01, cutoff = 0.001):

        # Decision threshold used for classification
        self.threshold = threshold

        # Random state used for weight initialization
        self.rng = random.Random(random_state)

        # Learning rate constant that defines the rate of gradient descent
        self.lr = learning_rate

        # The minimum distance between two rounds of gradient descent required
        # to justify continuing training.
        self.cutoff = cutoff

        # Initialized weights. This will be expanded when fit() is called
        self.w_v = [self.rng.uniform(-1, 1)]

        # Loss function and derivative used to calculate raw error and gradients
        self.loss = self.log_loss
        self.loss_d = self.log_loss_d


    '''
    Returns the value of the logistic function at some point h.
    '''
    def sigmoid(self, h):
        return 1 / (1 + math.exp(-h))

    '''
    Generates a numeric output between 0 and 1 given the current internal model
    state and an input vector x. If weights are provided, they will be used.
    Otherwise the model's current weights will be used. The largest difference
    between this model's prediction and a LinearRegressor's model prediction is
    that this model passes its values through a sigmoid function.
    '''
    def predict(self, x_v, weights = None):
        if weights is None:
            return self.sigmoid(sum( self.w_v[i] * x_v[i] for i in range(len(x_v)) ))
        else:
            return self.sigmoid(sum( weights[i] * x_v[i] for i in range(len(x_v)) ))

    '''
    Generate a binary class prediction based on an input vector x.
    '''
    def classify(self, x_v):
        return 1 if self.predict(x_v) > self.threshold else 0


    '''
    Generates precision and recall scores for the input 2d vector x and
    the target output vector y.
    '''
    def precision_and_recall(self, x_2v, y_v):
        pred = [self.classify(x_2v[i])  for i in range(len(y_v))]
        print(pred)
        tp = sum([pred[i] == y_v[i] == 1 for i in range(len(y_v))])
        tn = sum([pred[i] == y_v[i] == 0 for i in range(len(y_v))])
        fp = sum([pred[i] != y_v[i] == 0 for i in range(len(y_v))])
        fn = sum([pred[i] != y_v[i] == 1 for i in range(len(y_v))])
        print(tp)
        print(tn)
        print(fp)
        print(fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return precision, recall

    '''
    Generates a loss / cost measurement for this model based on a given output
    prediction 2d vector and true target value vector. This error function is
    log-loss, used for logistic regression. If weights are passed, they will be
    used; otherwise, this model's weights are used.
    '''
    def log_loss(self, x_2v, y_v, weights = None):
        return -sum( y_v[i] * math.log(self.predict(x_2v[i], weights = weights), math.e) +
          (1 - y_v[i]) * math.log(1 - self.predict(x_2v[i], weights = weights), math.e)
          for i in range(len(y_v)) ) / len(y_v)

    '''
    Defines the partial derivative of log-loss given an output prediction 2d vector
    and a true target vector. j is included to specify which weight we should
    take the partial derivative with respect to.
    '''
    def log_loss_d(self, x_2v, y_v, j):
        return sum( (self.predict(x_2v[i]) - y_v[i]) * x_2v[i][j] for i in range(len(y_v)) ) / len(y_v)

    '''
    Updates the mode's weight vector by stochastic gradient descent. This also
    factors in the model's learning rate.
    '''
    def update_weights(self, x_2v, y_v):
        for i in range(len(self.w_v)):
            self.w_v[i] -= (self.lr / len(y_v)) * self.loss_d(x_2v, y_v, i)

    '''
    Train the model on input and outputs x and y until the difference between
    iterations is less than this model's cutoff parameter. x must be a 2-dimensional
    list with rows being data and internal rows (columns) being features, while
    y must be a 1-dimensional list representing the output column.
    '''
    def fit(self, x_train, y_train):

        self.w_v = [self.rng.uniform(-1, 1) for n in range(len(x_train[0]) + 1)]
        x_new = [[1] + x_train[i][:] for i in range(len(x_train))]

        while True:

            wv1 = self.w_v[:]

            self.update_weights(x_new, y_train)
            print(self.loss(x_new, y_train))

            if self.loss(x_new, y_train, weights = wv1) - self.loss(x_new, y_train) < self.cutoff:
                break

if __name__ == "__main__":

    from sklearn import datasets
    import numpy as np

    # The famous iris dataset.
    # Our model prefers raw Python lists, so we
    # perform some casting before we continue.
    iris = datasets.load_iris()
    X = list(iris["data"][:, 3:]) # petal width
    X = [list(Xsub) for Xsub in X]
    Y = list((iris["target"] == 2).astype('int')) # 1 if Iris virginica, else 0

    log_reg = LogisticClassifier(threshold = 0.5, random_state = 451, learning_rate = 1e-2, cutoff = 1e-5)
    log_reg.fit(X, Y)

    precision, recall = log_reg.precision_and_recall(X, Y)
    print(log_reg.w_v)
    print(precision)
    print(recall)
