# Adam Rilatt
# 08 / 20 / 20
# Neuron Revision 2

import math
import random

ReLU = lambda x : max(0, x)
ReLU_d = lambda x : 1 if x > 0 else 0

class Neuron:

    '''
    NOTES:
    '''

    def __init__(self, data_dim):

        self.weight_v = [random.gauss(0, 2 / data_dim) for i in range(data_dim)]
        self.bias = random.gauss(0, 1)
        self.layer = None
        self.error = 0
        self.value = None

    def get_value(self, input_v):
        #TODO: refactor
        #TODO: cache value once generated, regenerate on weight adjust

        if self.value != None:
            return self.value

        value = self.bias
        for i in range(len(input_v)):
            value += input_v[i] * self.weight_v[i]

        self.value = value
        return value

class Layer:

    '''
    NOTES:
    '''

    def __init__(self, width, data_dim, activation):

        self.neurons = []
        self.activation = activation

        for i in range(width):
            self.neurons.append(Neuron(data_dim))
            self.neurons[-1].layer = self

    def get_output(self, input_v):
        #TODO: refactor
        output_v = []
        for neu in self.neurons:
            output_v.append(self.activation(neu.get_value(input_v)))

        return output_v


class Model:

    '''
    NOTES:
    loss
    backward prop
    dictate input layer width
    '''

    def __init__(self):
        self.layers = []

    def add_layer(self, width, data_dim, activation):

        if len(self.layers) < 1:
            self.layers.append(Layer(width, data_dim, activation))

        else:
            self.layers.append(Layer(width, len(self.layers[-1].neurons), activation))

    def predict(self, input_v):
        #TODO: refactor?
        output_v = input_v
        for layer in self.layers:
            output_v = layer.get_output(output_v)

        return output_v

    def get_loss(self, actual, predicted):
        #TODO: refactor?

        error = 0
        for i in range(len(predicted)):
            error += (actual[i] - predicted[i]) ** 2

        return error / len(predicted)

    def backpropagate(self, input_v, actual, predicted):
        #TODO: clean this jawn up

        for i, layer in enumerate(reversed(self.layers)):

            if i == 0:    # output layer
                for j, neu in enumerate(layer.neurons):
                    neu.error = (actual[j] - neu.get_value(input_v)) * ReLU_d(neu.get_value(input_v))

            else:         # hidden layer
                for j, neu in enumerate(layer.neurons):
                    error = 0
                    #TODO: may be crappy code
                    for neu_post in self.layers[len(self.layers) - i].neurons:
                        error += neu_post.weight_v[j] * neu_post.error
                    neu.error = error * ReLU_d(neu.get_value(input_v))

    def update_weights(self, input_v, learning_rate):

        for i, layer in enumerate(self.layers):

            if i > 0:
                input_v = [neu.get_value(input_v) for neu in self.layers[i - 1].neurons]

            for neu in layer.neurons:
                neu.value = None
                for j in range(len(input_v)):
                    neu.weight_v[j] = learning_rate * neu.error * input_v[j]
                    neu.bias = learning_rate * neu.error

    def train(self, inputs, expected, n_epochs = 10, learning_rate = 0.5):

        for epoch in range(n_epochs):
            error = 0
            for i, row in enumerate(inputs):
                output = self.predict(row)
                error += self.get_loss(expected[i], output)
                self.backpropagate(row, expected[i], output)
                self.update_weights(row, learning_rate)

            print("Epoch", epoch, "(error =", error, ")", output)


    def __repr__(self):
        #TODO: find something more sensible than this, good lord

        s = ""

        for layer in self.layers:

            s += "\n\n"

            for neu in layer.neurons:

                s += "wv=" + str([round(val, 2) for val in neu.weight_v])
                s += " b=" + str(round(neu.bias, 2))
                s += "\t"

        return s

# testing
if __name__ == "__main__":

    #random.seed(43)

    # model construction
    input = [1., 3.3]
    brain = Model()
    brain.add_layer(2, len(input), ReLU)
    brain.add_layer(2, len(input), ReLU)
    #print(brain)

    # predict and error
    expected = [0., 1.]
    output = brain.predict(input)
    print(output)

    loss = brain.get_loss(output, expected)
    print(loss)

    # training on data
    x_data = [
        [0.535, 3.423],
        [0.968, 2.238],
        [0.213, 1.110],
        [2.445, 0.336],
        [3.898, 0.881],
        [4.023, 0.211]
    ]

    y_data = [
        [0., 1.],
        [0., 1.],
        [0., 1.],
        [1., 0.],
        [1., 0.],
        [1., 0.]
    ]

    brain.train(x_data, y_data, n_epochs = 50, learning_rate = 0.1)

    output = brain.predict([0.201, 1.333])
    print(output)
    loss = brain.get_loss(output, expected)
    print(loss)
