# Adam Rilatt
# 07 / 14 / 2020
# Simple Neural Network -- Neuron

'''
This file defines a Neuron capable of receiving input and producing output.

TODO:
- Implement backpropagation

'''

import math
import random

# activation functions and their derivatives
RELU      = lambda x : max(0, x)
RELU_d    = lambda x : 1 if x > 0 else 0
SIGMOID   = lambda x : 1 / (1 + (math.e ** -x))
SIGMOID_d = lambda x : SIGMOID(x) * (1 - SIGMOID(x))

# loss functions and their derivatives
loss_RELU      = lambda y, yHat : 0.5 * (y - yHat) ** 2
loss_RELU_d    = lambda y, yHat : -(y - yHat)
loss_SIGMOID   = lambda y, yHat : -math.log(yHat) if y else -math.log(1 - yHat)
loss_SIGMOID_d = lambda y, yHat : -y * (1 / yHat) * SIGMOID_d(yHat)

class Model:

    ''' This neural network package currently only supports a Sequential model.
        It also only supports scalar input data.                             '''
    def __init__(self, learning_rate = 0.5):
        self.layers        = []
        self.input_data    = []
        self.learning_rate = learning_rate
        self.current_loss  = None

    ''' Layers can be continuously added until the desired depth is reached. '''
    def add_layer(self, layer):
        self.layers.append(layer)
        layer.set_parent_model(self)

        if len(self.layers) > 1:
            self.layers[-2].set_next_layer(layer)

    ''' Data enters the model through the first layer. The dimensionality of the
        data corresponds to the width of the input layer. The model then "pulls"
        an output value from the last layer, which propagates an update for all
        layers.                                                              '''
    def predict(self, data):

        # as the neurons update, each layer requests that the previous layer
        # update its neurons. the input layer draws from input_data, a 'fake'
        # layer already containing values.
        self.input_data.clear()
        for val in data:
            self.input_data.append(Neuron(value = val))

        self.layers[-1].update_neurons()

        output = []
        for neu in self.layers[-1].neurons:
            output.append(neu.value)

        return output


    ''' Depending on the output layer (sigmoid, relu, etc.), the model will
        calculate its loss function differently. For regression models (relu),
        mean squared error is used. For classification (sigmoid), cross-entropy
        is used.                                                             '''
    def update_loss(self, predicted, actual):

        error = 0

        if self.layers[-1].act_func == RELU:      # mean squared error

            for i in range(len(predicted)):
                # dividing by 2 makes integration easier later on
                error += 0.5 (actual[i] - predicted[i]) ** 2

        elif self.layers[-1].act_func == SIGMOID: # cross-entropy

            for i in range(len(predicted)):
                try:

                    if actual[i]:
                        error -= math.log(predicted[i])
                    else:
                        error -= math.log(1 - predicted[i])

                except ValueError as e: # log of 0 is undefined
                    pass

        self.current_loss = error





class Layer:

    def __init__(self, width, activation_function):
        self.width    = width
        self.act_func = activation_function
        self.is_input = False
        self.previous = None
        self.next     = None
        self.neurons  = None
        self.model    = None

    ''' A newly created Layer stores the reference to its parent model so that
        its functions can be accessed later. It also stores the layer before
        itself so that it can request output from that layer.                '''
    def set_parent_model(self, model):
        self.model = model

        if len(model.layers) == 0 or self == model.layers[0]:   # first layer
            self.is_input = True

        else:                                                   # normal layer
            self.previous = model.layers[len(model.layers) - 2]

        # populate the layer with new Neurons and register them with this Layer
        self.neurons = [Neuron() for i in range(self.width)]
        for neu in self.neurons: neu.set_parent_layer(self)

    def set_next_layer(self, layer):
        self.next = layer

    ''' Layers must receive data from the previous layer and use that data to
        calculate values for their own neurons, which may in turn feed the next
        layer.                                                               '''
    def update_neurons(self):
        if not self.is_input:
            self.previous.update_neurons()

        for neu in self.neurons:
            neu.set_parent_layer(self)  # update data widths (mostly for input
            neu.recv_input()            # layer) and update each neuron


    def __repr__(self):
        # toString method
        return "Layer(width = %d, act_func = %s, is_input = %s)" % (
                 self.width,
                 "SIGMOID" if self.act_func == SIGMOID else "RELU",
                 "True" if self.is_input else "False"
        )



class Neuron:

    def __init__(self, value = 0):
        self.value    = value
        self.bias     = 0
        self.parent   = None
        self.inputs   = None
        self.weight_v = []
        self.act_func = None


    ''' When a Neuron is added to a Layer, the Layer "registers" itself with
        the Neuron so its functions can be accessed later.                   '''
    def set_parent_layer(self, parent):
        self.parent   = parent
        self.act_func = parent.act_func

        if parent.is_input:
            self.inputs = parent.model.input_data
        else:
            self.inputs = parent.previous.neurons

        self.weight_v = [
        random.gauss(0, 2 / len(self.inputs)) for i in range(len(self.inputs))
        ]



    ''' When the parent Layer signals to do so, the Neuron should request the
        value of the Neurons in the previous Layer and apply its weights to
        those values.                                                        '''
    def recv_input(self):
        # general form z = (w1 * x1) + (w2 * x2) ... + (wn * xn) + b
        self.value = 0

        for i, neu in enumerate(self.inputs):
            self.value += neu.value * self.weight_v[i]

        self.value += self.bias
        self.value = self.act_func(self.value)

    def __repr__(self):
        # toString method
        return "Neuron(value = %f, weight_v = %s, bias = %f)" % (
                  self.value,
                  self.weight_v,
                  self.bias
        )



# testing
if __name__ == "__main__":

    #random.seed(42)

    # model creation
    brain = Model()
    brain.add_layer(Layer(2, RELU))
    brain.add_layer(Layer(2, RELU))
    brain.add_layer(Layer(1, SIGMOID))

    # making a prediction (no, it hasn't been trained on anything, but still!)
    # in this case, [3, 5, 8] should -> [0]
    results = brain.predict([3, 5, 8])
    print(results)
    brain.update_loss(results, [0])
    print(brain.current_loss)

    # printing final values of model
    for l in brain.layers:
        print(l)
        for n in l.neurons:
            print("\t" + str(n))
