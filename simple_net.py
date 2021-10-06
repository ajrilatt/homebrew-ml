# Adam Rilatt
# 10 / 19 / 20
# Simple Neural Net

'''
Defines a model for use with linear regression. No, really, that's it.
'''

import random
#random.seed(42)

num_samples   = 100
learning_rate = 0.001

# scalar data ==> 1 weight, 1 bias
weight = random.gauss(0, 1)
bias   = random.gauss(0, 1)

# literally perfect training data for y = 2x
x_train = [n for n in range(1, num_samples + 1)]
random.shuffle(x_train)
y_train = [2 * n for n in x_train]

for i, x in enumerate(x_train):

    # forward propagation
    y_pred = weight * x + bias

    # loss evalutation
    y_true = y_train[i]
    loss = 0.5 * (y_true - y_pred) ** 2

    # backward propagation
    loss_w = (y_true - y_pred) * -weight
    loss_b = (y_true - y_pred) * -1
    weight -= loss_w * learning_rate
    bias   -= loss_b * learning_rate

    print("Sample %d\t loss = %.4f\t w = %.2f b = %.2f" % (i, loss, weight, bias))
