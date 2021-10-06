# Adam Rilatt
# 13 April 2021
# Logistic Regression Classifier

import math
import random

x_train = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
y_train = [0, 0, 0, 0, 0, 0, 0, 0, 1,  0,  1,  0,  1,  1,  1,  1,  1,  1,  1,  1]

#random.seed(451)

threshold = 0.5
num_reps = 5000
start_learn_rate = 2
f = lambda k, x0, x : 1 / (1 - math.e ** (-k * (x - x0)))

k  = random.uniform(-1, 1)
x0 = random.uniform(-1, 1)
best_acc = 0
learn_rate = start_learn_rate

for n in range(num_reps):

    k_n  = k  + random.uniform(-1, 1) * learn_rate
    x0_n = x0 + random.uniform(-1, 1) * learn_rate

    preds = list(map(lambda x: f(k_n, x0_n, x), x_train))
    clsf  = [1 if i > threshold else 0 for i in preds]
    error = [clsf[i] == y_train[i] for i in range(len(y_train))]
    accuracy = sum(error) / len(error)

    if accuracy > best_acc:
        k  = k_n
        x0 = x0_n
        best_acc = accuracy

    learn_rate -= start_learn_rate / num_reps

print(best_acc)
print("k = ", k)
print("x0 = ", x0)
