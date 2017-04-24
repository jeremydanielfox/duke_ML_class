# Code written by Jeremy Fox for Cynthia Rudin's ML class at Duke

import numpy as np
import math
import matplotlib.pyplot as plt

def g(x):
    return theta.dot(x) + theta_not

def f(x):
    return sigma(theta.dot(x) + theta_not)

def sigma(z):
    return 1.0/(1.0+math.exp(-z))

def true_positive_rate(prediction,actual):
    n = np.sum(actual)
    true_pos = 0.0
    for i in range(len(prediction)):
        if prediction[i] == 1 and actual[i] == 1:
            true_pos = true_pos +1
    return true_pos/n

def false_positive_rate(prediction,actual):
    n = 0.0
    false_pos = 0.0
    for i in range(actual.size):
        if actual[i]==0:
            n += 1
        if actual[i]==0 and prediction[i] == 1:
            false_pos += 1
    return false_pos/n


# import csv and remove headers
csv = np.genfromtxt("dataset.csv", delimiter=",")
data = np.delete(csv, 0,0)
# remove y values from csv and store as other array
y_values = data[:,3]
data = np.delete(data,3,1)

# set up theta values
theta = np.array((.05,-3.0,2.5))
theta_not = 0.3

# First, calculate f(x) for each point

f_values = []
for x in data:
    f_values.append(f(x))

# combine f and y values into useful tuples
values = []
for i in range(y_values.size):
    values.append((y_values[i],f_values[i]))
# sort these tuples
values = sorted(values, key=lambda value: value[1])

# use a plane sweep to generate tpr/fpr data
roc_fpr = []
roc_tpr = []
offset = 0.00001
for pair in values:
    y_val = pair[0]
    f_val = pair[1]
    threshold = f_val - offset
    predictions = map(lambda val: 1 if val-threshold >= 0 else 0, f_values)
    tpr = true_positive_rate(predictions,y_values)
    fpr = false_positive_rate(predictions,y_values)
    roc_fpr.append(fpr)
    roc_tpr.append(tpr)

plt.plot(roc_fpr,roc_tpr,'ro')
plt.axis([-.2, 1.2, -.2, 1.2])
plt.axhline(0)
plt.axvline(0)
plt.xlabel('False Positive Rate', fontsize=18)
plt.ylabel('True Positive Rate', fontsize=16)
plt.show()


