# COMP90051 (2024 S2) Trevor Adelson (tadelson@student.unimelb.edu.au) 693698
import K
import numpy as np
# Input: numpy vector alpha, with n entries
#        numpy matrix X of features, with n rows (samples), d columns (features)
#            X[i,r] is the r-th feature of the i-th sample
#        numpy vector y of labels, with n entries (samples)
#            y[i] is the label (+1 or -1) of the i-th sample
#        numpy vector z, with d entries
# Output: label (+1 or -1)
def run(alpha,X,y,z):
    n, _ = X.shape
    inner_sum = 0
    for i in range(n):
        inner_sum += (alpha[i] * y[i] * K.run(X[i], z))
    if inner_sum > 0:
        label = 1
    else:
        label = -1
    return label
