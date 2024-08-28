import numpy as np
# Input: scalar C
#        number of iterations L
#        numpy matrix X of features, with n rows (samples), d columns (features)
#            X[i,r] is the r-th feature of the i-th sample
#        numpy vector y of labels, with n entries (samples)
#            y[i] is the label (+1 or -1) of the i-th sample
# Output: numpy vector w, with d entries
def run(C,L,X,y):
    n, d = X.shape
    w = np.zeros((d,))
    # Your code goes here
    return w
