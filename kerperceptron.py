import K
import numpy as np
# Input: number of iterations L
#        numpy matrix X of features, with n rows (samples), d columns (features)
#            X[i,r] is the r-th feature of the i-th sample
#        numpy vector y of labels, with n entries (samples)
#            y[i] is the label (+1 or -1) of the i-th sample
# Output: numpy vector alpha, with n entries
#         number of iterations that were actually executed (iter+1)
def run(L,X,y):
    n, d = X.shape
    alpha = np.zeros((n,))
    iter = 0
    # Your code goes here
    return alpha, iter+1
