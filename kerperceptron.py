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
    for iter in range(L):
        all_points_classified_correctly = True
        for i in range(n):
            inner_sum = 0
            for j in range(n):
                inner_sum += (alpha[j] * y[j] * K.run(X[j], X[i]))
            if y[i] * inner_sum <= 0:
                alpha[i] += 1
                all_points_classified_correctly = False
        if all_points_classified_correctly:
            break
    return alpha, iter+1
