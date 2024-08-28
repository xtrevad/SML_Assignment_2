# COMP90051 Jean Honorio (jean.honorio@unimelb.edu.au) 2024-2
import numpy as np
import math
# Input: numpy vector x of d entries
#        numpy vector z of d entries
# Output: kernel K(x,z) = exp(-1/2 * norm(x-z)^2)
# Example on how to call the script:
#     import K
#     v = K.run( np.array([1, 4, 3]) , np.array([2, 7, -1]) )
def run(x,z):
    x = np.asarray(x).flatten()
    z = np.asarray(z).flatten()
    if x.size != z.size:
        raise ValueError
    return math.exp(-0.5 * np.sum((x-z) ** 2))
