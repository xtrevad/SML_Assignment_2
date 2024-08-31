import numpy as np
# Input: numpy vector w, with d entries
#        numpy vector z, with d entries
# Output: label (+1 or -1)
def run(w,z):
    label = -1.
    if (w @ z) > 0:
        label = 1
    else: 
        label = -1
    return label
