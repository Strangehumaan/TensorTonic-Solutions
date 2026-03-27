import numpy as np

def sigmoid(x):
    input = np.asarray(x)
    sig = (1/(1+np.exp(-(input))))
    return sig 