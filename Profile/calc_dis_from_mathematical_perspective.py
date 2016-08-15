import numpy as np
def mycdist(a, b, w):
    w = w[np.newaxis, np.newaxis, :]
    c, d = a[:, np.newaxis, :], b[np.newaxis, :, :]
    g = (c - d) * w
    h = np.sqrt(np.sum(g**2, axis=2))
    return h
