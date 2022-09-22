import numpy as np

def stretch_distribution(z, a=2):
    return 1/np.sqrt(z) if (z >= 1/a and z <= a) else 0
