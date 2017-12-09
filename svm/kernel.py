import numpy as np


class Kernel(object):
    """Check kernels here https://en.wikipedia.org/wiki/Support_vector_machine"""
    @staticmethod
    def linear():
        return lambda x, y: np.inner(x, y)

    @staticmethod
    def gaussian(sigma):
        return lambda x, y: np.exp( - np.sum(np.power( x-y , 2)) / (2 *  sigma ** 2))