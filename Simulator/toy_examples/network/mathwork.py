import random
import numpy
import scipy
import math

def unifrnd(a, b):
    """
    :return: randomly generated by the uniform distribution bounded by [a, b]
    """
    return random.uniform(a, b)

def gamrnd(alpha, beta):
    """
    :return: randomly generated by the gamma distribution
    """
    return random.gammavariate(alpha, beta)

def exp(x):
    return math.exp(x)

def log2(x):
    return math.log2(x)