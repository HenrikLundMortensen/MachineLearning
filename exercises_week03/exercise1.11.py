import numpy as np
from scipy.special import binom as binom


def q3(p):
    """

    """
    n = 25
    prop = 0
    for i in range(13):
        prop += binom(n,i)*p**i * (1-p)**(n-i)

    return 1 - prop




