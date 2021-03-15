import logging
import functools

import numpy as np
import pytest

from pycobyla import Cobyla
from tests.test_originals import cobyla_tester


def gaussian(x, mu=None, sig=None, A=1):
    n = len(x)
    mu =  np.zeros(n) if mu is None else mu
    sig = np.ones(n) if sig is None else sig

    zz = ((((x - mu) / sig) ** 2) / 2).sum()
    return A * np.exp(-zz)


def neg_gaussian(x, mu=None, sig=None, A=1):
    return -gaussian(x, mu=mu, sig=sig, A=A)
        

def test_problem_1_gaussian_2d():
    '''
    Test gaussian 2d Gaussian with mu=(0,0), sig=(1,1) 

    G((x, y), mu=(0,0), sig=(1,1), A=1)
    C1(x, y) = 1 - x >= 0
    C2(x, y) = 1 + x >= 0
    C3(x, y) = 1 - y >= 0
    C4(x, y) = 1 + y >= 0
    
    '''
    G = neg_gaussian
    c1 = lambda x: 1 - x[0]
    c2 = lambda x: 1 + x[0]
    c3 = lambda x: 1 - x[1]
    c4 = lambda x: 1 + x[1]

    
    C = (c1, c2, c3, c4)
    x = np.ones(2)
    
    mu = np.array((0, 0))
    cobyla_tester(G, C, x, mu)


def test_problem_2_gaussian_2d_random_mu():
    '''
    Test gaussian 2d Gaussian with mu="random sample", sig=(1,1)

    G((x, y), mu=random, sig=(1,1), A=1)
    C1(x, y) = 1 - x >= 0
    C2(x, y) = 1 + x >= 0
    C3(x, y) = 1 - y >= 0
    C4(x, y) = 1 + y >= 0
    
    '''
    mu = np.random.random(2)
    G = functools.partial(neg_gaussian, mu=mu)
    c1 = lambda x: 1 - x[0]
    c2 = lambda x: 1 + x[0]
    c3 = lambda x: 1 - x[1]
    c4 = lambda x: 1 + x[1]
    
    C = (c1, c2, c3, c4)
    x = np.ones(2)
    
    cobyla_tester(G, C, x, mu)
