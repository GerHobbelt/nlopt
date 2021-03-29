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


def cone(x, a=1, b=1):
    return (((x[0] / a) ** 2) + ((x[1] / b) ** 2)) ** .5


def paraboloid(x, a=1, b=1):
    return ((x[0] / a) ** 2) + ((x[1] / b) ** 2)
        

def test_problem_gaussian_2d():
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


def test_problem_gaussian_2d_random_mu():
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


def test_problem_two_gaussian_2d():
    '''
    mu1 = (-0.5, -0.5)
    mu2 = (0.5, 0.5)
    sig1 = sig2 = (1, 1)
    
    C1(x, y) = 5 - x >= 0
    C2(x, y) = 5 + x >= 0
    C3(x, y) = 5 - y >= 0
    C4(x, y) = 5 + y >= 0
    
    '''
    mu1 = np.array((-2.5, -2.5))
    mu2 = np.array((2.5, 2.5))
    
    G1 = functools.partial(neg_gaussian, mu=mu1, A=1)
    G2 = functools.partial(neg_gaussian, mu=mu2, A=2)
    G = lambda x: G1(x) + G2(x)

    k = 5
    c1 = lambda x: k - x[0]
    c2 = lambda x: k + x[0]
    c3 = lambda x: k - x[1]
    c4 = lambda x: k + x[1]
    
    C = (c1, c2, c3, c4)
    x = np.ones(2)
    
    cobyla_tester(G, C, x, mu2)


def test_problem_cone_unconstrains():
    '''
    Test cone
    F(x, y) = (x^2 + y^2) ** .5
    
    '''
    F = lambda x: ((x[0] ** 2) + (x[1] ** 2)) ** .5
    C = ()
    
    x = np.ones(2)
    known_x = np.zeros(2)

    cobyla_tester(cone, C, x, known_x)


def test_problem_cone_constrains():
    '''
    Test cone
    F(x, y) = (x^2 + y^2) ** .5
    
    '''
    F = lambda x: ((x[0] ** 2) + (x[1] ** 2)) ** .5
    k = 5
    c1 = lambda x: k - x[0]
    c2 = lambda x: k + x[0]
    c3 = lambda x: k - x[1]
    c4 = lambda x: k + x[1]
    C = (c1, c2, c3, c4)
    
    x = np.ones(2)
    known_x = np.zeros(2)

    cobyla_tester(cone, C, x, known_x)


def test_problem_shifted_cone_constrains():
    '''
    Test cone
    F(x, y) = ((x - x0)^2 + (y - y0)^2) ** .5
    
    '''
    x0 = np.random.random(2)
    F = lambda x: cone(x - x0)
    
    k = 5
    c1 = lambda x: k - x[0]
    c2 = lambda x: k + x[0]
    c3 = lambda x: k - x[1]
    c4 = lambda x: k + x[1]
    C = (c1, c2, c3, c4)
    
    x = np.ones(2)

    cobyla_tester(F, C, x, x0)


def test_problem_paraboloid():
    '''
    Test paraboloid

    F(x, y) = (x / a)^2 + (y / b)^2
    
    '''
    x = np.ones(2)
    C = ()
    known_x = np.zeros(2)
    
    cobyla_tester(paraboloid, C, x, known_x)


def test_problem_shifted_paraboloid():
    '''
    Test paraboloid

    F(x, y) = (x / a)^2 + (y / b)^2
    
    '''
    x0 = np.random.random(2)
    F = lambda x: paraboloid(x - x0)
    C = ()
    x = np.ones(2)
    
    cobyla_tester(F, C, x, x0)
    

def test_stop_fault():
    F = lambda x: (2 - np.cos(x[0]) + x[1] ** 2) ** 2
    C = ()
    
    x = np.ones(2)
    known_x = np.zeros(2)

    cobyla_tester(F, C, x, known_x, maxfun=667)
    cobyla_tester(F, C, x, known_x, maxfun=668)

    



