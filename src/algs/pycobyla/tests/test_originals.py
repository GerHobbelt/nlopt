import numpy as np
import pytest

from pycobyla import Cobyla


def test_problem_1():
    '''
    Test problem 1 (Minimization of a simple quadratic function of two variables)

    F(x, y) = (10 * ((x + 1)^2)) + y^2;
    
    '''
    x = np.ones(2)
    F = lambda x: (10 * ((x[0] + 1) ** 2)) + (x[1] ** 2)
    C = ()
    rhobeg = .5
    rhoend = 1e-6
    maxfun = 3500

    opt = Cobyla(x, F, C, rhobeg=rhobeg, rhoend=rhoend, maxfun=maxfun)
    opt.run()
    
    knwon_x = np.array((-1, 0), dtype=opt.float)
    error = sum((opt.x - knwon_x) ** 2)
    assert error < 1e-6


def test_problem_2():
    '''
    Test problem 2 (2D unit circle calculation)

    F(x, y) = x * y
    C1(x, y) = 1 - x^2 - y^2 >= 0 

    '''
    x = np.ones(2)
    F = lambda x: (x[0]* x[1]) 
    c1 = lambda x: 1 - (x[0] ** 2) - (x[1] ** 2)
    C = (c1,)
    rhobeg = .5
    rhoend = 1e-8
    maxfun = 3500

    opt = Cobyla(x, F, C, rhobeg=rhobeg, rhoend=rhoend, maxfun=maxfun)
    opt.run()

    knwon_x = np.array((1 / (2 ** .5), -1 / (2 ** .5)), dtype=opt.float)
    error = sum((opt.x - knwon_x) ** 2)
    assert error < 1e-6


def test_problem_3():
    '''
    Test problem 3 

    F(x, y) = 
    C1(x, y) =

    '''
    x = np.ones(3)
    F = lambda x: x[0]* x[1] * x[2]
    c1 = lambda x: 1 - (x[0] ** 2) - (2 * (x[1] ** 2))  - (3 * (x[2] ** 2))
    C = (c1,)
    rhobeg = .5
    rhoend = 1e-6
    maxfun = 3500

    opt = Cobyla(x, F, C, rhobeg=rhobeg, rhoend=rhoend, maxfun=maxfun)
    opt.run()

    knwon_x = np.array(((1 / (3 ** .5), 1 / (6 ** .5), -1 / 3)), dtype=opt.float)
    error = sum((opt.x - knwon_x) ** 2)
    assert error < 1e-6
    
