import numpy as np
import pytest

from pycobyla import Cobyla


def test_problem_2():
    '''
    Test problem 2 (2D unit circle calculation)

    F(x, y) = x * y
    C1(x, y) = 1 - x^2 - y^2 >= 0 
    
    '''
    x = np.array((1, 1), dtype=np.float)
    F = lambda x: (x[0]* x[1]) 
    c1 = lambda x: 1 - (x[0] ** 2) - (x[1] ** 2)
    C = (c1,)
    rhobeg = .5
    rhoend = 1e-6
    maxfun = 3500

    opt = Cobyla(x, F, C, rhobeg=rhobeg, rhoend=rhoend, maxfun=maxfun)
    opt.run()

    knwon_x = np.array((.5 ** 2, - 1 / (2 **.5)))
    error = sum((opt.x - knwon_x) ** 2)
    
    assert  error< 1e-6
    
