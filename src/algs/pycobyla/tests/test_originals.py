import numpy as np
import pytest

from pycobyla import Cobyla


def cobyla_tester(F, C, x, knwon_x, rhobeg=.5, rhoend=1e-6, maxfun=3500):
    opt = Cobyla(x, F, C, rhobeg=rhobeg, rhoend=rhoend, maxfun=maxfun)
    opt.run()
    error = sum((opt.x - knwon_x) ** 2)
    assert error < 1e-6
    
    return opt


def test_problem_1():
    '''
    Test problem 1 (Minimization of a simple quadratic function of two variables)

    F(x, y) = (10 * ((x + 1)^2)) + y^2;
    
    '''
    F = lambda x: (10 * ((x[0] + 1) ** 2)) + (x[1] ** 2)
    C = ()
    x = np.ones(2)
    known_x = np.array((-1, 0))

    cobyla_tester(F, C, x, known_x)



def test_problem_2():
    '''
    Test problem 2 (2D unit circle calculation)

    F(x, y) = x * y
    C1(x, y) = 1 - x^2 - y^2 >= 0 

    '''
    F = lambda x: (x[0]* x[1]) 
    c1 = lambda x: 1 - (x[0] ** 2) - (x[1] ** 2)
    C = (c1,)
    x = np.ones(2)
    known_x = np.array((1 / (2 ** .5), -1 / (2 ** .5)))
    rhoend = 1e-8

    cobyla_tester(F, C, x, known_x, rhoend=rhoend)



def test_problem_3():
    '''
    Test problem 3 (3D ellipsoid calculation)

    F(x, y, z) = x
    C1(x, y, z) = 1 - x^2 - (2 * y^2) - (3 * z^2) 

    '''
    F = lambda x: x[0]* x[1] * x[2]
    c1 = lambda x: 1 - (x[0] ** 2) - (2 * (x[1] ** 2))  - (3 * (x[2] ** 2))
    C = (c1,)
    x = np.ones(3)
    known_x = np.array(((1 / (3 ** .5), 1 / (6 ** .5), -1 / 3)))

    cobyla_tester(F, C, x, known_x)


def test_problem_4():
    '''
    Test problem 4 (Weak version of Rosenbrock's problem)

    F(x, y) = (x^2 - y)^2 + (x + 1)^2
    
    '''
    F = lambda x: (((x[0] ** 2) - x[1]) ** 2) + ((x[0] + 1) ** 2)
    C = ()
    x = np.ones(2)
    known_x = np.array((-1, 1))

    cobyla_tester(F, C, x, known_x)


def test_problem_5():
    '''
    Test problem 5 (Intermediate Rosenbrock)

    F(x, y) =  (10 * ((x^2 - y)^2)) + (x + 1)^2
    
    '''
    F = lambda x: (10 * (((x[0] ** 2) - x[1]) ** 2 )) + ((x[0] + 1) ** 2)
    C = ()
    x = np.ones(2)
    known_x = np.array((-1, 1))

    cobyla_tester(F, C, x, known_x)


def test_problem_6():
    '''
    Test problem 6 (Equation (9.1.15) in Fletcher's book)
    F(x, y) = - x - y
    C1(x, y) >= z -x^2
    C2(x, y) >= 1 - x^2 - y^2

    This problem is taken from Fletcher's book Practical Methods of
    Optimization and has the equation number (9.1.15)
    
    '''
    F = lambda x: - x[0] - x[1]
    c1 = lambda x: x[1] - (x[0] ** 2)
    c2 = lambda x: 1 - (x[0] ** 2) - (x[1] ** 2)
    C = (c1, c2)
    x = np.ones(2)
    known_x = np.array(((.5 ** .5), (.5 ** .5)))

    cobyla_tester(F, C, x, known_x, maxfun=15000)

    
def test_problem_7():
    '''
    Test problem 7 (Equation (14.4.2) in Fletcher's book)
    F(x, y, z) = z
    C1(x, y, z) >= (5 * x) - y + z
    C2(x, y, z) >= z - (5 * x) - y
    
    '''
    F = lambda x: x[2]
    c1 = lambda x: (5 * x[0]) - x[1] + x[2]
    c2 = lambda x: x[2] - (x[0] ** 2) - (x[1] ** 2) - (4 * x[1])
    c3 = lambda x: x[2] - (5 * x[0]) - x[1]
    C = (c1, c2, c3)
    x = np.ones(3)
    known_x = np.array((0, -3, -3))

    cobyla_tester(F, C, x, known_x)
    
