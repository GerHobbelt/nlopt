import sys
import yaml
import logging
import pathlib
from dataclasses import dataclass

import numpy as np
import pytest

import pycobyla
from pycobyla import Cobyla


logger = logging.getLogger(__name__)
logger.propagate = False
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)


RHOBEG = .5
RHOEND = 1e-12


    
@dataclass
class Result:
    nfvals: int
    fmin: float
    x: np.array
    error: float
    
    
def opt_info(opt, error):
    logger.info('')
    logger.info(f'nfvals: {opt.nfvals}')
    logger.info(f'fmin: {opt.fmin:}')
    logger.info(f'x: {opt.x}')
    logger.info(f'error: {error}')
    
    
def cobyla_tester(F, C, x, known_x, rhobeg=RHOBEG, rhoend=RHOEND, maxfun=7500, tol=1e-8):
    opt = Cobyla(x, F, C, rhobeg=rhobeg, rhoend=rhoend, maxfun=maxfun)
    opt.run()
    
    error = sum((opt.x - known_x) ** 2) ** .5
    opt_info(opt, error)
    assert error < tol

    return opt, error


def test_problem_1():
    '''
    Test problem 1 (Minimization of a simple quadratic function of two variables)

    F(x, y) = (10 * ((x + 1)^2)) + y^2;
    
    '''
    F = lambda x: (10 * ((x[0] + 1) ** 2)) + (x[1] ** 2)
    C = ()
    x = np.ones(2)
    known_x = np.array((-1, 0))

    opt, error = cobyla_tester(F, C, x, known_x)


def test_problem_2():
    '''
    Test problem 2 (2D unit circle calculation)

    F(x, y) = x * y
    C1(x, y) = 1 - x^2 - y^2 >= 0

    '''
    F = lambda x: (x[0] * x[1]) 
    c1 = lambda x: 1 - (x[0] ** 2) - (x[1] ** 2)
    C = (c1,)
    x = np.ones(2)
    known_x = np.array((1 / (2 ** .5), -1 / (2 ** .5)))

    opt, error = cobyla_tester(F, C, x, known_x)


def test_problem_3():
    '''
    Test problem 3 (3D ellipsoid calculation)

    F(x, y, z) = x
    C1(x, y, z) = 1 - x^2 - (2 * y^2) - (3 * z^2) >= 0

    '''
    F = lambda x: x[0] * x[1] * x[2]
    c1 = lambda x: 1 - (x[0] ** 2) - (2 * (x[1] ** 2))  - (3 * (x[2] ** 2))
    C = (c1,)
    x = np.ones(3)
    known_x = np.array(((1 / (3 ** .5), 1 / (6 ** .5), -1 / 3)))

    opt, error = cobyla_tester(F, C, x, known_x)


def test_problem_4():
    '''
    Test problem 4 (Weak version of Rosenbrock's problem)

    F(x, y) = (x^2 - y)^2 + (x + 1)^2
    
    '''
    F = lambda x: (((x[0] ** 2) - x[1]) ** 2) + ((x[0] + 1) ** 2)
    C = ()
    x = np.ones(2)
    known_x = np.array((-1, 1))

    opt, error = cobyla_tester(F, C, x, known_x)


def test_problem_5():
    '''
    Test problem 5 (Intermediate Rosenbrock)

    F(x, y) =  (10 * ((x^2 - y)^2)) + (x + 1)^2
    
    '''
    F = lambda x: (10 * (((x[0] ** 2) - x[1]) ** 2 )) + ((x[0] + 1) ** 2)
    C = ()
    x = np.ones(2)
    known_x = np.array((-1, 1))

    opt, error = cobyla_tester(F, C, x, known_x)


def test_problem_6():
    '''
    Test problem 6 (Equation (9.1.15) in Fletcher's book)
    F(x, y) = - x - y
    C1(x, y) = y -x^2 >= 0
    C2(x, y) = 1 - x^2 - y^2 >= 0

    This problem is taken from Fletcher's book Practical Methods of
    Optimization and has the equation number (9.1.15)
    
    '''
    F = lambda x: - x[0] - x[1]
    c1 = lambda x: x[1] - (x[0] ** 2)
    c2 = lambda x: 1 - (x[0] ** 2) - (x[1] ** 2)
    C = (c1, c2)
    x = np.ones(2)
    known_x = np.array(((.5 ** .5), (.5 ** .5)))

    opt, error = cobyla_tester(F, C, x, known_x)

    
def test_problem_7():
    '''
    Test problem 7 (Equation (14.4.2) in Fletcher's book)
    F(x, y, z) = z
    C1(x, y, z) = (5 * x) - y + z >= 0
    C2(x, y, z) = z - (5 * x) - y >= 0

    This problem is taken from Fletcher's book Practical Methods of
    Optimization and has the equation number (14.4.2)
    
    '''
    F = lambda x: x[2]
    c1 = lambda x: (5 * x[0]) - x[1] + x[2]
    c2 = lambda x: x[2] - (x[0] ** 2) - (x[1] ** 2) - (4 * x[1])
    c3 = lambda x: x[2] - (5 * x[0]) - x[1]
    C = (c1, c2, c3)
    x = np.ones(3)
    known_x = np.array((0, -3, -3))

    opt, error = cobyla_tester(F, C, x, known_x)


def test_problem_8():
    '''
    Test problem 8 (Rosen-Suzuki)
    F(x, y, z, v) = x^2 + y^2 + (2 * (z^2)) + v^2 - 5*x - 5*y - 21*z + 7v
    C1(x, y, z, v) = 8 - x^2 - y^2 - z^3 - v^4 - x + y - z + v >= 0
    C2(x, y, z, v) = 10 - x^2 - (2 * y^2) - z^2 - (2 * v^2) + x + v >= 0
    C3(x, y, z, v) = 5 - (2 * x^2) - y^2 - z^2 - (2 * x) + y + v >= 0

    This problem is taken from page 66 of Hock and Schittkowski's book Test
    Examples for Nonlinear Programming Codes. It is their test problem
    Number 43, and has the name Rosen-Suzuki
    
    '''
    F = lambda x: (x[0] ** 2) + (x[1] ** 2) + (2 * (x[2] ** 2)) + (x[3] ** 2) - (5 * x[0]) - (5 * x[1]) - (21 * x[2]) + (7 * x[3])
    c1 = lambda x: 8 - np.dot(x, x) - x[0] + x[1] - x[2] + x[3]
    c2 = lambda x: 10 - (x[0] ** 2) - (2 * (x[1] ** 2)) - (x[2] ** 2) - (2 * (x[3] ** 2)) + x[0] + x[3]
    c3 = lambda x: 5 - (2 * (x[0] ** 2)) - (x[1] ** 2) - (x[2] ** 2) - (2 * x[0]) + x[1] + x[3]
    C = (c1, c2, c3)
    x = np.ones(4)
    known_x = np.array((0, 1, 2, -1))

    opt, error = cobyla_tester(F, C, x, known_x)

    
def test_problem_9():
    '''
    Test problem 9 (Rosen-Suzuki)
    F(x0, x1, x2, x3, x4, x5. x6) =
    C1(x0, x1, x2, x3, x4, x5. x6) = >= 
    C2(x0, x1, x2, x3, x4, x5. x6) = >= 
    C3(x0, x1, x2, x3, x4, x5. x6) = >=
    C4(x0, x1, x2, x3, x4, x5. x6) = >=

    This problem is taken from page 111 of Hock and Schittkowski's
    book Test Examples for Nonlinear Programming Codes. It is their
    test problem Number 100
    
    '''
    F = lambda x: ((x[0] - 10) ** 2) + (5 * ((x[1] - 12) ** 2)) + (x[2] ** 4) \
        + (3 * ((x[3] - 11) ** 2)) + (10 * (x[4] ** 6)) + (7 * (x[5] ** 2)) \
        + (x[6] ** 4) - (4 * x[5] * x[6]) - (10 * x[5]) - (8 * x[6])
    
    c1 = lambda x: 127 - (2 * (x[0] ** 2)) - (3 * (x[1] ** 4)) - x[2] - (4 * (x[3] ** 2)) - (5 * x[4]) 
    c2 = lambda x: 282 - (7 * x[0]) - (3 * x[1]) - (10 * (x[2] ** 2)) - x[3] + x[4]
    c3 = lambda x: 196 - (23 * x[0]) - (x[1] ** 2) - (6 * (x[5] ** 2)) + (8 * x[6])
    c4 = lambda x: (-4 * (x[0] ** 2)) - (x[1] ** 2) + (3 * x[0] * x[1]) - (2 * (x[2] ** 2)) - (5 * x[5]) + (11 * x[6])
    C = (c1, c2, c3, c4)
    x = np.ones(7)
    known_x = np.array(
        (2.330499, 1.951372, -.4775414, 4.365726, -.624487, 1.038131, 1.594227)
    )
    
    opt, error = cobyla_tester(F, C, x, known_x, tol=1e-6)


def test_problem_10():
    '''
    Test problem 10 (Test problem 10 (Hexagon area))
    F(x0, x1, x2, x3, x4, x5, x6, x7, x8) = -((x0 * x3) - (x1 * x2) + (x2 * x8) - (x4 * x8) + (x4 * x7) - (x5 * x6)) / 2
    C1(x0, x1, x2, x3, x4, x5, x6, x7, x8) = 1 - (x2 ** 2) - (x3 ** 2) >= 0
    C2(x0, x1, x2, x3, x4, x5, x6, x7, x8) = 1 - (x8 ** 2) >= 0
    C3(x0, x1, x2, x3, x4, x5, x6, x7, x8) = 1 - (x4 ** 2) - (x5 ** 2) >= 0
    C4(x0, x1, x2, x3, x4, x5, x6, x7, x8) = 1 - (x0 ** 2) - ((x1 - x8) ** 2) >= 0
    C5(x0, x1, x2, x3, x4, x5, x6, x7, x8) = 1 - ((x0 - x4) ** 2) - ((x1 - x5) ** 2) >= 0
    C6(x0, x1, x2, x3, x4, x5, x6, x7, x8) = 1 - ((x0 - x6) ** 2) - ((x1 - x7) ** 2) >= 0
    C7(x0, x1, x2, x3, x4, x5, x6, x7, x8) = 1 - ((x2 - x4) ** 2) - ((x3 - x5) ** 2) >= 0
    C8(x0, x1, x2, x3, x4, x5, x6, x7, x8) = 1 - ((x2 - x6) ** 2) - ((x3 - x7) ** 2) >= 0
    C9(x0, x1, x2, x3, x4, x5, x6, x7, x8) = 1 - (x6 ** 2) - ((x7 - x8) ** 2) >= 0
    c10(x0, x1, x2, x3, x4, x5, x6, x7, x8) = (x0 * x3) - (x1 * x2) >= 0
    C11(x0, x1, x2, x3, x4, x5, x6, x7, x8) = x2 * x8 >= 0
    C12(x0, x1, x2, x3, x4, x5, x6, x7, x8) = - (x4 * x8) >= 0
    C13(x0, x1, x2, x3, x4, x5, x6, x7, x8) = (x4 * x7) - (x5 * x6) >= 0
    C14(x0, x1, x2, x3, x4, x5, x6, x7, x8) = x8 >= 0

    This problem is taken from page 415 of Luenberger's book Applied
    Nonlinear Programming. It is to maximize the area of a hexagon of
    unit diameter
    
    '''

    F = lambda x: -((x[0] * x[3]) - (x[1] * x[2]) + (x[2] * x[8]) - (x[4] * x[8]) + (x[4] * x[7]) - (x[5] * x[6])) / 2
    c1 = lambda x: 1 - (x[2] ** 2) - (x[3] ** 2)
    c2 = lambda x: 1 - (x[8] ** 2)
    c3 = lambda x: 1 - (x[4] ** 2) - (x[5] ** 2)
    c4 = lambda x: 1 - (x[0] ** 2) - ((x[1] - x[8]) ** 2)
    c5 = lambda x: 1 - ((x[0] - x[4]) ** 2) - ((x[1] - x[5]) ** 2)
    c6 = lambda x: 1 - ((x[0] - x[6]) ** 2) - ((x[1] - x[7]) ** 2)
    c7 = lambda x: 1 - ((x[2] - x[4]) ** 2) - ((x[3] - x[5]) ** 2)
    c8 = lambda x: 1 - ((x[2] - x[6]) ** 2) - ((x[3] - x[7]) ** 2)
    c9 = lambda x: 1 - (x[6] ** 2) - ((x[7] - x[8]) ** 2)
    c10 = lambda x: (x[0] * x[3]) - (x[1] * x[2])
    c11 = lambda x: x[2] * x[8]
    c12 = lambda x: - (x[4] * x[8])
    c13 = lambda x: (x[4] * x[7]) - (x[5] * x[6])
    c14 = lambda x: x[8]
    C = (c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14)
    x = np.ones(9)

    opt = Cobyla(x, F, C, rhobeg=RHOBEG, rhoend=RHOEND, maxfun=3500)
    opt.run()

    known_x = np.zeros(9)
    tempa = opt.x[[0,2,4,6]].sum()
    tempb = opt.x[[1,3,5,7]].sum()
    tempc = .5 / ((tempa * tempa + tempb * tempb) ** .5)
    tempd = tempc * (3 ** .5)
    known_x[0] = tempd * tempa + tempc * tempb
    known_x[1] = tempd * tempb - tempc * tempa
    known_x[2] = tempd * tempa - tempc * tempb
    known_x[3] = tempd * tempb + tempc * tempa

    known_x[4:] = known_x[:5]

    error = sum((opt.x - known_x) ** 2)
    opt_info(opt, error)
    assert error < 1e-6
    

