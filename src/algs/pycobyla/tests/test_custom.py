import functools

import numpy as np
import pytest

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
        

def pyramid(x, center=np.zeros(2), width=2, height=1):
    ww = width / 2
    xx, yy = cc = np.array(x) - np.array(center)
    abs_x, abs_y = abs(cc)
    
    if (yy <= xx) and (yy > -xx):
        # sector 1
        hh = (1 - (abs_x / ww))
        
    elif (yy > xx) and (yy > -xx):
        # sector 2
        hh = (1 - (abs_y / ww))
        
    elif (yy > xx) and (yy <= -xx):
        # sector 3
        hh = (1 - (abs_x / ww))
        
    else:
        # sector 4
        hh = (1 - (abs_y / ww))

    return hh * height if 0 < hh <= 1 else 0


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


def test_powell_paper_bad_convergence_problem():
    '''
    Test

    F(x) = -abs(x - 3)

    C1(x) = 0.25 - abs(x) >= 0
    
    '''
    F = lambda x: -abs(x[0] - 3)
    c1 = lambda x: 0.25 - abs(x)

    C = (c1,)
    x = np.ones(1)
    known_x = np.array((-0.25,))

    opt = cobyla_tester(F, C, x, known_x)


def test_pyramid_problem():
    '''
    Test pyramid
    
    '''
    F = lambda x: -pyramid(x, center=np.zeros(2), width=2, height=1)

    C = ()
    x = np.array((0.5 - 1e-16, 0))
    known_x = np.zeros(2)

    opt = cobyla_tester(F, C, x, known_x)


@pytest.mark.skip('This problem has very bad response')
def test_pyramid_problem_fails():
    '''
    Test pyramid with bad response
    
    '''
    F = lambda x: -pyramid(x, center=np.zeros(2), width=2, height=1)

    C = ()
    x = np.array((0.8418772017014113373534200945869088173, 0.8139157946609998361964244395494461060))
    known_x = np.zeros(2)

    opt = cobyla_tester(F, C, x, known_x, tol=1e-6)
    

@pytest.mark.skip('Better with constrain but still poor')
def test_pyramid_random_safe_start_problem():
    '''
    Test pyramid with random start position
    
    '''
    F = lambda x: -pyramid(x, center=np.zeros(2), width=2, height=1)
    c1 = lambda x: .5 - sum(x ** 2)
    
    C = (c1,)
    x = np.random.uniform(low=-1, high=1, size=2)
    known_x = np.zeros(2)

    opt = cobyla_tester(F, C, x, known_x, tol=1e-1)

