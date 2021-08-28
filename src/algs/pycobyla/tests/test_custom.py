import functools

import numpy as np
import nlopt
import pytest

from pycobyla import Cobyla

from tests.test_originals import cobyla_tester


def gaussian(x, mu=None, sig=None, A=1):
    x = np.array(x)
    nn = len(x)
    mu =  np.zeros(nn) if mu is None else mu
    sig = np.ones(nn) if sig is None else sig

    zz = ((((x - mu) / sig) ** 2) / 2).sum()
    return A * np.exp(-zz)


def cone(x, a=1, b=1):
    return (((x[0] / a) ** 2) + ((x[1] / b) ** 2)) ** .5


def paraboloid(x, a=1, b=1):
    return ((x[0] / a) ** 2) + ((x[1] / b) ** 2)


def plane(x, n, p): 
    res = 0 if (n[-1] == 0) else (-((n[:-1] @ (x - p[:-1])) / n[-1]) + p[-1])
    return res


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


def pyramid_faces(x, center=np.zeros(2), radius=1, height=1, faces=4):
    assert faces > 2

    xx, yy = cc = np.array(x) - np.array(center)
    phi = np.arctan2(yy, xx)

    stp = 2 * np.pi / faces
    a1 = np.floor(phi / stp) * stp
    a2 = a1 + stp
    
    q1 = radius * np.array((np.cos(a1), np.sin(a1)))
    q2 = radius * np.array((np.cos(a2), np.sin(a2)))
    uu = q2 - q1
    norm = np.linalg.norm(uu)
    vv = np.array((-uu[1], uu[0]))
    vv = vv / norm

    hh = (cc - q1) @ vv
    if hh < 0:
        return 0

    c1 = norm / 2  
    total = ((radius ** 2) - (c1 ** 2)) ** .5
    return (hh / total) * height


def logistic_bivariant_density(x, mu=None, sig=None):
    mu = np.zeros(len(x)) if mu is None else mu
    sig = np.ones(len(x)) if sig is None else sig

    sd = sig ** .5
    kk = -(x - mu) / sd
    res = 2 * np.exp(sum(kk)) / (sd.prod() * ((1 + np.exp(kk).sum()) ** 3))

    return res


def waves(x, T0=2*np.pi, A0=1, T1=2*np.pi, A1=1):
    w = (2 * np.pi)
    w0 = w / T0
    w1 = w / T1
    return (A0 * np.cos(w0 * x[0])) + (A1 * np.cos(w1 * x[1]))


def test_problem_gaussian_2d():
    '''
    Test gaussian 2d Gaussian with mu=(0,0), sig=(1,1) 

    G((x, y), mu=(0,0), sig=(1,1), A=1)
    C1(x, y) = 1 - x >= 0
    C2(x, y) = 1 + x >= 0
    C3(x, y) = 1 - y >= 0
    C4(x, y) = 1 + y >= 0
    
    '''
    G = functools.partial(gaussian, A=-1)
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
    G = functools.partial(gaussian, mu=mu, A=-1)
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
    
    G1 = functools.partial(gaussian, mu=mu1, A=-1)
    G2 = functools.partial(gaussian, mu=mu2, A=-2)
    G = lambda x: G1(x) + G2(x)

    border = 5
    c1 = lambda x: border - x[0]
    c2 = lambda x: border + x[0]
    c3 = lambda x: border - x[1]
    c4 = lambda x: border + x[1]
    
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


def test_problem_shifted_cone_constrains_fixed():
    '''
    Test cone
    F(x, y) = ((x - x0)^2 + (y - y0)^2) ** .5
    
    '''
    x0 = np.array((0.12322525149802754, 0.7876063619785026))
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

    cobyla_tester(F, C, x, known_x)


def test_pyramid_problem():
    '''
    Test pyramid
    
    '''
    F = lambda x: -pyramid(x, center=np.zeros(2), width=2, height=1)

    C = ()
    x = np.array((0.5 - 1e-16, 0))
    known_x = np.zeros(2)

    cobyla_tester(F, C, x, known_x)


def test_linear_and_nonlinear_programming_pag_544_prob_4():
    '''
    Chapter 16, problem 4 (page 544) from book Linear and Nonlinear Programming
    Authors: Stephen P. Nash, Ariela Sofer
    
    '''
    F = lambda x:  -1 / ((x[0] ** 2) + 1)
    c1 = lambda x: x[0] - 1

    C = (c1,)
    x = np.array((5,))
    known_x = np.array((1,))
    opt, *_ = cobyla_tester(F, C, x, known_x, tol=1e-12)

    
def test_linear_and_nonlinear_programming_pag_544_prob_4_random_start():
    '''
    Chapter 16, problem 4 (page 544) from book Linear and Nonlinear Programming
    Authors: Stephen P. Nash, Ariela Sofer
    
    '''
    F = lambda x:  -1 / ((x[0] ** 2) + 1)
    c1 = lambda x: x[0] - 1

    C = (c1,)
    x = np.random.uniform(low=-5, high=5, size=(1,))
    known_x = np.array((1,))
    opt, *_ = cobyla_tester(F, C, x, known_x, tol=1e-12)


def test_linear_and_nonlinear_programming_example_pag_435():
    '''
    Chapter 14.1 Quadratic programming example
    Author: David E. Luenberger
    
    '''
    F = lambda x: (2 * (x[0] ** 2)) + (x[0] * x[1]) + (x[1] ** 2) + (-12 * x[0]) + (-10 * x[1])
    c1 = lambda x: 4 - (x[0] + x[1])
    c2 = lambda x: x[0]
    c3 = lambda x: x[1]
    
    C =  (c1, c2, c3)
    x = np.zeros(2)
    known_x = np.array((3/2, 5/2))
    opt, *_ = cobyla_tester(F, C, x, known_x, tol=1e-8)


def test_nlopt_repo_issue_370():
    '''
    Issue: https://github.com/stevengj/nlopt/issues/370
    
    '''
    F = lambda x: (2 - np.cos(x[0]) + x[1] ** 2) ** 2
    C = ()
    x = np.zeros(2)

    opt = Cobyla(x, F, C, rhobeg=.5, rhoend=1e-12, maxfun=3500)
    opt.run()
    print(f'\nCobyla: {opt.x}')


def test_logistic_bivariant_density():
    '''
    Fundamentos de la programación lineal y optimización en redes
    Author: David Pujolar Morales
    
    '''
    mu = np.random.uniform(low=-1, high=1, size=2)
    F = lambda x: -logistic_bivariant_density(x, mu=mu)
    F(np.zeros(2))
    C = ()
    x = np.random.uniform(low=-1, high=1, size=2)
    opt = Cobyla(x, F, C, rhobeg=.5, rhoend=1e-12, maxfun=3500)
    opt.run()
    
    error = sum((opt.x - mu) ** 2) ** .5
    print(f'\nCobyla: {opt.x}')
    print(f'error: {error}')
    assert error < 1e-8


def test_2_peaks():
    '''
    '''
    mu1 = np.array((.5, .5))
    mu2 = np.array((-.5, -.5))
    sig = .25 * np.ones(2)
    G1 = functools.partial(gaussian, mu=mu1, sig=sig, A=-2)
    G2 = functools.partial(gaussian, mu=mu2, sig=sig, A=-1)
    F = lambda x: G1(x) + G2(x)
    k = 1
    c1 = lambda x: k - x[0]
    c2 = lambda x: k + x[0]
    c3 = lambda x: k - x[1]
    c4 = lambda x: k + x[1]
    C = (c1, c2, c3, c4)

    x = np.zeros(2)
    opt = Cobyla(x, F, C, rhobeg=.5, rhoend=1e-12, maxfun=3500)
    opt.run()

    error = sum((opt.x - mu1) ** 2) ** .5
    print(f'\nCobyla: {opt.x}')
    print(f'error: {error}')


def test_2_peaks_random_start():
    '''
    '''
    mu1 = np.array((.5, .5))
    mu2 = np.array((-.5, -.5))
    sig = .5 * np.ones(2)
    G1 = functools.partial(gaussian, mu=mu1, sig=sig, A=-2)
    G2 = functools.partial(gaussian, mu=mu2, sig=sig, A=-1)
    F = lambda x: G1(x) + G2(x)
    k = 1
    c1 = lambda x: k - x[0]
    c2 = lambda x: k + x[0]
    c3 = lambda x: k - x[1]
    c4 = lambda x: k + x[1]
    C = (c1, c2, c3, c4)

    x = np.random.uniform(low=-1, high=1, size=2)
    opt = Cobyla(x, F, C, rhobeg=.5, rhoend=1e-12, maxfun=3500)
    opt.run()

    error1 = sum((opt.x - mu1) ** 2) ** .5
    error2 = sum((opt.x - mu2) ** 2) ** .5
    print(f'\nStart: {x}')
    print(f'Cobyla: {opt.x}')
    print(f"mu1 error: {error1} {'*' if error1 < error2 else ''}")
    print(f"mu2 error: {error2} {'*' if error2 < error1 else ''}")


def test_wave_field_random_start():
    F = functools.partial(waves, T0=1, A0=1, T1=1, A1=1)
    k = 1
    c1 = lambda x: k - x[0]
    c2 = lambda x: k + x[0]
    c3 = lambda x: k - x[1]
    c4 = lambda x: k + x[1]
    C = (c1, c2, c3, c4)

    start_x = np.random.uniform(low=-.7, high=.7, size=2)
    centers = np.array(((.5, .5), (-.5, .5), (-.5, -.5), (.5, -.5)))
    dist = ((centers - start_x)**2).sum(axis=1)
    _, idx = min(zip(dist, range(len(dist))))
    known_x = centers[idx]
    
    opt = Cobyla(start_x, F, C, rhobeg=.5, rhoend=1e-12, maxfun=3500)
    opt.run()

    error = sum((opt.x - known_x) ** 2) ** .5
    print(f'\nStart: {start_x}')
    print(f'Cobyla: {opt.x}')
    print(f'Error: {error}')


@pytest.mark.skip('This problem has very bad response')
def test_pyramid_problem_fails():
    '''
    Test pyramid with bad response
    
    '''
    F = lambda x: -pyramid(x, center=np.zeros(2), width=2, height=1)

    C = ()
    x = np.array((0.8418772017014113373534200945869088173, 0.8139157946609998361964244395494461060))
    known_x = np.zeros(2)

    opt, *_ = cobyla_tester(F, C, x, known_x, tol=1e-6)
    

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

    opt, *_ = cobyla_tester(F, C, x, known_x, tol=1e-1)
