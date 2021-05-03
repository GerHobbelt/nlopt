import random
import functools

import numpy as np
import pytest

from pycobyla import Cobyla
import tests.test_custom  as tc

TOL = 1e-8
RHOEND = 1e-9


def test_problem_min_at_the_x_boundary():
    mu = np.array((1, np.random.random()))
    G = functools.partial(tc.neg_gaussian, mu=mu)
    c1 = lambda x: 1 - x[0]
    c2 = lambda x: 1 + x[0]
    c3 = lambda x: 1 - x[1]
    c4 = lambda x: 1 + x[1]

    C = (c1, c2, c3, c4)
    x = np.ones(2)

    opt = Cobyla(x, G, C, rhoend=RHOEND)
    opt.run()
    assert ((opt.x - mu) < TOL).all()


def test_problem_min_at_the_y_boundary():
    G = tc.neg_gaussian
    mu = np.array((np.random.random(), 1))
    G = functools.partial(tc.neg_gaussian, mu=mu)
    c1 = lambda x: 1 - x[0]
    c2 = lambda x: 1 + x[0]
    c3 = lambda x: 1 - x[1]
    c4 = lambda x: 1 + x[1]

    C = (c1, c2, c3, c4)
    x = np.ones(2)

    opt = Cobyla(x, G, C, rhoend=RHOEND)
    opt.run()
    assert ((opt.x - mu) < TOL).all()


def test_problem_min_at_the_edge_corner_boundary():
    G = tc.neg_gaussian
    mu = random.choice(((1,1), (-1, 1), (-1, 1), (-1, -1)))
    
    G = functools.partial(tc.neg_gaussian, mu=mu)
    c1 = lambda x: 1 - x[0]
    c2 = lambda x: 1 + x[0]
    c3 = lambda x: 1 - x[1]
    c4 = lambda x: 1 + x[1]

    C = (c1, c2, c3, c4)
    x = np.ones(2)

    opt = Cobyla(x, G, C, rhoend=RHOEND)
    opt.run()
    assert ((opt.x - mu) < TOL).all()


def test_problem_min_out_of_the_boundary():
    r_circle = lambda x, radius, sig: sig * (((radius ** 2) - (x ** 2)) ** 0.5)
    
    radius = 2
    sig = np.random.choice((-1, 1))
    x_circle = np.random.uniform(low=-radius, high=radius)
    y_circle = r_circle(x_circle, radius=radius, sig=sig)
    mu = np.array((x_circle, y_circle))
    
    G = functools.partial(tc.neg_gaussian, mu=mu)
    c1 = lambda x: 1 - x[0]
    c2 = lambda x: 1 + x[0]
    c3 = lambda x: 1 - x[1]
    c4 = lambda x: 1 + x[1]

    C = (c1, c2, c3, c4)
    x = np.ones(2)

    opt = Cobyla(x, G, C, rhoend=RHOEND)
    opt.run()
    print(f'mu: {mu}  /  opt.x: {opt.x}')

    
def test_problem_min_out_of_the_boundary_circle():
    r_circle = lambda x, radius, sig: sig * (((radius ** 2) - (x ** 2)) ** 0.5)
    
    radius = 2
    sig = np.random.choice((-1, 1))
    x_circle = np.random.uniform(low=-radius, high=radius)
    x_circle = radius * np.random.random()
    y_circle = r_circle(x_circle, radius=radius, sig=sig)
    mu = np.array((x_circle, y_circle))
    
    G = functools.partial(tc.neg_gaussian, mu=mu)
    c1 = lambda x: 1 - sum(x ** 2)

    C = (c1,)
    x = np.ones(2)

    opt = Cobyla(x, G, C, rhoend=RHOEND)
    opt.run()
    print(f'mu: {mu}  /  opt.x: {opt.x}')


