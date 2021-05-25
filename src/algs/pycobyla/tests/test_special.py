import functools

import pytest
import numpy as np
import nlopt

from pycobyla import Cobyla

import tests.test_custom as tc
from tests.test_originals import cobyla_tester
    

def test_pyramid_bad_optimization_due_to_data_precision():
    '''
    Test pyramid seems not work well in certain cases when data is not well conditioned

    '''
    F = functools.partial(tc.pyramid, center=np.zeros(2), width=2, height=-1)
    C = ()
    x = np.array((0.94388211340220107281596, -0.61268428625789606023488))
    opt = Cobyla(x, F, C, rhobeg=.5, rhoend=1e-8, maxfun=3500)
    opt.run()

    print(f'\nOriginal: {x}')
    print(f'Optimize: {opt.x}')
    print(f'Error: {sum(opt.x ** 2)}')


def test_4_faces_pyramid_bad_optimization_due_to_data_precision():
    '''
    '''
    radius = 2
    center = np.zeros(2)
    F = functools.partial(tc.pyramid_faces, center=center, radius=radius, height=-1, faces=4)
    c1 = lambda x: 1 - sum(x ** 2)
    C = (c1,)
    x = np.array((0.56699681246957212010784, 0.13886994284481257722064))
    opt = Cobyla(x, F, C, rhobeg=.5, rhoend=1e-17, maxfun=3500)
    opt.run()

    print(f'\nOriginal: {x}')
    print(f'Optimize: {opt.x}')
    print(f'Error: {sum((opt.x - center) ** 2) ** .5}')


def test_2_planes_bad_optimization():
    '''
    '''
    n1 = np.array((1, 1, 1))
    n2 = np.array((-1, 1, 1))
    p = np.zeros(3)

    h1 = functools.partial(tc.plane, n=n1, p=p)
    h2 = functools.partial(tc.plane, n=n2, p=p)

    F = lambda x: -h1(x) if x[0] >= 0 else -h2(x)
    c1 = lambda x: x[1]
    C = (c1,)

    x = np.array((0.23243240513870577768074, 2.66331299470191806832986))
    opt = Cobyla(x, F, C, rhobeg=.5, rhoend=1e-12, maxfun=3500)
    opt.run()

    print(f'\nOriginal: {x}')
    print(f'Optimize: {opt.optimal_vertex}')
    print(f'Error: {sum((opt.x - np.zeros(2)) ** 2) ** .5}')

    FF = lambda x, _grad: F(x)
    nlopt_opt = nlopt.opt(nlopt.LN_COBYLA, 2)
    nlopt_opt.set_min_objective(FF)
    nlopt_opt.add_inequality_constraint(lambda x, *_: -c1(x))
    nlopt_opt.set_maxeval(1500)
    known_optimized = nlopt_opt.optimize(x)
    print(f'nlopt: {known_optimized}')
    

@pytest.mark.skip
def test_4_faces_pyramid_bad_optimization_loop():
    '''
    '''
    TOL = 1e-2
    counter = total = 0

    radius = 2
    F = functools.partial(tc.pyramid_faces, center=(0, 0), radius=radius, height=-1, faces=4)
    c1 = lambda x: 1 - sum(x ** 2)
    C = (c1, )
    known_x = np.zeros(2)

    print(f'\nError > {TOL}')
    while (True):
        total += 1
        x = np.random.uniform(low=-1, high=1, size=2)
        opt = Cobyla(x, F, C, rhobeg=.5, rhoend=1e-12, maxfun=3500)
        opt.run()
    
        error = sum((opt.x - known_x) ** 2) ** .5
        if error > TOL:
            counter += 1
            print(f'  - [{x[0]:.23f}, {x[1]:.23f},'
                  f' {opt.x[0]:.23f}, {opt.x[1]:.23f},'
                  f' {-F((opt.x)):.3f},'
                  f' {counter}, {total}, {(counter / total) * 100:02.2f}]')


@pytest.mark.skip
def test_8_faces_pyramid_bad_optimization_loop():
    '''
    '''
    TOL = 1e-11
    counter = total = 0

    radius = 2
    F = functools.partial(tc.pyramid_faces, center=(0, 0), radius=radius, height=-1, faces=8)
    C = ()
    known_x = np.zeros(2)

    print(f'\nError > {TOL}')
    while (True):
        total += 1
        x = np.random.uniform(low=-1, high=1, size=2)
        opt = Cobyla(x, F, C, rhobeg=.5, rhoend=1e-12, maxfun=3500)
        opt.run()
    
        error = sum((opt.x - known_x) ** 2) ** .5
        if error > TOL:
            counter += 1
            print(f'  - [{x[0]:.23f}, {x[1]:.23f},'
                  f' {opt.x[0]:.23f}, {opt.x[1]:.23f},'
                  f' {-F((opt.x)):.3f},'
                  f' {counter}, {total}, {(counter / total) * 100:02.2f}]')


@pytest.mark.skip
def test_pyramid_bad_optimization_loop():
    '''
    '''
    TOL = 1e-1
    counter = total = 0

    width = 2
    F = functools.partial(tc.pyramid, center=np.zeros(2), width=width, height=-1)
    C = ()
    known_x = np.zeros(2)

    print(f'\nError > {TOL}')
    while (True):
        total += 1
        x = np.random.uniform(low=-(width/2), high=(width/2), size=2)
        opt = Cobyla(x, F, C, rhobeg=.5, rhoend=1e-8, maxfun=3500)
        opt.run()
    
        error = sum((opt.x - known_x) ** 2) ** .5
        if error > TOL:
            counter += 1
            print(f'  - [{x[0]:.23f}, {x[1]:.23f},'
                  f' {opt.x[0]:.23f}, {opt.x[1]:.23f},'
                  f' {-F((opt.x)):.3f},'
                  f' {counter}, {total}, {(counter / total) * 100:02.2f}]')


@pytest.mark.skip
def test_pyramid_bad_optimization_loop_with_circle_constrain():
    '''
    '''
    TOL = 1e-1
    counter = total = 0

    width = 2
    F = functools.partial(tc.pyramid, center=np.zeros(2), width=width, height=-1)
    c1 = lambda x: .3 - sum(x ** 2)
    C = (c1,)
    known_x = np.zeros(2)

    print(f'\nError > {TOL}')
    while (True):
        total += 1
        x = np.random.uniform(low=-(width/2), high=(width/2), size=2)
        opt = Cobyla(x, F, C, rhobeg=.5, rhoend=1e-8, maxfun=3500)
        opt.run()
    
        error = sum((opt.x - known_x) ** 2) ** .5
        if error > TOL:
            counter += 1
            print(f'  - [{x[0]:.23f}, {x[1]:.23f},'
                  f' {opt.x[0]:.23f}, {opt.x[1]:.23f},'
                  f' {-F((opt.x)):.3f},'
                  f' {counter}, {total}, {(counter / total) * 100:02.2f}]')


@pytest.mark.skip
def test_2_planes_bad_optimization_loop():
    '''
    '''
    TOL = 1e-1
    counter = total = 0

    n1 = np.array((1, 1, 1))
    n2 = np.array((-1, 1, 1))
    p = np.zeros(3)

    h1 = functools.partial(tc.plane, n=n1, p=p)
    h2 = functools.partial(tc.plane, n=n2, p=p)

    F = lambda x: -h1(x) if x[0] >= 0 else -h2(x)
    c1 = lambda x: x[1]
    C = (c1,)
    known_x = np.zeros(2)

    print(f'\nError > {TOL}')
    while (True):
        total += 1
        x = np.random.uniform(low=0, high=3, size=2)
        opt = Cobyla(x, F, C, rhobeg=.5, rhoend=1e-8, maxfun=3500)
        opt.run()
    
        error = sum((opt.x - known_x) ** 2) ** .5
        if error > TOL:
            counter += 1
            print(f'  - [{x[0]:.23f}, {x[1]:.23f},'
                  f' {opt.x[0]:.23f}, {opt.x[1]:.23f},'
                  f' {F((opt.x)):.3f},'
                  f' {counter}, {total}, {(counter / total) * 100:02.2f}]')
