import functools

import numpy as np

from pycobyla import Cobyla

import tests.test_custom as tc
from tests.test_originals import cobyla_tester


def test_pyramid_bad_optimization_due_to_data_precision():
    '''
    Test pyramid seems not work well in certain cases when data is not well conditioned

    See: L440_update_simplex, JSX tag
    '''
    F = lambda x: -tc.pyramid(x, center=np.zeros(2), width=2, height=1)
    C = ()
    x = np.array((0.94388211340220107281596, -0.61268428625789606023488))
    opt = Cobyla(x, F, C, rhobeg=.5, rhoend=1e-8, maxfun=3500)
    opt.run()

    print(opt.x)
    

def test_4_faces_pyramid_bad_optimization_loop():
    TOL = 1e-3
    counter = total = 0

    radius = 2 * (2 **.5)
    F = functools.partial(tc.pyramid_faces, center=(0, 0), radius=radius, height=-1, faces=4)
    C = ()
    known_x = np.zeros(2)

    print(f'\nError > {TOL}')
    while (True):
        total += 1
        x = np.random.uniform(low=-1, high=1, size=2)
        opt = Cobyla(x, F, C, rhobeg=.5, rhoend=1e-8, maxfun=3500)
        opt.run()
    
        error = sum((opt.x - known_x) ** 2) ** .5
        if error > TOL:
            counter += 1
            print(f'  - [{x[0]:.23f}, {x[1]:.23f},'
                  f' {opt.x[0]:.23f}, {opt.x[1]:.23f},'
                  f' {-F((opt.x)):.3f},'
                  f' {counter}, {total}, {(counter / total) * 100:02.2f}]')

            
def test_8_faces_pyramid_bad_optimization_loop():
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


def test_pyramid_bad_optimization_loop():
    TOL = 1e-1
    MAX_ITER = 500
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


def test_pyramid_bad_optimization_loop_with_circle_constrain():
    TOL = 1e-1
    MAX_ITER = 500
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
