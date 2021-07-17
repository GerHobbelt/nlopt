import functools

import nlopt
import scipy.optimize
import pytest
import numpy as np

from pycobyla import Cobyla

import tests.data
import tests.test_custom as tc
from tests.test_originals import cobyla_tester
    

def test_pyramid_bad_optimization_due_to_data_precision():
    '''
    Test pyramid seems not work well in certain cases when data is not well conditioned

    '''
    center = np.zeros(2)
    F = functools.partial(tc.pyramid, center=center, width=2, height=-1)
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
    opt = Cobyla(x, F, C, rhobeg=.5, rhoend=1e-17, maxfun=700)
    opt.run()

    print(f'\nOriginal: {x}')
    print(f'Optimize: {opt.x}')
    print(f'Error: {sum((opt.x - center) ** 2) ** .5}')

    FF = lambda x, _grad: F(x)
    nlopt_opt = nlopt.opt(nlopt.LN_COBYLA, 2)
    nlopt_opt.set_min_objective(FF)
    nlopt_opt.add_inequality_constraint(lambda x, *_: -c1(x))
    nlopt_opt.set_maxeval(700)  # 750 iterations get blocked
    known_optimized = nlopt_opt.optimize(x)
    print(f'nlopt: {known_optimized}')
    print(f'Error: {sum((known_optimized - center) ** 2) ** .5}')

    
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
    print(f'Error: {sum((known_optimized - np.zeros(2)) ** 2) ** .5}')
    

@pytest.mark.skip
def test_4_faces_pyramid_bad_optimization_loop():
    '''
    '''
    w_error = 1e-2
    counter = total = 0

    radius = 2
    F = functools.partial(tc.pyramid_faces, center=(0, 0), radius=radius, height=-1, faces=4)
    c1 = lambda x: 1 - sum(x ** 2)
    C = (c1, )
    known_x = np.zeros(2)

    while (True):
        total += 1
        x = np.random.uniform(low=-1, high=1, size=2)
        opt = Cobyla(x, F, C, rhobeg=.5, rhoend=1e-12, maxfun=3500)
        opt.run()

        error = sum((opt.x - known_x) ** 2) ** .5
        if error > w_error:
            w_error = error
            counter += 1
            print(f'  - [{x[0]:.23f}, {x[1]:.23f},'
                  f' {opt.x[0]:.23f}, {opt.x[1]:.23f},'
                  f' {-opt.F((opt.x)):.3f},'
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

            
@pytest.mark.skip
def test_2_planes_bad_optimization_loop_nlopt():
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

        FF = lambda x, _grad: F(x)
        nlopt_opt = nlopt.opt(nlopt.LN_COBYLA, 2)
        nlopt_opt.set_min_objective(FF)
        nlopt_opt.add_inequality_constraint(lambda x, *_: -c1(x))
        nlopt_opt.set_maxeval(600)  # Up to 800 get stuck
        known_optimized = nlopt_opt.optimize(x)
    
        error = sum((known_optimized - known_x) ** 2) ** .5
        if error > TOL:
            counter += 1
            print(f'  - [{x[0]:.23f}, {x[1]:.23f},'
                  f' {known_optimized[0]:.23f}, {known_optimized[1]:.23f},'
                  f' {F(known_optimized):.3f},'
                  f' {counter}, {total}, {(counter / total) * 100:02.2f}]')

            
@pytest.mark.skip
def test_waves_field():
    '''
    '''
    T0, A0 = 1, 1
    T1, A1 = 1, 1
    border = 1

    from pycobyla.cobyla import logger
    logger.disabled = True
    
    F = functools.partial(tc.waves, T0=T0, A0=A0, T1=T1, A1=A1)
    c1 = lambda x: border - x[0]
    c2 = lambda x: border + x[0]
    c3 = lambda x: border - x[1]
    c4 = lambda x: border + x[1]
    C = (c1, c2, c3, c4)

    print('')
    while (True):
        start_x = np.random.uniform(low=-1, high=1, size=2)
        opt = Cobyla(start_x, F, C, rhobeg=.1, rhoend=1e-12, maxfun=6000)
        try:
            opt.run()
        except UserWarning:
            print(f'({start_x[0]:.23f}, {start_x[1]:.23f}),')

            
def test_waves_scipy():
    T0, A0 = 1, 1
    T1, A1 = 1, 1
    start_x = (-0.32951277043785864862002, 0.97237533384796859259325)
    border = 1
    rhobeg = .1
    rhoend = 1e-12
    maxiter = 6000

    F = functools.partial(tc.waves, T0=T0, A0=A0, T1=T1, A1=A1)
    c1 = lambda x: border - x[0]
    c2 = lambda x: border + x[0]
    c3 = lambda x: border - x[1]
    c4 = lambda x: border + x[1]
    C = (c1, c2, c3, c4)
    
    opt = Cobyla(start_x, F, C, rhobeg=rhobeg, rhoend=rhoend, maxfun=maxiter)

    try:
        opt.run()
    except UserWarning:
        pass
    
    print(f'PyCobyla: {opt.F(opt.optimal_vertex)}\n')

    
    res = scipy.optimize.minimize(
        opt.F, start_x, args=(), method='COBYLA', constraints=(), 
        options={'rhobeg': rhobeg, 'maxiter': maxiter, 'disp': False, 'tol': rhoend}
    )
    print(res)


def test_bad_waves_field_scipy():
    '''
    This test check that scypi.optimize COBYLA is not working well
    with detected points
    
    '''
    T0, A0 = 1, 1
    T1, A1 = 1, 1
    border = 1
    rhobeg = .1
    rhoend = 1e-12
    maxiter = 6000

    F = functools.partial(tc.waves, T0=T0, A0=A0, T1=T1, A1=A1)
    c1 = lambda x: border - x[0]
    c2 = lambda x: border + x[0]
    c3 = lambda x: border - x[1]
    c4 = lambda x: border + x[1]
    C = (c1, c2, c3, c4)

    for start_x in tests.data.BAD_WAVES_FIELD_START_X:
        res = scipy.optimize.minimize(
            F, start_x, args=(), method='COBYLA', constraints=(), 
            options={'rhobeg': rhobeg, 'maxiter': maxiter, 'disp': False, 'tol': rhoend}
        )
        
        norm = sum(res.x**2)**.5
        if norm < 1:
            print(start_x)
        

    
