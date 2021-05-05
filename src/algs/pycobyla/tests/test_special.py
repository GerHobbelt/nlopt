import numpy as np

from tests.test_custom import pyramid
from tests.test_originals import cobyla_tester


def test_pyramid_bad_optimization_due_to_data_precision():
    '''
    Test pyramid seems not work well in certain cases when data is not well conditioned

    See: L440_update_simplex, JSX tag
    '''
    F = lambda x: -pyramid(x, center=np.zeros(2), width=2, height=1)

    C = ()
    x = np.array((0.25845740809076367, 0.80861280676575664))
    known_x = np.zeros(2)

    opt = cobyla_tester(F, C, x, known_x)
