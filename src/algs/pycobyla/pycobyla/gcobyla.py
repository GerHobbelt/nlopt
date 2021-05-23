from enum import auto

import numpy as np

from .cobyla import Cobyla


class GCobyla(Cobyla):
    AFTER_LOAD_INITIAL_SIMPLEX_CHECKPOINT = auto()
    
    BEFORE_REVIEW_CURRENT_SIMPLEX_CHECKPOINT = auto()
    AFTER_REVIEW_CURRENT_SIMPLEX_CHECKPOINT = auto()

    BEFORE_GENERATE_X_START_CHECKPOINT = auto()
    AFTER_GENERATE_X_START_CHECKPOINT = auto()

    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stage = self.NEW_ITERATION
        self.track = np.array((self.x,))

        
    @property
    def simplex(self):
        return self.optimal_vertex + np.array((np.zeros(self.n), *(self.sim)))

    
    def _add_track(self):
        if (self.optimal_vertex == self.track[-1]).all() == False:
            self.track = np.vstack((self.track, self.optimal_vertex))

            
    def g_L140_review_current_simplex(self):
        yield self.BEFORE_REVIEW_CURRENT_SIMPLEX_CHECKPOINT
        super().L140_review_current_simplex()
        self._add_track()
        yield self.AFTER_REVIEW_CURRENT_SIMPLEX_CHECKPOINT

        
    def g_L370_generate_x_start(self):
        yield self.BEFORE_GENERATE_X_START_CHECKPOINT
        self._stage = super().L370_generate_x_start()
        self._add_track()
        yield self.AFTER_GENERATE_X_START_CHECKPOINT

        
    def g_run(self):
        self.set_initial_simplex()
        self._add_track()
        yield self.AFTER_LOAD_INITIAL_SIMPLEX_CHECKPOINT
        
        self.ibrnch = True

        while self._stage == self.NEW_ITERATION:
            yield from self.g_L140_review_current_simplex()
            yield from self.g_L370_generate_x_start()

