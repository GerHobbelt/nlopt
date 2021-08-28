'''
Pure Python version of COBYLA algorithm.
Ported from COBYLA C version.
Reviewed from original paper.

'''

import logging
from enum import auto

import numpy as np

from .trstlp import Trstlp


logger = logging.getLogger(__name__)


class Cobyla:
    FINISH = auto()
    NEW_ITERATION = auto()

    # Constants (numerical test that did not provide clear answer)
    ALPHA = 0.25
    BETA = 2.1
    DELTA = 1.1  # Set by software, 1 < DELTA <= BETA
    GAMMA = 0.5

    # Experimental constants
    BARMU_EVAL_FACTOR = 1.5  # Revise mu box. Pag. 56 [Co1]
    BARMU_SET_FACTOR = 2  # [Co1]
    RHO_ACCEPTABILITY_1 = 0.5  # Pag. 56, [Co2]
    RHO_ACCEPTABILITY_2 = 0.1  # Pag. 56, [Co3]
    RHO_REDUX_FACTOR = 0.5  # Pag. 56, [Co4]
    RHO_CONDITION_SCALE = 3  # Pag. 56, [Co5]

    # Float precision
    float = np.float128
    
    
    def __init__(self, start_x, F, C, rhobeg=.5, rhoend=1e-6, maxfun=3500):
        n = len(start_x)
        m = len(C)

        self.n = n
        self.m = m
        self.start_x = start_x
        self.x = np.array(start_x, dtype=self.float)
        self.F = F
        self.C = C
        self.rhoend = rhoend
        self.rho = rhobeg
        self.maxfun = maxfun
        
        # mpp (m constrains, fval, resmax)
        self.con = None  # m constrains values
        self.neg_cmin = None
        self.fval = 0
        self.resmax = 0

        # control
        self.nfvals = 0

        # simplex
        self.sim = self.rho * np.eye(n, dtype=self.float)
        self.optimal_vertex = self.x.copy()

        # inverse simplex
        self.simi = (1 / self.rho) * np.eye(n, dtype=self.float)

        # for each vertex, m constrains, f, resmax values
        # last one for the best vertex
        self.datmat = np.zeros((n + 1, m + 2), dtype=self.float)

        self.a = None  # (m+1) * n

        self.vsig = None
        self.veta = None

        # flags
        self.ibrnch = False
        self.iflag = False  # Acceptable simplex

        # Params
        self.parmu = 0

        
    @property
    def data(self):  # pragma: no cover
        print(f'nfvals: {self.nfvals}')
        print(f'x: {self.x}')
        print(f'optimal_vertex: {self.optimal_vertex}')
        print(f'datmat: \n{self.datmat}')
        print(f'a: \n{self.a}')
        print(f'sim: \n{self.sim}')
        print(f'simi: \n{self.simi}')
        print(f'optimal vertex: \n{self.optimal_vertex}')
        print(f'parmu: {self.parmu}')
        print(f'rho: {self.rho}')

        
    @property
    def current_values(self):
        return np.array((*self.con, self.fval, self.resmax), dtype=self.float)
    
    
    @property
    def parsig(self):
        return self.ALPHA * self.rho

    
    @property
    def pareta(self):
        return self.BETA * self.rho

    
    @property
    def neg_cfmin(self):
        return -self.datmat[-1, :-1]

    
    @property
    def fmin(self):
        return self.datmat[-1, -2]


    @property
    def res(self):
        return self.datmat[-1, -1]

        
    def run(self):
        self.set_initial_simplex()
        self.ibrnch = True

        while True:
            self.L140_review_current_simplex()
            if self.L370_generate_x_start() == self.FINISH:
                break

    def phi(self, fx, res):
        return fx + (self.parmu * res)
    
            
    def _calcfc(self):
        if ((self.nfvals >= self.maxfun) and (self.nfvals > 0)):  # pragma: no cover
            # Error: COBYLA_USERABORT (rc = 1)
            self.L600_L620_terminate()
            logger.error('maximum number of function evaluations reach')
            raise UserWarning('cobyla: maximum number of function evaluations reach')

        self.nfvals += 1
        try:
            self.fval = self.F(self.x)
        except Exception:  # pragma: no cover
            # Error: COBYLA_USERABORT (rc = 3)
            self.L600_L620_terminate()
            logger.error('cobyla: user requested end of minimitzation')
            raise UserWarning('cobyla: user requested end of minimitzation')

        c_iter = (constrain(self.x) for constrain in self.C)
        self.con = np.array(tuple(c_iter), dtype=self.float).ravel()
        self.resmax = max((0, *(-self.con)))
        
        
    def _set_datmat_step(self, jdrop):
        if self.fmin <= self.fval:
            self.x[jdrop] = self.optimal_vertex[jdrop]  # restores the previous values
        else:
            self.optimal_vertex[jdrop] = self.x[jdrop]
            self.datmat[jdrop] = self.datmat[-1]
            self.datmat[-1] = self.current_values

            self.sim[:(jdrop + 1), jdrop] = -self.rho
            for row in range(jdrop + 1):
                self.simi[row, jdrop] = -sum(self.simi[row, :(jdrop + 1)])
        
    
    def set_initial_simplex(self):
        self._calcfc()
        self.datmat[-1] = self.current_values

        for jdrop in range(self.n):
            self.x[jdrop] += self.rho
            self._calcfc()
            self.datmat[jdrop] = self.current_values
            self._set_datmat_step(jdrop)


    def _set_optimal_vertex(self):
        # Identify the optimal vertex of the current simplex
        nbest = -1

        phimin = self.phi(fx=self.fmin, res=self.res)
        for j, row in zip(range(self.n), self.datmat):
            *_, fx_j, resmax_j = row
            phi_value = self.phi(fx_j, resmax_j)
            if phi_value < phimin:
                nbest = j
                phimin = phi_value
            elif ((phi_value == phimin) and (self.parmu == 0)):
                resmax_best = self.datmat[nbest, -1]
                nbest = j if (resmax_j < resmax_best) else nbest
                
        # Switch the best vertex into pole position if it is not there already,
        # and also update SIM, SIMI and DATMAT
        if (nbest > -1):
            self.datmat[[nbest, -1]] = self.datmat[[-1, nbest]]
            temp = self.sim[nbest].copy()
            self.sim[nbest] = np.zeros(self.n)
            self.optimal_vertex += temp
            self.sim -= temp
            self.simi[..., nbest] = -self.simi.sum(axis=1)  # Add row elements

        # Make an error return if SIMI is a poor approximation to the inverse of
        # the leading N by N submatrix of SIM
        sim_simi = self.sim @ self.simi
        error = abs(sim_simi - np.eye(self.n)).max()
        error = 0 if error < 0 else error
        if error > .1:  # pragma: no cover
            # Error: COBYLA_MAXFUN (rc = 2)
            self.L600_L620_terminate()
            logger.error('cobyla: rounding errors are becoming damaging')
            raise UserWarning('cobyla: rounding errors are becoming damaging')
        
        
    def _linear_coef(self):
        # Calculate the coefficients of the linear approximations to the objective
        # and constraint functions, placing minus the objective function gradient
        # after the constraint gradients in the array A. The vector W is used for
        # working space
        neg_cfmin = self.neg_cfmin  # -self.datmat[-1, :-1]
        self.neg_cmin = neg_cfmin[:-1]  # JSX, 2021: Original reuses self.con 

        diff = (self.datmat[:-1, :-1] + neg_cfmin)  # Matrix diff: (constrains,fx) vs best
        self.a = (self.simi @ diff).T
        self.a[-1] *= -1

        
    def _is_acceptable_simplex(self, parsig, pareta):
        # Calculate the values of sigma and eta, and set IFLAG=False if the current
        # simplex is not acceptable
        self.vsig = (1 / ((self.simi ** 2).sum(axis=0))) ** .5
        self.veta = (self.sim ** 2).sum(axis=1) ** .5
        return not((self.vsig < parsig).any() or (self.veta > pareta).any())

    
    def _new_vertex_improve_acceptability(self, pareta):
        mth = lambda x: x[1]
        jdrop, max_veta = max(zip(range(self.n), self.veta), key=mth)
        if max_veta < pareta:
            jdrop, min_sig = min(zip(range(self.n), self.vsig), key=mth)
        
        # Calculate the step to the new vertex and its sign
        temp = self.GAMMA * self.rho * self.vsig[jdrop]
        dx = temp * self.simi[..., jdrop]
        kdx = self.vsig[jdrop] / (self.GAMMA * self.rho)

        ssum = self.a @ dx
        temp = self.datmat[-1, :-1]

        cvmaxp = max((0, *(-ssum - temp)[:-1]))
        cvmaxm = max((0, *(ssum - temp)[:-1]))

        cond = ((self.parmu * (cvmaxp - cvmaxm)) > (2 * ssum[-1]))
        
        # Update the elements of SIM and SIMI, and set the next X
        dx, kdx = (-dx, -kdx) if cond else (dx, kdx)
        self.sim[jdrop] = dx

        self.simi[..., jdrop] *= kdx  #  JSX, 2021: Original: self.simi[..., jdrop] /= (self.simi[..., jdrop] @ dx)
        temp = dx @ self.simi
        target = self.simi[..., jdrop].copy()
        self.simi -= np.broadcast_to(target, self.simi.shape).T * temp
        self.simi[..., jdrop] = target

        self.x = self.optimal_vertex + dx

        self._calcfc()
        self.datmat[jdrop] = self.current_values
        self.ibrnch = True

        
    def L140_review_current_simplex(self):
        '''
        L140:
        
        Ensures that x(0) is the optimal vertex.
        Set self.iflag = True iff the simplex is acceptable.        
        '''
        parsig = self.parsig
        pareta = self.pareta
        
        self._set_optimal_vertex()
        self._linear_coef()
        self.iflag = self._is_acceptable_simplex(parsig, pareta)

        # If a new vertex is needed to improve acceptability, then decide which
        # vertex to drop from simplex
        if (self.ibrnch is True) or (self.iflag is True):
            return

        self._new_vertex_improve_acceptability(pareta)
        
        self._set_optimal_vertex()
        self._linear_coef()
        self.iflag = self._is_acceptable_simplex(parsig, pareta)
        
        
    def L370_generate_x_start(self):
        '''
        L370:

        x(*) generation. mu param could be updated.
        '''
        # Calculate DX=x(*)-x(0). Branch if the length of DX is less than 0.5*RHO
        trstlp = Trstlp(self)
        ifull, dx = trstlp.run()

        if ifull is False:
            temp = sum(dx ** 2)
            cond = (temp < ((self.RHO_ACCEPTABILITY_1 * self.rho) ** 2))
            if cond:
                self.ibrnch = True
                return self.L550_update_params(ifull)

        # Predict the change to F and the new maximum constraint violation if the
        # variables are altered from x(0) to x(0)+DX
        temp = self.a @ dx
        cdiff, ftemp = self.neg_cmin - temp[:-1], -temp[-1]
        resnew = max((0, *cdiff))

        # Increase PARMU if necessary and branch back if this change alters the
        # optimal vertex. Otherwise PREREM and PREREC will be set to the predicted
        # reductions in the merit function and the maximum constraint violation
        # respectively
        prerec = self.res - resnew  # PREdicted REduction maximum Constraint violation
        barmu = (ftemp / prerec) if (prerec > 0) else 0

        if self.parmu < (self.BARMU_EVAL_FACTOR * barmu):
            self.parmu = self.BARMU_SET_FACTOR * barmu
            res = self.res
            phi_min = self.phi(self.fmin, res)
            phi_values = self.phi(fx=self.datmat[..., -2], res=self.datmat[..., -1])
            for phi_val, res_val in zip(phi_values, self.datmat[:-1, -1]):
                if (phi_val < phi_min):
                    return self.NEW_ITERATION
                if (phi_val == phi_min) and (self.parmu == 0) and (res_val < res):
                    return self.NEW_ITERATION

        prerem = (self.parmu * prerec) - ftemp  # PREdicted REduction Merit function

        # Calculate the constraint and objective functions at x(*). Then find the
        # actual reduction in the merit function
        self.x = self.optimal_vertex + dx
        self.ibrnch = True
        
        self._calcfc()
        return self.L440_update_simplex(ifull, dx, prerec, prerem)
    
        
    def L440_update_simplex(self, ifull, dx, prerec, prerem):
        '''
        L440:

        '''
        vmold = self.phi(fx=self.fmin, res=self.res)
        vmnew = self.phi(fx=self.fval, res=self.resmax) 
        trured = vmold - vmnew
        if (self.parmu == 0) and (self.fval == self.fmin):
            prerem = prerec
            trured = self.res - self.resmax
        
        # Begin the operations that decide whether x(*) should replace one of the
        # vertices of the current simplex, the change being mandatory if TRURED is
        # positive. Firstly, JDROP is set to the index of the vertex that is to be
        # replaced

        jdrop = -1
        ratio = 1 if (trured <= 0) else 0
        ratios = abs(dx @ self.simi)
        for j, new_ratio in zip(range(self.n), ratios):
            if new_ratio > ratio:
                jdrop, ratio = j, new_ratio
                
        sigbar = ratios * self.vsig
        edgmax = self.DELTA * self.rho
        mask = (sigbar >= self.parsig) | (sigbar >= self.vsig)

        if mask.any():
            temp = ((dx - self.sim) ** 2).sum(axis=1) ** .5 if trured > 0 else self.veta
            temp = temp[mask]
            idx = np.arange(len(mask))[mask]
            for j, ttemp in zip(idx, temp):
                if ttemp > edgmax: 
                    jdrop, edgmax = j, ttemp

        if jdrop == -1:
            return self.L550_update_params(ifull)

        self._update_simplex_with_x_start(dx, jdrop)

        # Branch back for further iterations with the current RHO
        if (trured > 0) and (trured >= (self.RHO_ACCEPTABILITY_2 * prerem)):
            return self.NEW_ITERATION

        return self.L550_update_params(ifull)


    def _update_simplex_with_x_start(self, dx, jdrop):
        # Revise the simplex by updating the elements of SIM, SIMI and DATMAT
        self.sim[jdrop] = dx
        self.simi[..., jdrop] /= (dx @ self.simi[..., jdrop])
        target = self.simi[..., jdrop].copy()
        temp = dx @ self.simi
        self.simi -= (np.broadcast_to(target, self.simi.shape).T * temp)
        self.simi[..., jdrop] = target
        self.datmat[jdrop] = self.current_values
    
        
    def L550_update_params(self, ifull):
        '''
        L550:

        Updates mu and rho params.
        '''
        if (self.iflag is False):
            self.ibrnch = False
            return self.NEW_ITERATION
            
        # Otherwise reduce RHO if it is not at its least value and reset PARMU
        if (self.rho > self.rhoend):
            cond = (self.rho <= (self.RHO_CONDITION_SCALE * self.rhoend))
            self.rho = self.rhoend if cond else (self.RHO_REDUX_FACTOR * self.rho)
            if self.parmu > 0:
                denom = 0
                ccmin = self.datmat[..., :-2].min(axis=0)
                ccmax = self.datmat[..., :-2].max(axis=0)
                for cmin, cmax in zip(ccmin, ccmax):
                    if (cmin < (cmax / 2)):
                        temp = max(cmax, 0) - cmin
                        denom = temp if denom <= 0 else min(denom, temp)

                vfx = self.datmat[..., -2]
                fmin = vfx.min()
                fmax = vfx.max()
                if denom == 0:
                    self.parmu = 0
                elif ((fmax - fmin) < (self.parmu * denom)):
                    self.parmu = (fmax - fmin) / denom
            return self.NEW_ITERATION

        return self.L600_L620_terminate(ifull)

    
    def L600_L620_terminate(self, ifull=False):
        '''
        L600, L620:

        '''
        # Return the best calculated values of the variables
        if (ifull is False):
            # L600
            self.x = self.optimal_vertex
            self.fval = self.fmin
            self.resmax = self.res

        # L620
        self.maxfun = self.nfvals
        return self.FINISH