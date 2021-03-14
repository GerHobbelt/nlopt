import logging

import numpy as np

from .trstlp import Trstlp


logger = logging.getLogger(__name__)


class Cobyla:
    FINISH = 0

    # Constants
    DELTA = 1.1
    ALPHA = 0.25
    BETA = 2.1
    GAMMA = 0.5

    # Float precision
    float = np.float64
    
    
    def __init__(self, x, F, C, rhobeg=.5, rhoend=1e-6, maxfun=3500):
        n = len(x)
        m = len(C)

        self.n = n
        self.m = m
        self.x = np.array(x, dtype=self.float)
        self.F = F
        self.C = C
        self.rhoend = rhoend
        self.rho = rhobeg
        self.maxfun = maxfun
        
        # mpp (m constrains, fval, resmax)
        self.con = None # m constrains values
        self.fx = 0
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

        self.a = None # (m+1) * n

        self.vsig = None
        self.veta = None

        # flags
        self.ibrnch = False
        self.iflag = False # Acceptable simplex

        # Params
        self.parmu = 0


    @property
    def data(self): # pragma: no cover
        print(f'nfvals: {self.nfvals}')
        print(f'x: {self.x}')
        print(f'optimal_vertex: {self.optimal_vertex}')
        print(f'current_values: {self.current_values}')
        print(f'datmat: \n{self.datmat}')
        print(f'a: \n{self.a}')
        print(f'sim: \n{self.sim}')
        print(f'simi: \n{self.simi}')

        
    @property
    def current_values(self):
        return np.array((*self.con, self.fval, self.resmax), dtype=self.float)

    
    @property
    def orig_con(self):
        '''
        WARNING: self.fx is not self.fval
        
        '''
        return np.array((*self.con, self.fx, self.resmax), dtype=self.float)

    
    @property
    def parsig(self):
        return self.ALPHA * self.rho

    
    @property
    def pareta(self):
        return self.BETA * self.rho


    @property
    def fmin(self):
        return self.datmat[-1, -2]


    @property
    def res(self):
        return self.datmat[-1, -1]

        
    def run(self):
        self.set_initial_simplex()
        self.ibrnch = True

        self.L140()
        while True:
            self.L140()
            if self.L370() == self.FINISH:
                break

            
    def _calcfc(self):
        if ((self.nfvals >= self.maxfun) and (self.nfvals > 0)): # pragma: no cover
            # Error: COBYLA_USERABORT (rc = 1)
            self.L600_L620()
            logger.error('maximum number of function evaluations reach')
            raise UserWarning('cobyla: maximum number of function evaluations reach')

        self.nfvals += 1
        try:
            self.fval = self.F(self.x)
        except Exception as e: # pragma: no cover
            # Error: COBYLA_USERABORT (rc = 3)
            self.L600_L620()
            logger.error('cobyla: user requested end of minimitzation')
            raise UserWarning('cobyla: user requested end of minimitzation')

        c_iter = (constrain(self.x) for constrain in self.C)
        self.con = np.array(tuple(c_iter), dtype=self.float)
        self.resmax = max((0, *(-self.con)))
        
        
    def _set_datmat_step(self, jdrop):
        if jdrop < self.n:
            if self.fmin <= self.fval:
                self.x[jdrop] = self.optimal_vertex[jdrop]
            else:
                self.optimal_vertex[jdrop] = self.x[jdrop]
                self.datmat[jdrop] = self.datmat[-1]
                self.datmat[-1,] = self.current_values

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
        nbest = None
        phi = lambda fx, res: fx + (self.parmu * res)
        
        phimin = phi(fx=self.fmin, res=self.res)
        for j, row in zip(range(self.n), self.datmat):
            *_, fx_j, resmax_j = row
            phi_value = phi(fx_j, resmax_j)
            if phi_value < phimin:
                nbest = j
                phimin = phi_value
            elif ((phi_value == phimin) and (self.parmu == 0)):
                resmax_best = self.datmat[nbest, -1]
                cond = (resmax_j < resmax_best).any()
                nbest = j if cond else nbest
                

        # Switch the best vertex into pole position if it is not there already,
        # and also update SIM, SIMI and DATMAT
        if (nbest is not None):
            self.datmat[[nbest, -1]] = self.datmat[[-1, nbest]]
            temp = self.sim[nbest].copy()
            self.sim[nbest] = np.zeros(self.n)
            self.optimal_vertex += temp
            self.sim -= temp
            self.simi[..., nbest] = -self.simi.sum(axis=1)

        # Make an error return if SIGI is a poor approximation to the inverse of
        # the leading N by N submatrix of SIG
        sim_simi = np.dot(self.sim, self.simi)
        error = abs(sim_simi - np.eye(self.n)).max()
        error = 0 if error < 0  else error
        if error > .1: # pragma: no cover
            # Error: COBYLA_MAXFUN (rc = 2)
            self.L600_L620()
            logger.error('cobyla: rounding errors are becoming damaging')
            raise UserWarning('cobyla: rounding errors are becoming damaging')
        
        
    def _linear_coef(self):
        # Calculate the coefficients of the linear approximations to the objective
        # and constraint functions, placing minus the objective function gradient
        # after the constraint gradients in the array A. The vector W is used for
        # working space
        tcon = *_, self.fx = -self.datmat[-1, :-1]
        self.con = tcon[:-1]

        ww = (self.datmat[:-1, :-1] + tcon).T
        self.a = np.dot(ww, self.simi.T)  # (m+1) * n
        self.a[-1] *= -1

        
    def _is_acceptable_simplex(self, parsig, pareta):
        # Calculate the values of sigma and eta, and set IFLAG=False if the current
        # simplex is not acceptable
        self.vsig = 1 / (self.simi ** 2).sum(axis=0) ** .5 # col sum
        self.veta = (self.sim ** 2).sum(axis=1) ** .5 # row sum 
        return not((self.vsig < parsig).any() or (self.veta > pareta).any())

    
    def _new_vertex_improve_acceptability(self, pareta):
        jdrop, temp = -1, pareta
        for j in range(self.n):
            if self.veta[j] > temp:
                jdrop, temp = j, self.veta[j]
                
        if jdrop == -1:
            for j in range(self.n):
                if self.vsig[j] < temp:
                    jdrop, temp = j, self.vsig[j]
        
        # Calculate the step to the new vertex and its sign
        temp = self.GAMMA * self.rho * self.vsig[jdrop]
        dx = temp * self.simi[..., jdrop]

        ssum = np.dot(self.a, dx)
        temp = self.datmat[-1, :-1]

        cvmaxp = max((0, *(-ssum - temp)[:-1]))
        cvmaxm = max((0, *(ssum - temp)[:-1]))

        cond = (self.parmu * (cvmaxp - cvmaxm) > (2 * ssum[-1]))

        # Update the elements of SIM and SIMI, and set the next X
        dx = -dx if cond else dx
        self.sim[jdrop] = dx

        self.simi[..., jdrop] /= np.dot(self.simi[..., jdrop], dx)
        temp = np.dot(dx, self.simi)
        target = self.simi[..., jdrop].copy()
        self.simi -= ((np.ones(self.simi.shape) * target).T * temp)
        self.simi[..., jdrop] = target

        self.x = self.optimal_vertex + dx
        return jdrop


    def L140(self): # pragma: no cover
        parsig = self.parsig
        pareta = self.pareta
        
        self._set_optimal_vertex()
        self._linear_coef()
        self.iflag = self._is_acceptable_simplex(parsig, pareta)

        # If a new vertex is needed to improve acceptability, then decide which
        # vertex to drop from simplex
        if (self.ibrnch == True) or (self.iflag == True):
            return

        jdrop = self._new_vertex_improve_acceptability(pareta)
        
        self._calcfc()
        self.datmat[jdrop] = self.current_values
        self.ibrnch = True
        
        self._set_optimal_vertex()
        self._linear_coef()
        self.iflag = self._is_acceptable_simplex(parsig, pareta)
        
        
    def L370(self):
        # Calculate DX=x(*)-x(0). Branch if the length of DX is less than 0.5*RHO
        trstlp = Trstlp(self)
        ifull, dx = trstlp.run()
        
        if ifull == False:
            temp = sum(dx ** 2)
            cond = (temp < 0.25 * (self.rho ** 2)) 
            if cond:
                self.ibrnch = True
                return self.L550(ifull)

        # Predict the change to F and the new maximum constraint violation if the
        # variables are altered from x(0) to x(0)+DX
        self.fval = 0
        temp = (self.a * dx).sum(axis=1)
        csum, fsum = self.con - temp[:-1], self.fval - temp[-1]
        resnew = max((0, *csum))

        # Increase PARMU if necessary and branch back if this change alters the
        # optimal vertex. Otherwise PREREM and PREREC will be set to the predicted 
        # reductions in the merit function and the maximum constraint violation
        # respectively
        barmu = 0
        prerec = self.res - resnew
        if prerec > 0 :
            barmu = fsum / prerec

        if self.parmu < (barmu * 1.5):
            self.parmu = barmu * 2
            res = self.res
            phi = self.fmin + (self.parmu * res)
            phi_values = self.datmat[..., -2] + (self.parmu * self.datmat[..., -1])
            for phi_val, res_val in zip(phi_values, self.datmat[:-1, -1]):
                if (phi_val < phi):
                    return
                if (phi_val == phi) and (self.parmu == 0) and (res_val < res):
                    return

        prerem = (self.parmu * prerec) - fsum

        # Calculate the constraint and objective functions at x(*). Then find the 
        # actual reduction in the merit function
        self.x = self.optimal_vertex + dx
        self.ibrnch = True
        
        self._calcfc()
        return self.L440(ifull, dx, prerec, prerem)
    
        
    def L440(self, ifull, dx, prerec, prerem):
        vmold = self.fmin + (self.parmu * self.res)
        vmnew = self.fval + (self.parmu * self.resmax)
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
        temp = abs(np.dot(dx, self.simi))
        for j, value in zip(range(self.n), temp):
            if value > ratio:
                ratio, jdrop = value, j
                
        sigbar = temp * self.vsig
        edgmax = self.DELTA * self.rho
        mask = (sigbar >= self.parsig) | (sigbar >= self.vsig)

        lflag = None
        if mask.any():
            temp = ((dx - self.sim) ** 2).sum(axis=1) ** .5 if trured > 0 else self.veta
            temp = temp[mask]
            idx = np.arange(len(mask))[mask]
            for j, ttemp in zip(idx, temp):
                if ttemp > edgmax:
                    lflag = j
                    edgmax = ttemp

        if lflag is not None:
            jdrop = lflag
        if jdrop == -1:
            return self.L550(ifull)

        # Revise the simplex by updating the elements of SIM, SIMI and DATMAT
        self.sim[jdrop] = dx
        temp = np.dot(dx, self.simi[..., jdrop])
        self.simi[..., jdrop] /= temp
        target = self.simi[..., jdrop].copy()
        temp = np.dot(dx, self.simi)
        self.simi -= ((np.ones(self.simi.shape) * target).T * temp)
        self.simi[..., jdrop] = target
        self.datmat[jdrop] = np.array((*self.con, self.fval, self.resmax))

        # Branch back for further iterations with the current RHO
        if (trured > 0) and (trured >= prerem * 0.1):
            return

        return self.L550(ifull)

        
    def L550(self, ifull):
        if (self.iflag == False):
            self.ibrnch = False
            return
            
        # Otherwise reduce RHO if it is not at its least value and reset PARMU
        if (self.rho > self.rhoend):
            self.rho = self.rhoend if (self.rho <= (self.rhoend * 1.5)) else (self.rho / 2)
            if self.parmu > 0:
                denom = 0
                for col, ref in zip(self.datmat[:-1, :-2].T, self.datmat[-1, :-2]):
                    cmin = min((ref, col.min()))
                    cmax = max((ref, col.max()))
                    if (cmin < (cmax / 2)):
                        temp = max(cmax, 0) - cmin
                        denom = temp if denom <= 0 else min(denom, temp)
                        
                temp = self.datmat[..., -2].T
                cmin = temp.min()
                cmax = temp.max()
                if denom == 0:
                    self.parmu = 0
                elif ((cmax - cmin) < (self.parmu * denom)): 
                    self.parmu = (cmax - cmin) / denom
            return

        return self.L600_L620(ifull)

    
    def L600_L620(self, ifull=False):
        # Return the best calculated values of the variables
        if (ifull == False):
            # L600
            self.x = self.optimal_vertex
            self.fval = self.fmin
            self.resmax = self.res

        # L620
        self.maxfun = self.nfvals
        return self.FINISH
