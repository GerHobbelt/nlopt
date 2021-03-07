import numpy as np

from .trstlp import Trstlp


class Cobyla:
    # Stages 
    LL140 = 140
    LL370 = 370
    LL440 = 440
    FINISH = 0

    # Constants
    DELTA = 1.1
    ALPHA = 0.25
    BETA = 2.1
    GAMMA = 0.5

    # Float precision
    float = np.float
    
    
    def __init__(self, x, F, C, rhobeg=.5, rhoend=1e-6, maxfun=3500):
        n = len(x)
        m = len(C)

        self.n = n
        self.m = m
        self.x = np.array(x, dtype=self.float)
        self.F = F
        self.C = C
        self.rhobeg = rhobeg
        self.rhoend = rhoend
        self.rho = self.rhobeg
        self.maxfun = maxfun
        
        # mpp (m constrains, fval, resmax)
        self.con = None # m constrains values
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
        self.dx = None

        # flags
        self.ibrnch = False
        self.iflag = False # Acceptable simplex
        self.ifull = None

        # Params
        self.parmu = 0

        # Others
        self.prerec = None
        self.prerem = None


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
    def parsig(self):
        return self.ALPHA * self.rho

    
    @property
    def pareta(self):
        return self.BETA * self.rho

        
    def run(self):
        self.set_initial_simplex()
        self.ibrnch = True

        # LL370, LL440
        stage = self.L140()
        while stage != self.FINISH:
            if stage == self.LL140:
                # LL370, LL440
                mth = self.L140
            elif stage == self.LL370:
                # LL140, LL440, FINISH
                mth = self.L370
            else:
                # LL140, FINISH
                mth = self.L440

            stage = mth()    
        
        
    def _calcfc(self):
        if ((self.nfvals >= self.maxfun) and (self.nfvals > 0)):
            # Error: COBYLA_USERABORT (rc = 1)
            self.L600_L620()
            raise UserWarning('cobyla: maximum number of function evaluations reach')

        self.nfvals += 1
        try:
            self.fval = self.F(self.x)
        except Exception as e:
            # Error: COBYLA_USERABORT (rc = 3)
            self.L600_L620()
            raise UserWarning('cobyla: user requested end of minimitzation')

        c_iter = (constrain(self.x) for constrain in self.C)
        self.con = np.array(tuple(c_iter), dtype=self.float)
        self.resmax = max((0, *(-self.con)))
        
        
    def _set_datmat_step(self, jdrop):
        if jdrop < self.n:
            if self.datmat[-1, -2] <= self.fval:
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

            
    def _calcfc_iteration(self, pos=-1):
        self._calcfc()
        if self.ibrnch == True:
            return self.LL440
        self.datmat[pos] = self.current_values


    def _set_optimal_vertex(self):
        # Identify the optimal vertex of the current simplex
        nbest = None
        phi = lambda fx, resmax: fx + (self.parmu * resmax)
        
        phimin = phi(fx=self.datmat[-1, -2], resmax=self.datmat[-1, -1])
        for j, row in zip(range(self.n), self.datmat):
            *_, fx_j, resmax_j = row
            phi_value = phi(fx_j, resmax_j)
            if phi_value < phimin:
                nbest = j
                phimin = phi_value
            else:
                resmax_best = self.datmat[nbest, -1]
                cond = (phi_value == phimin) and (self.parmu == 0) and (resmax_j < resmax_best).any()
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
        if error > .1:
            # Error: COBYLA_MAXFUN (rc = 2)
            self.L600_L620()
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
        veta_max, jdrop = max(zip(self.veta, range(self.n)))
        vsig_max, jdrop = max(zip(self.vsig, range(self.n))) \
            if (pareta >= veta_max) else (self.vsig[jdrop], jdrop)

        # Calculate the step to the new vertex and its sign
        temp = self.GAMMA * self.rho * vsig_max
        self.dx = temp * self.simi[..., jdrop]

        ssum = np.dot(self.a, self.dx)
        temp = self.datmat[-1, :-1]

        cvmaxp = max((0, *(-ssum - temp)[:-1]))
        cvmaxm = max((0, *(ssum - temp)[:-1]))

        cond = (self.parmu * (cvmaxp - cvmaxm) > (2 * ssum[-1]))

        # Update the elements of SIM and SIMI, and set the next X
        self.dx *= -1 if cond else 1
        self.sim[jdrop] = self.dx

        self.simi[..., jdrop] /= np.dot(self.simi[..., jdrop], self.dx)
        temp = np.dot(self.dx, self.simi)
        target = self.simi[..., jdrop].copy()
        self.simi -= ((np.ones(self.simi.shape) * target).T * temp)
        self.simi[..., jdrop] = target

        self.x = self.optimal_vertex + self.dx
        return jdrop


    def L140(self):
        parsig = self.parsig
        pareta = self.pareta
        
        while True:
            self._set_optimal_vertex()
            self._linear_coef()
            self.iflag = self._is_acceptable_simplex(parsig, pareta)

            # If a new vertex is needed to improve acceptability, then decide which
            # vertex to drop from simplex
            if (self.ibrnch == True) or (self.iflag == True):
                break

            jdrop = self._new_vertex_improve_acceptability(pareta)
            if self._calcfc_iteration(pos=jdrop) == self.LL440:
                return self.LL440
            self.ibrnch = True

        return self.LL370

            
    def L140_simplex_update(self): # pragma: no cover
        parsig = self.parsig
        pareta = self.pareta
        
        self._set_optimal_vertex()
        self._linear_coef()
        self.iflag = self._is_acceptable_simplex(parsig, pareta)

        # If a new vertex is needed to improve acceptability, then decide which
        # vertex to drop from simplex
        if (self.ibrnch == True) or (self.iflag == True):
            return self.LL370

        self._new_vertex_improve_acceptability(pareta)
        if self._calcfc_iteration() == self.LL440:
            return self.LL440
        
        self.ibrnch = True
        self._set_optimal_vertex()
        self._linear_coef()
        self.iflag = self._is_acceptable_simplex(parsig, pareta)
        return self.LL370
        
        
    def L370(self):
        # Calculate DX=x(*)-x(0). Branch if the length of DX is less than 0.5*RHO
        trstlp = Trstlp(self)
        self.ifull, self.dx = trstlp.run()
        
        if self.ifull == False:
            temp = sum(self.dx ** 2)
            cond = (temp < 0.25 * (self.rho ** 2)) 
            if cond:
                self.ibrnch = True
                return self.L550()

        # Predict the change to F and the new maximum constraint violation if the
        # variables are altered from x(0) to x(0)+DX
        self.fval = 0
        temp = (self.a * self.dx).sum(axis=1)
        csum, fsum = self.con - temp[:-1], self.fval - temp[-1]
        resnew = max((0, *csum))

        # Increase PARMU if necessary and branch back if this change alters the
        # optimal vertex. Otherwise PREREM and PREREC will be set to the predicted 
        # reductions in the merit function and the maximum constraint violation
        # respectively
        barmu = 0
        self.prerec = self.datmat[-1, -1] - resnew
        if self.prerec > 0 :
            barmu = fsum / self.prerec

        if self.parmu < (barmu * 1.5):
            self.parmu = barmu * 2
            res = self.datmat[-1, -1]
            phi = self.datmat[-1, -2] + (self.parmu * self.datmat[-1, -1])
            phi_values = self.datmat[..., -2] + (self.parmu * self.datmat[..., -1])
            for phi_val, res_val in zip(phi_values, self.datmat[:-1, -1]):
                if (phi_val < phi):
                    return self.LL140
                if (phi_val == phi) and (self.parmu == 0) and (res_val < res):
                    return self.LL140

        self.prerem = (self.parmu * self.prerec) - fsum

        # Calculate the constraint and objective functions at x(*). Then find the 
        # actual reduction in the merit function
        self.x = self.optimal_vertex + self.dx
        self.ibrnch = True
        if self._calcfc_iteration() == self.LL440:
            return self.LL440
        
        return self.LL140

        
    def L440(self):
        vmold = self.datmat[-1, -2] + (self.parmu * self.datmat[-1, -1])
        vmnew = self.fval + (self.parmu * self.resmax)
        trured = vmold - vmnew
        if (self.parmu == 0) and (self.fval == self.datmat[-1, -2]):
            self.prerem = self.prerec
            trured = self.datmat[-1, -1] - self.resmax
        
        # Begin the operations that decide whether x(*) should replace one of the
        # vertices of the current simplex, the change being mandatory if TRURED is
        # positive. Firstly, JDROP is set to the index of the vertex that is to be
        # replaced
        jdrop = -1
        ratio = 1 if (trured <= 0) else 0
        temp = abs(np.dot(self.dx, self.simi))
        for j, value in zip(range(self.n), temp):
            if value > ratio:
                ratio, jdrop = value, j
                
        sigbar = temp * self.vsig
        edgmax = self.DELTA * self.rho
        mask = (sigbar >= self.parsig) | (sigbar >= self.vsig)

        lflag = None
        if mask.any():
            temp = ((self.dx - self.sim) ** 2).sum(axis=1) ** .5 if trured > 0 else self.veta
            temp = temp[mask]
            idx = np.arange(len(mask))[mask]
            for j, ttemp in zip(idx, temp):
                if ttemp > edgmax:
                    lflag = j
                    edgmax = ttemp

        if lflag is not None:
            jdrop = lflag
        if jdrop == -1:
            return self.L550()

        # Revise the simplex by updating the elements of SIM, SIMI and DATMAT
        self.sim[jdrop] = self.dx
        temp = np.dot(self.dx, self.simi[..., jdrop])
        self.simi[..., jdrop] /= temp
        target = self.simi[..., jdrop].copy()
        temp = np.dot(self.dx, self.simi)
        self.simi -= ((np.ones(self.simi.shape) * target).T * temp)
        self.simi[..., jdrop] = target
        self.datmat[jdrop] = np.array((*self.con, self.fval, self.resmax))

        # Branch back for further iterations with the current RHO
        if (trured > 0) and (trured >= self.prerem * 0.1):
            return self.LL140

        return self.L550()

        
    def L550(self):
        if (self.iflag == False):
            self.ibrnch = False
            return self.LL140
            
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
            return self.LL140

        return self.L600_L620()

    
    def L600_L620(self):
        # Return the best calculated values of the variables
        if (self.ifull == False):
            # L600
            self.x = self.optimal_vertex
            self.fval = self.datmat[-1, -2]
            self.resmax = self.datmat[-1, -1]

        # L620
        self.maxfun = self.nfvals
        return self.FINISH
