import numpy as np


class Cobyla:
    def __init__(self, x, F, C, rhobeg=.5, rhoend=1e-6, maxfun=3500):
        n = len(x)
        m = len(C)

        self.n = n
        self.m = m
        self.x = x
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

        # simplex
        self.sim = self.rho * np.matrix(np.eye(n))
        self.optimal_vertex = self.x.copy()

        # inverse simplex
        self.simi = (1 / self.rho) * np.matrix(np.eye(n))

        # for each vertex, m constrains values, f, resmax
        # last one for the best vertex
        self.datmat = np.zeros((n + 1, m + 2))

        self.a = None # (m+1) * n

        self.vsig = None
        self.veta = None
        self.sigb = np.zeros((n,))
        self.dx = None

        # flags
        self.ibrnch = 0
        self.iflag = 0 # Simplex acceptable simplex
        self.ifull = None

        # Params
        self.alpha = 0.25
        self.beta = 2.1
        self.gamma = 0.5
        self.parmu = 0
        self.parsig = 0
        self.ratio = 0

        # Others
        self.nfvals = 0
        self.prerec = None
        self.prerem = None
        self.jdrop = None

    @property
    def current_values(self):
        return np.array((*self.con, self.fval, self.resmax), dtype=np.float)

        
    def run(self):
        breakpoint()
        self.set_initial_simplex()
        self.ibrnch = 1
        self.L140()
        self.L370()
        
        
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
        
        self.con = np.array(tuple(constrain(self.x) for constrain in self.C), dtype=np.float)
        self.resmax = max((0, *(-self.con)))
        if self.ibrnch == 1:
            return self.L440()
        
        
    def _set_datmat_step(self, jdrop):
        f = self.datmat[-1, -2]
        if jdrop < self.n:
            if f <= self.fval:
                self.x[jdrop] = self.optimal_vertex[jdrop]
            else:
                self.optimal_vertex[jdrop] = self.x[jdrop]
                self.datmat[jdrop] = self.datmat[-1]
                self.datmat[-1,] = self.current_values

                self.sim[:(jdrop + 1), jdrop] -= self.rho
                for row in range(jdrop + 1):
                    self.simi[row, jdrop] = -sum(self.simi[row, :(jdrop + 1)])


    def _calcfc_iteration(self, pos=-1):
        self._calcfc()
        self.datmat[pos,] = self.current_values
        
    
    def set_initial_simplex(self):
        self._calcfc_iteration()

        for jdrop in range(self.n):
            self.x[jdrop] += self.rho
            self._calcfc_iteration(pos=jdrop)
            self._set_datmat_step(jdrop)


    def _set_optimal_vertex(self):
        # Identify the optimal vertex of the current simplex
        nbest = -1
        phi = lambda fx, resmax: fx + (self.parmu * resmax)
        
        phimin = phi(fx=self.datmat[-1, -2], resmax=self.datmat[-1, -1])
        for j, row in zip(range(self.n), self.datmat):
            *_, fx_j, resmax_j = row
            phi_value = phi(fx_j, resmax_j)
            if phi_value < phimin:
                nbest = j
            else:
                resmax_best = self.datmat[nbest, -1]
                cond = (phi_value == phimin) and (self.parmu == 0) and (resmax_j < resmax_best)
                nbest = j if cond else nbest

        # Switch the best vertex into pole position if it is not there already,
        # and also update SIM, SIMI and DATMAT
        if (nbest != -1):
            self.datmat[[nbest, -1]] = self.datmat[[-1, nbest]]
            temp = np.array(self.sim[nbest], dtype=np.float)
            self.sim[nbest] = np.zeros(self.n)
            self.optimal_vertex += temp
            self.sim -= temp
            self.simi[nbest] = -self.simi.sum(axis=0)

        # Make an error return if SIGI is a poor approximation to the inverse of
        # the leading N by N submatrix of SIG
        sim_simi = self.sim * self.simi
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
        tcon = *self.con, self.fx = -self.datmat[-1, :-1]

        w = np.matrix(self.datmat[:-1, :-1] + tcon)
        self.a = (self.simi * w).T  # (m+1) * n
        self.a[-1] *= -1

        
    def _is_acceptable_simplex(self, pareta):
        # Calculate the values of sigma and eta, and set IFLAG=0 if the current
        # simplex is not acceptable
        self.parsig = self.alpha * self.rho
        self.vsig = (np.array((1 / (self.simi ** 2).sum(axis=0))) ** .5).ravel()
        self.veta = (np.array((self.sim ** 2).sum(axis=1)) ** .5).ravel()
        return not(np.any(self.vsig < self.parsig) or np.any(self.veta > pareta))

    
    def _new_vertex_improve_acceptability(self, pareta):
        veta_max, self.jdrop = max(zip(self.veta, range(self.n)))
        vsig_max, self.jdrop = max(zip(self.vsig, range(self.n))) if pareta >= veta_max else self.vsig[self.jdrop], self.jdrop

        # Calculate the step to the new vertex and its sign
        temp = gamma * rho * vsig_max
        self.dx = temp * self.simi[..., self.jdrop]

        ssum = np.array(np.dot(self.a, self.dx)).ravel()
        temp = self.datmat[-1, :-1]

        cvmaxp = max((0, *(-ssum[:-1] -temp)))
        cvmaxm = max((0, *(ssum[:-1] -temp)))

        cond = (self.parmu * (cvmaxp - cvmaxm) > (2 * ssum[-1]))
        dxsign = (-1 ** cond)

        # Update the elements of SIM and SIMI, and set the next X
        self.dx *= dxsign
        self.sim[self.jdrop] = self.dx
        self.simi[..., self.jdrop] /= np.dot(self.simi[..., self.jdrop], self.dx)

        pdot = np.dot(dx, simi)
        target = self.simi[..., self.jdrop]
        self.simi -= np.multiply(pdot, self.simi)
        self.simi[..., self.jdrop] = target
        
        self.x = self.sim[-1] + self.dx


    def L140(self):
        while True:
            self._set_optimal_vertex()
            self._linear_coef()

            pareta = self.beta * self.rho
            self.iflag = self._is_acceptable_simplex(pareta)

            # If a new vertex is needed to improve acceptability, then decide which
            # vertex to drop from simplex
            if self.ibrnch == 1 or self.iflag == 1:
                return

            self._new_vertex_improve_acceptability(pareta)
            self._calcfc_iteration()
            self.ibrnch = 1

            
    def L140_new(self):
        self._set_optimal_vertex()
        self._linear_coef()

        pareta = self.beta * self.rho
        self.iflag = self._is_acceptable_simplex(pareta)

        # If a new vertex is needed to improve acceptability, then decide which
        # vertex to drop from simplex
        if self.ibrnch == 1 or self.iflag == 1:
            return

        self._new_vertex_improve_acceptability(pareta)
        self._calcfc_iteration()
        self.ibrnch = 1

        self._set_optimal_vertex()
        self._linear_coef()
        self.iflag = self._is_acceptable_simplex(pareta)
        
        
    def L370(self):
        # Calculate DX=x(*)-x(0). Branch if the length of DX is less than 0.5*RHO
        trstlp = Trstlp(self)
        trstlp.run()
        
        if self.ifull == 0:
            temp = sum(self.dx ** 2)
            cond = (temp < 0.25 * (self.rho ** 2)) 
            if cond:
                self.ibrnch = 1
                return self.L550()

        # Predict the change to F and the new maximum constraint violation if the
        # variables are altered from x(0) to x(0)+DX
        self.con[-2] = 0
        ssum = self.con[:-1] - (self.a * self.dx).T.flat
        resnew = max((0, *ssum[:-1]))

        # Increase PARMU if necessary and branch back if this change alters the
        # optimal vertex. Otherwise PREREM and PREREC will be set to the predicted 
        # reductions in the merit function and the maximum constraint violation
        # respectively
        barmu = 0
        self.prerec = self.datmat[-1, -1] - resnew
        if self.prerec > 0 :
            barmu = ssum[-1] / self.prerec

        if self.parmu < (barmu * 1.5):
            self.parmu = barmu * 2
            phi = datmat[-1, -2] + (self.parmu * datmat[-1, -1])
            temp = datmat[..., -2] + (self.parmu * datmat[..., -1])
            if (temp < phi).any():
                return self.L140()
            mask = (temp == phi)
            if mask.any() and (self.parmu == 0):
                if datmat[-1][mask].flat[0] < datmat[-1, -1]:
                    return self.L140()
                
        self.prerem = (self.parmu * self.prerec) - ssum[-1]

        # Calculate the constraint and objective functions at x(*). Then find the 
        # actual reduction in the merit function
        self.x = self.self.optimal_vertex + self.dx
        self.ibrnch = 1
        self._calcfc_iteration()
        self.L140()

        
    def L440(self):
        vmold = self.datmat[-1, -2] + (self.parmu * datmat[-1, -1])
        vmnew = self.fval + (self.parmu * self.resmax)
        trured = vmold - vmnew
        if (self.parmu == 0) and (self.f == self.datmat[-1, -2]):
            self.prerem = self.prerec
            trured = self.datmat[-1, -1] - self.resmax
        
        # Begin the operations that decide whether x(*) should replace one of the
        # vertices of the current simplex, the change being mandatory if TRURED is
        # positive. Firstly, JDROP is set to the index of the vertex that is to be
        # replaced
        
        self.ratio = 1 if (trured <= 0) else 0
        self.jdrop = 0
        temp = abs(np.array(self.dx * self.simi))
        mask = (temp > ratio)
        if mask.any():
            self.jdrop = np.arange(self.n)[mask].flat[-1]
            
        sigbar = temp * self.vsig

        delta = 1.1
        edgmax = delta * self.rho
        mask = (sigbar >= self.parsig) | (sigbar >= self.vsig)

        lflag = None
        if mask.any():
            temp = (sum((self.dx - self.sim) ** 2) if trured > 0 else veta) ** .5
            temp = temp[mask]
            idx = np.arange(len(mask))[mask]
            for j, ttemp in zip(idx, temp):
                if ttemp > edgmax:
                    lflag = idx
                    edgmax = temp
                    
        if lflag is not None:
            self.jdrop = lflag
        if self.jdrop == 0:
            return self.L550()

        # Revise the simplex by updating the elements of SIM, SIMI and DATMAT
        self.sim[self.jdrop] = self.dx
        self.dx * self.simi[..., self.jdrop]
        
        # Revise the simplex by updating the elements of SIM, SIMI and DATMAT
        temp = (self.dx * self.simi[...,0]).flat[0]
        self.simi[..., self.jdrop] /= temp
        target = self.simi[..., self.jdrop]
        temp = self.dx * self.simi
        self.simi -= np.repeat(np.array(target), len(temp)).reshape(self.simi.shape) * temp
        self.simi[..., self.jdrop] = target
        self.datmat[..., self.jdrop] = np.array((*self.con, self.fval, self.resmax))

        # Branch back for further iterations with the current RHO
        if (trured > 0) and (trured >= prerem * 0.1):
            return self.L140()

        
    def L550(self):
        if (self.iflag == 0):
            self.ibrnch = 0
            return self.L140()
            
        # Otherwise reduce RHO if it is not at its least value and reset PARMU
        if (self.rho > self.rhoend):
            self.rho = self.rhoend if (self.rho <= (self.rhoend * 1.5)) else (self.rho / 2)
            if parmu > 0:
                denom = 0
                for col, ref in zip(self.datmat[:-1, :-2].T, self.datmat[-1, :-2]):
                    cmin = min((ref, *col.min()))
                    cmax = max((ref, *col.max()))
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

                
    def L600_L620(self):
        # Return the best calculated values of the variables
        if (self.ifull != 1):
            # L600
            self.x = self.sim[-1]
            self.fval = self.datmat[-1, -2]
            self.resmax = self.datmat[-1, -1]

        # L620
        self.maxfun = self.nfvals
        return
        

class Trstlp:
    def __init__(self, cobyla: Cobyla):
        self.cobyla = cobyla
        
        self.mcon = self.cobyla.m
        self.z = np.eye(self.cobyla.n) # n * n
        self.zdota = np.zeros(self.cobyla.n) # n
        
        self.icount = 0
        self.optold = 0
        self.nact = -1
        self.nactx = -1
        self.icon = self.mcon
        self.sp = None
        self.spabs = None
        self.tot = None
        self.ratio = None
        self.iout = None # To drop?
        self.kk = None
        self.stpful = None
        self.step = None
        
        self.sdirn = np.zeros(self.cobyla.n)
        self.dxnew = None # n

        self.resmax, self.icon = max(zip((0, *self.cobyla.con), (None, *range(self.cobyla.m))))
        self.iact = np.arange(self.cobyla.m + 1)
        self.vmultc = np.array((*(self.resmax - self.cobyla.con), 0), dtype=np.float)
        self.vmultd = np.zeros(self.cobyla.m + 1) 

        self.cobyla.ifull = 1
        self.cobyla.dx = np.zeros(self.cobyla.n)


    def run(self):
        # End the current stage of the calculation if 3 consecutive iterations
        # have either failed to reduce the best calculated value of the objective
        # function or to increase the number of active constraints since the best
        # value was calculated. This strategy prevents cycling, but there is a
        # remote possibility that it will cause premature termination
        self.L70()        

        
    def L70(self):
        breakpoint()
        optnew = self.resmax if (self.mcon == self.cobyla.m) else -np.dot(self.dx, self.a[-1])

        if (self.icount == 0) or (optnew < self.optold):
            self.optold = optnew
            self.nactx = self.nact
            self.icount = 3
        elif (self.nact > self.nactx):
            self.nactx = self.nact
            self.icount = 3
        else:
            self.icount -= 1
            if self.icount == 0:
                return self.L490()

        # If ICON exceeds NACT, then we add the constraint with index IACT(ICON) to
        # the active set. Apply Givens rotations so that the last N-NACT-1 columns
        # of Z are orthogonal to the gradient of the new constraint, a scalar
        # product being set to zero if its nonzero value could be due to computer
        # rounding errors. The array DXNEW is used for working space
        if (self.icon <= self.nact):
            return self.L260()

        self.kk = self.iact[self.icon]
        self.dxnew = self.a[kk]
        self.tot = 0

        for k in range(self.cobyla.n - 1, -1, -1):
            if k > self.nact:
                temp = self.z[k] * self.dxnew
                self.sp = sum(temp)
                self.spabs = sum(abs(temp))

                acca = self.spabs + (0.1 * abs(self.sp))
                accb = self.spabs + (0.2 * abs(self.sp))
                
                cond = ((self.spabs >= acca) or (acca >= accb))
                self.sp = 0 if cond else self.sp 
                self.tot = self.sp if (self.tot == 0) else self.tot
                
            else:
                temp = ((self.sp ** 2) + (self.tot ** 2)) ** 0.5
                alpha = self.sp / temp
                beta = self.tot / temp
                self.tot = temp
                self.z[k] = (alpha * self.z[k]) + (beta * self.z[k + 1])
                self.z[k + 1] = alpha * self.z[k + 1] - (beta * self.z[k])
                
        self.add_new_constrain()


    def add_new_constrain(self):
        # Add the new constraint if this can be done without a deletion from the
        # active set

        if (self.tot != 0):
            self.nact += 1
            self.zdota[self.nact] = self.tot
            self.vmultc[self.icon] = self.vmultc[self.nact]
            self.vmultc[self.nact] = 0
            return self.L210()

        return self.constant_gradient(self.nact)

    def constant_gradient(self, kmax):
        # The next instruction is reached if a deletion has to be made from the
        # active set in order to make room for the new active constraint, because
        # the new constraint gradient is a linear combination of the gradients of 
        # the old active constraints. Set the elements of VMULTD to the multipliers
        # of the linear combination. Further, set IOUT to the index of the
        # constraint to be deleted, but branch if no suitable index can be found
        self.ratio = -1
        
        for k in range(kmax, 0, -1):
            temp = self.z[k] * self.dxnew
            zdotv = sum(temp)
            zdvabs = sum(abs(temp))

            acca = zdvabs + (0.1 * (abs(zdotv)))
            accb = zdvabs + (0.2 * (abs(zdotv)))
            if ((zdvabs < acca) and (acca < accb)):
                temp = zdotv / self.zdota[k]
                if ((temp > 0) and (self.iact[k] <= self.cobyla.m)):
                    tempa = self.vmultc[self.nact] / temp
                    if (self.ratio < 0) or (tempa < self.ratio):
                        self.ratio = tempa
                        self.iout = k

                if (self.nact >= 2):
                    self.dxnew -= (temp * self.cobyla.a[self.iact[k]])

                self.vmultd[k] = temp
            else:
                self.vmultd[k] = 0

        if (self.ratio < 0):
            return self.L490()

        self.revise_lagrange_multipliers_reorder_active_cons()
    

    def revise_lagrange_multipliers_reorder_active_cons(self):
        # Revise the Lagrange multipliers and reorder the active constraints so
        # that the one to be replaced is at the end of the list. Also calculate the
        # new value of ZDOTA(NACT) and branch if it is not acceptable
        temp = self.vmultc[:(self.nact + 1)] - (self.ratio * self.vmultd[:(self.nact + 1)])
        self.vmultc[:(self.nact + 1)] = ((temp > 0 ) * temp)

        if (self.icon < self.nact):
            isave = self.iact[self.icon]
            vsave = self.vmultc[self.icon]
            for k in range(self.icon, self.nact + 1):
                kw = self.iact[k + 1]
                sp = np.dot(self.z[k], self.a[kw])
                temp = ((sp ** 2) + (self.zdota[k + 1] ** 2)) ** 0.5
                alpha = self.zdota[k + 1] / temp
                beta = sp / temp
                self.zdota[k + 1] = alpha * self.zdota[k]
                self.zdota[k] = temp

                temp = (alpha * self.z[k + 1]) + (beta * self.z[k])
                self.z[k + 1] = (alpha * self.z[k]) - (beta * self.z[k + 1])
                self.z[k] = temp
                self.iact[k] = kw
                self.vmultc[k] = self.vmultc[k + 1]

            self.iact[k] = isave
            self.vmultc[k] = vsave

        temp = np.dot(self.z[self.nact], self.a[self.kk])
        if (temp == 0):
            return self.L490()
        
        self.zdota[self.nact] = temp
        self.vmultc[self.icon] = 0
        self.vmultc[self.nact] = ratio

        self.L210()

        
    def L210(self):
        # Update IACT and ensure that the objective function continues to be
        # treated as the last active constraint when MCON>M
        
        self.iact[self.icon] = self.iact[self.nact]
        self.iact[self.nact] = self.kk
        if ((self.mcom > self.cobyla.m) and (self.kk != self.mcon)):
            k = self.nact - 1
            sp = np.dot(self.z[k], self.cobyla.a[kk])
            temp = ((sp ** 2) + (self.zdota[self.nact] ** 2)) ** 0.5
            alpha = self.zdota[self.nact] / temp
            beta = sp / temp
            self.zdota[self.nact] = alpha * self.zdota[k]
            self.zdota[k] = temp
            
            temp = (alpha * self.z[self.nact]) + (beta * self.z[k])
            self.z[self.nact] = (alpha * self.z[k]) - (beta * self.z[self.nact])
            self.z[k] = temp
            
            self.iact[self.nact] = self.iact[k]
            self.iact[k] = self.kk
            self.vmultc[k], self.vmultc[self.nact] = self.vmultc[self.nact], self.vmultc[k]

        # If stage one is in progress, then set SDIRN to the direction of the next 
        # change to the current vector of variables

        if (self.mcon > self.cobyla.m):
            return self.L320()

        self.kk = self.iact[self.nact]
        temp = (np.dot(self.sdirn, self.cobyla.a[self.kk]) - 1) / self.zdota[self.nact]
        self.sdirn -= (temp * self.z[self.nact])
        
        self.L340()


    def L260(self):
        # Delete the constraint that has the index IACT(ICON) from the active set
        if (self.icon < self.nact):
            isave = self.iact[self.icon]
            vsave = self.vmultc[self.icon]
            for k in range(self.icon, self.nact + 1):
                kp = k + 1
                self.kk = self.iact[kp]
                
                sp = np.dot(self.z[k], self.a[kk])
                temp = ((sp ** 2) + (self.zdota[kp] ** 2)) ** 0.5
                alpha = self.zdota[kp] / temp
                beta = sp / temp
                self.zdota[kp], self.zdota[k] = alpha * self.zdota[k], temp

                temp = ((alpha * self.z[kp]) + (beta * self.z[k]))
                self.z[kp] = ((alpha * self.z[k]) - (beta * self.z[kp]))
                self.z[k] = temp
                self.iact[k] = self.kk
                self.vmultc[k] = self.vmultc[kp]
                
            self.iact[self.nact] = isave
            self.vmultc[self.nact] = vsave
            
        self.nact -= 1

        # If stage one is in progress, then set SDIRN to the direction of the next
        # change to the current vector of variables
        if (self.mcon > self.cobyla.m):
            return self.L320()

        temp = np.dot(self.sdirn, self.z[self.nact + 1])
        self.sdirn -= temp * self.z[self.nact + 1]
        
        self.L340()


    def L320(self):
        self.sdirn = self.z[self.nact] / self.zdota[self.nact]
        self.L340()

        
    def L340(self):
        # Calculate the step to the boundary of the trust region or take the step
        # that reduces RESMAX to zero. The two statements below that include the
        # factor 1.0E-6 prevent some harmless underflows that occurred in a test
        # calculation. Further, we skip the step if it could be zero within a
        # reasonable tolerance for computer rounding errors
        dd = (self.cobyla.rho ** 2)
        mask = (abs(self.cobyla.dx) >= (self.cobyla.rho * 1e-6))
        dd -= sum(self.cobyla.dx[mask] ** 2)
        sd = np.dot(self.cobyla.dx, self.sdirn)
        ss = np.dot(self.sdirn, self.sdirn)

        if (dd <= 0):
            return self.L490()
        
        temp = (ss * dd) ** 0.5
        if (abs(sd) >= (temp * 1e-6)):
            temp = ((ss * dd) + (sd ** 2)) ** 0.5
            
        self.stpful = dd / (temp + sd)
        self.step = self.stpful
        if(self.mcon == self.cobyla.m):
            acca = self.step + (self.resmax * 0.1)
            accb = self.step + (self.resmax * 0.2)
            if ((self.step >= acca) or (acca >= accb)):
                return self.L480()
            
            self.step = min(self.step, self.resmax)
            
        self.set_dxnew()

        
    def set_dxnew(self):
        # Set DXNEW to the new variables if STEP is the steplength, and reduce 
        # RESMAX to the corresponding maximum residual if stage one is being done.
        # Because DXNEW will be changed during the calculation of some Lagrange
        # multipliers, it will be restored to the following value later
        
        self.dxnew = self.cobyla.dx + (self.step * self.sdirn)
        
        if (self.mcon == self.cobyla.m):
            resold, self.resmax = self.resmax, 0
            for k in range(0, self.nact + 1):
                kk = self.iact[k]
                temp = self.cobyla.con[kk] - np.dot(self.cobyla.a[kk], self.dxnew)
                self.resmax = max(self.resmax, temp)
                
        # Set VMULTD to the VMULTC vector that would occur if DX became DXNEW. A
        # device is included to force VMULTD(K)=0.0 if deviations from this value
        # can be attributed to computer rounding errors. First calculate the new
        # Lagrange multipliers.
        for k in range(self.nact, 0, -1):
            temp = (self.z[k] * self.dxnew)
            zdotw = sum(temp)
            zdwabs = sum(abs(temp))

            acca = zdwabs + (0.1 * abs(zdotw))
            accb = zdwabs + (0.2 * abs(zdotw))
            if (zdwabs >= acca) or (acca >= accb):
                zdotw = 0

            self.vmultd[k] = zdotw / self.zdota[k]
            if (k > 0):
                kk = self.iact[k]
                self.dxnew -= self.vmultd[k] * self.cobyla.a[kk]

        if (self.mcon > self.cobyla.m):
            self.vmultd[self.nact] = max(0, self.vmultd[self.nact])
                        
        # Complete VMULTC by finding the new constraint residuals
        self.dxnew = self.cobyla.dx + (self.step * self.sdirn)
        if (self.mcon > self.nact):
            for k in range(self.nact, self.mcon):
                kk = self.iact[k]
                temp = np.dot(self.cobyla.a[kk], self.dxnew)
                ssum = self.resmax - self.cobyla.con[kk] + temp
                ssumabs = self.resmax + abs(self.cobyla.con[kk]) + abs(temp)

                acca = ssumabs + (0.1 * abs(ssum))
                accb = ssumabs + (0.2 * abs(ssum))
                cond = ((ssumabs >= acca) or (acca >= accb))
                self.vmultd[k] = 0 if cond else ssum

        # Calculate the fraction of the step from DX to DXNEW that will be taken
        self.ratio = 1
        self.icon = None
        for k in range(self.mcon):
            if self.vmultd[k] < 0:
                temp = self.vmultc[k] / (self.vmultc[k] - vmultd)
                if (temp < self.ratio):
                    self.ratio = temp
                    self.icon = k

        # Update DX, VMULTC and RESMAX
        temp = 1 - self.ratio
        self.cobyla.dx = (temp * self.cobyla.dx) + (ratio * self.dxnew)
        self.vmultc = (temp * self.vmultc) + (ratio * self.vmultd)
        self.vmultc *= (self.vmultc > 0)

        if (self.mcon == self.cobyla.m):
            self.resmax = resold + (self.ratio * (self.resmax - resold))
        
        # If the full step is not acceptable then begin another iteration.
        # Otherwise switch to stage two or end the calculation
        if (self.icon is not None):
            return self.L70()

        if (self.step == self.stpful):
            return 0

        self.L480()

        
    def L480(self):
        self.mcon = self.cobyla.m + 1
        self.icon = self.iact[-1] = self.cobyla.m
        self.vmultc[-1] = 0
        return self.L60()

    
    def L490(self):
        if (self.mcon == self.cobyla.m):
            return self.L480()
        
        self.cobyla.ifull = 0
        return 0
                
            
        
        
                
            
        
        

        
            
            
