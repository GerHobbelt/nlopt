import numpy as np

class Cobyla:
    def __init__(self, x, F, C, rhobeg=0, rhoend=1, maxfun=1000):
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
        self.state = 0
        
        # mpp (m constrains, fval, resmax)
        self.con = None # m constrains values
        self.fval = 0 
        self.resmax = 0

        # simplex
        self.sim = self.rho * np.matrix(np.eye((n, n)))
        self.optimal_vertex = self.x.copy()

        # inverse simplex
        self.simi = (1 / self.rho) * np.matrix(np.eye((n, n)))

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
        self.iflag = 0
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

    @property
    def current_values(self):
        np.array((*self.con, self.fval, self.resmax))

        
    def calcfc(self):
        self.nfvals += 1
        try:
            self.fval = self.F(self.x)
        except Exception as e:
            # Error: COBYLA_USERABORT (rc = 3)
            # GOTO L600
            raise UserWarning('cobyla: user requested end of minimitzation')
        
        self.con = np.array(tuple(constrain(self.x) in self.C))
        self.resmax = max((0, *(-self.con)))
        if self.ibranch == 1:
            # GOTO L440
            pass

        if (self.iflag == 0):
            self.ibrnch = 0
            # GOTO L140
        else:
            pass
        

        
    def _set_datmat_step(self, jdrop):
        f = datmat[-1, -2]
        if f <= self.fval:
            self.x[jdrop] = self.optimal_vertex[jdrop]
        else:
            self.optimal_vertex[jdrop] = self.x[jdrop]
            self.datmat[jdrop] = self.datmat[-1]
            self.datmat[-1,] = self.current_values

            self.sim[:(jdrop + 1), jdrop] -= self.rho
            for row in range(jdrop + 1):
                self.simi[row, jdrop] = -sum(self.simi[row, :(jdrop + 1)])

                
    def set_datmat(self):
        self.calcfc()
        self.datmat[-1,] = self.current_values
        self._set_data,at_step(-1)
        
        for jdrop in range(self.n):
            self.calcfc()
            self._set_data,at_step(jdrop)
            self.x[jdrop] += self.rho
            
        
    def cobylb(self):
        if ((self.nfvals > 0) and (self.nfvals >= self.maxfun)):
            # Error: COBYLA_MAXFUN (rc = 1)
            # GOTO L600
            raise UserWarning('cobyla: maximum number of function evaluation')

        self.set_datmat()

        self.ibrnch = 1
        self.promote_best()
        self.linear_coef()

        self.acceptable_simplex()
        
        
    def promote_best(self):
        # L130, L140
        nbest = self.n + 1
        phi = lambda fx, resmax: fx + (self.parmu * resmax)
        
        phimin = phi(fx=self.datmat[-1, -2], resmax=self.datmat[-1, -1])
        for j, row in zip(range(self.n + 1), self.datmat):
            *_, fx_j, resmax_j = row[j]
            temp = phi(fx_j, resmax_j)
            if temp < phimin:
                nbest = j
            else:
                resmax_best = self.datmat[nbest, -1]
                cond = (temp == phimin) and (self.parmu == 0) and (resmax_j < resmax_best)
                nbest = j if cond else nbest

        if (nbest <= self.n):
            self.datmat[[nbest, -1]] = self.datmat[[-1, nbest]]
            temp = np.array(self.sim[nbest])
            self.sim[nbest] = np.zeros(len(temp))
            self.optimal_vertex += temp
            self.sim -= temp
            self.simi[nbest] = -self.simi.sum(axis=0)

        error = (self.sim * self.simi).max()
        if error > .1:
            # Error: COBYLA_MAXFUN (rc = 2)
            raise UserWarning('cobyla: rounding errors are becoming damaging')

        
    def linear_coef(self):
        tcon = *con, fx = -self.datmat[-1, :-1]
        self.con = con
        self.fval = fx

        w = np.matrix(self.datmat[:-1] + tcon)
        self.a = (self.simi * w).T  # (m+1) * n
        self.a[-1] *= -1

        
    def acceptable_simplex(self):
        self.parsig = self.alpha * rho
        pareta = self.beta * rho
        self.vsig = 1 / (sum(np.array(self.simi)**2, axis=0))**.5
        self.veta = (sum(np.array(self.sim)**2, axis=1))**.5
        self.iflag = not(np.any(vsig < parsig) or np.any(veta > pareta))

        # If a new vertex is needed to improve acceptability, then decide which
        # vertex to drop from simplex
        if self.ibrnch == 1 or self.iflag == 1:
            return

        veta_max, jdrop = max(zip(self.veta, range(self.n)))
        vsig_max, jdrop = max(zip(self.vsig, range(self.n))) if pareta >= veta_max else self.vsig[jdrop], jdrop

        # Calculate the step to the new vertex and its sign
        temp = gamma * rho * vsig_max
        self.dx = temp * self.simi[..., jdrop]

        ssum = np.array(np.dot(a, dx)).ravel()
        temp = self.datmat[-1, :-1]

        cvmaxp = max((0, *(-ssum[:-1] -temp)))
        cvmaxm = max((0, *(ssum[:-1] -temp)))

        cond = (self.parmu * (cvmaxp - cvmaxm) > (2 * ssum[-1]))
        dxsign = -1 if cond else 1

        # Update the elements of SIM and SIMI, and set the next X
        self.dx *= dxsign
        self.sim[jdrop] = self.dx
        self.simi[..., jdrop] /= np.dot(self.simi[..., jdrop], self.dx)

        pdot = np.dot(dx, simi)
        target = self.simi[..., jdrop]
        self.simi -= np.multiply(pdot, self.simi)
        self.simi[..., jdrop] = target
        
        self.x = self.sim[-1] + self.dx
        # GOTO 40
        
        
    def _ajustments(self):
        # Calculate DX=x(*)-x(0). Branch if the length of DX is less than 0.5*RHO
        self.trstlp()
        if ifull == 0:
            temp = sum(self.dx ** 2)
            cond = (temp < 0.25 * (self.rho ** 2)) 
            if cond:
                self.ibrnch = 1
                return # GOTO L550

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
                # GOTO L140
                return
            mask = (temp == phi)
            if mask.any() and (self.parmu == 0):
                if datmat[-1][mask].flat[0] < datmat[-1, -1]:
                    # GOTO L140
                    return
                
        self.prerem = (self.parmu * self.prerec) - ssum[-1]

        # Calculate the constraint and objective functions at x(*). Then find the 
        # actual reduction in the merit function
        self.x = self.self.optimal_vertex + self.dx
        self.ibrnch = 1

        # GOTO L40
        return

    def replace_simplex_vertice(self):
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
        jdrop = 0
        temp = abs(np.array(self.dx * self.simi))
        mask = (temp > ratio)
        if mask.any():
            jdrop = np.arange(self.n)[mask].flat[-1]
            
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
            jdrop = lflag
        if jdrop == 0:
            # GOTO L550
            return

        self.sim[jdrop] = self.dx
        self.dx * self.simi[..., jdrop]
        
        # Revise the simplex by updating the elements of SIM, SIMI and DATMAT
        temp = (self.dx * self.simi[...,0]).flat[0]
        self.simi[..., jdrop] /= temp
        target = self.simi[..., jdrop]
        temp = self.dx * self.simi
        self.simi -= np.repeat(np.array(target), len(temp)).reshape(self.simi.shape) * temp
        self.simi[..., jdrop] = target
        self.datmat[..., jdrop] = np.array((*self.con, self.fval, self.resmax))

        # Branch back for further iterations with the current RHO
        if (trured > 0) and (trured >= prerem * 0.1):
            # GOTO L140
            return

        
    def reduce_rho_reset_parmu(self):
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

                
    def best_calculated_values(self):
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
        self.nact = 0
        self.nactx = 0
        self.icon = 0
        self.sp = None
        self.spabs = None
        self.tot = None
        self.ratio = None
        self.iout = None # To drop?
        self.kk = None
        self.stpful = None
        self.step = None
        
        self.sdirn = None # m + 1
        self.dxnew = None # n
        self.vmultd = None # n
        
        self.resmax, self.icon = max(zip((0, *cobyla.con), (None, range(self.cobyla.m))))
        self.iact = np.arange(self.cobyla.m + 1)
        self.vmultc = resmax - self.cobyla.con

        cobyla.ifull = 1
        cobyla.dx = np.zeros(self.cobyla.n)

        
    def L60(self):
        # End the current stage of the calculation if 3 consecutive iterations
        # have either failed to reduce the best calculated value of the objective
        # function or to increase the number of active constraints since the best
        # value was calculated. This strategy prevents cycling, but there is a
        # remote possibility that it will cause premature termination
        self.optold = 0
        self.icount = 0
        self.L70()

        
    def L70(self):
        optnew = resmax if (mcon == self.cobyla.m) else -np.dot(self.dx, self.a[-1])

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
                # GOTO L490
                return

        # If ICON exceeds NACT, then we add the constraint with index IACT(ICON) to
        # the active set. Apply Givens rotations so that the last N-NACT-1 columns
        # of Z are orthogonal to the gradient of the new constraint, a scalar
        # product being set to zero if its nonzero value could be due to computer
        # rounding errors. The array DXNEW is used for working space
        if (self.icon <= self.nact):
            return self.L260()

        self.kk = self.iact[self.icon] # TODO: Warning this. Try to remove! 
        self.dxnew = self.a[kk]
        self.tot = 0

        for k in range(self.cobyla.n - 1, 0, -1):
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
                self.z[k + 1] = alpha * self.z[k + 1]
                
        return self.add_new_constrain()


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
            # GOTO L490
            return

        return self.revise_lagrange_multipliers_reorder_active_cons()
    

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
            # GOTO L490
            return
        
        self.zdota[self.nact] = temp
        self.vmultc[self.icon] = 0
        self.vmultc[self.nact] = ratio

        return self.L210()

        
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
        
        return self.L340()


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
        return self.L340()


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
            # GOTO L490
            return
        
        temp = (ss * dd) ** 0.5
        if (abs(sd) >= (temp * 1e-6)):
            temp = ((ss * dd) + (sd ** 2)) ** 0.5
            
        self.stpful = dd / (temp + sd)
        self.step = self.stpful
        if(self.mcon == self.cobyla.m):
            acca = self.step + (self.resmax * 0.1)
            accb = self.step + (self.resmax * 0.2)
            if ((self.step >= acca) or (acca >= accb)):
                # GOTO L480
                return
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

            temp = zdwabs + abs(zdotw)
            acca = temp * 0.1
            accb = temp * 0.2
            if (zdwabs >= acca) or (acca >= accb):
                zdotw = 0

            self.vmultd[k] = zdotw / self.zdota[k]
            if (k > 0):
                kk = self.iact[k]
                self.dxnew -= self.vmultd[k] * self.cobyla.a[kk]

        if (self.mcon >= self.cobyla.m):
            self.vmultd[self.nact] = max(0, self.vmultd[self.nact])
            
            
        # Complete VMULTC by finding the new constraint residuals
        self.dxnew = self.cobyla.dx + (self.step * self.sdirn)
        if (self.mcon > self.nact):
            pass
        
                
            
        
        
                
            
        
        

        
            
            
