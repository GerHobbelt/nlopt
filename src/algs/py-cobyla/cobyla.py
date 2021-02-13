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
        self.work = np.zeros((n,))

        self.iact = np.zeros((m + 1,))

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


    def trstlp(self):
        iz = 1
        izdota = iz + self.n * self.n
        ivmc = izdota + self.n
        isdirn = ivmc + self.m + 1
        idxnew = isdirn + self.n
        ivmd = idxnew + self.n

        return 0
        
        
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
        
