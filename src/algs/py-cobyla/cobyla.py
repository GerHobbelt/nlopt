import numpy as np

class Cobyla:
    def __init__(self, X, F, C, rhobeg=0, rhoend=1, maxfun=1000):
        n = len(x)
        m = len(C)

        self.n = n
        self.m = m
        self.X = X
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
        self.optimal_vertex = self.X.copy()

        # inverse simplex
        self.simi = (1 / self.rho) * np.matrix(np.eye((n, n)))

        # for each vertex, m constrains values, f, resmax
        # last one for the best vertex
        self.datmat = np.zeros((n + 1, m + 2))

        # a matrix (n x n)
        self.a = None

        self.vsig = None
        self.veta = None
        self.sigb = np.zeros((n,))
        self.dx = np.zeros((n,))
        self.work = np.zeros((n,))

        self.iact = np.zeros((m + 1,))

        # flags
        self.ibrnch = 0
        self.iflag = 0

        ### Local
        self.alpha = 0.25
        self.beta = 2.1
        self.gamma = 0.5
        self.delta = 1.1
        self.parmu = 0
        self.parsig = 0
        
        self.nfvals = 0

        
    def calcfc(self):
        # Error: COBYLA_USERABORT (rc = 3)
        # raise UserWarning('cobyla: user requested end of minimitzation')
        self.nfvals += 1
        self.fval = self.F(self.X)
        self.con = np.array(tuple(constrain(self.X) in self.C))
        self.resmax = max((0, *(-self.con)))

        
    @property
    def current_values(self):
        np.array((*self.con, self.fval, self.resmax))

        
    def _set_datmat_step(self, jdrop):
        f = datmat[-1, -2]
        if f <= self.fval:
            self.X[jdrop] = self.optimal_vertex[jdrop]
        else:
            self.optimal_vertex[jdrop] = self.X[jdrop]
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
            self.X[jdrop] += self.rho
            
        
    def cobylb(self):
        if ((self.nfvals > 0) and (self.nfvals >= self.maxfun)):
            # Error: COBYLA_MAXFUN (rc = 1)
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
        *con, fx = self.datmat[-1, :-1]
        self.con -= con
        self.fval -= fx

        w = np.matrix(self.datmat[:-2] + self.con)
        self.a = (w * self.simi)
        self.a[-1] *= -1

        
    def acceptable_simplex(self):
        self.parsig = self.alpha * rho
        pareta = self.beta * rho
        self.vsig = 1 / (sum(np.array(self.simi)**2, axis=0))**.5
        self.veta = (sum(np.array(self.sim)**2, axis=1))**.5
        self.iflag = not(np.any(vsig < parsig) or np.any(veta > pareta))
        
        if self.ibrnch == 1 or self.iflag == 1:
            return

        
        
            
            
                
            
        
        
            
        
        
        
        
        
        
        
        
        
