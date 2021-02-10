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
        self.sim = self.rho * np.eye((n, n))
        self.optimal_vertex = self.X.copy()

        # inverse simplex
        self.simi = (1 / self.rho) * np.eye((n, n))

        # for each vertex, m constrains values, f, resmax
        # last one for the best vertex
        self.datmat = np.zeros((n + 1, m + 2))

        self.vsig = np.zeros((n + 1,  m))
        self.veta = np.zeros((n,))
        self.sigb = np.zeros((n,))
        self.dx = np.zeros((n,))
        self.work = np.zeros((n,))

        self.iact = np.zeros((m + 1,))

        ### Local
        self.alpha = 0.25
        self.beta = 2.1
        self.gamma = 0.5
        self.delta = 1.1
        self.parmu = 0
        
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
            
        
        
        
        
        
        
        
        
        
