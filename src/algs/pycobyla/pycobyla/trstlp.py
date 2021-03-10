import numpy as np


class Trstlp:
    FINISH = 0
    KEEP_STAGE = 1
    CHANGE_STAGE = 2
    
    
    def __init__(self, cobyla: 'Cobyla'):
        self.cobyla = cobyla
        
        self.mcon = self.cobyla.m
        self.z = np.eye(self.cobyla.n) # n * n
        self.zdota = np.zeros(self.cobyla.n) # n
        
        self.icount = 0
        self.optold = 0
        self.nact = -1
        self.nactx = -1
        self.sp = None
        self.spabs = None
        self.tot = None
        self.iout = None # TODO To drop?
        self.stpful = None
        self.step = None
        self.dxnew = None # n
        
        self.sdirn = np.zeros(self.cobyla.n)
        self.resmax, self.icon = max(zip((0, *self.cobyla.con), (-1, *range(self.cobyla.m))))
        self.iact = np.arange(self.cobyla.m + 1)
        self.iact[-1] = -1
        self.vmultc = np.array((*(self.resmax - self.cobyla.con), 0), dtype=np.float)
        self.vmultd = np.zeros(self.cobyla.m + 1)

        # Data to return
        self.ifull = True
        self.dx = np.zeros(self.cobyla.n)


    def run(self):
        # End the current stage of the calculation if 3 consecutive iterations
        # have either failed to reduce the best calculated value of the objective
        # function or to increase the number of active constraints since the best
        # value was calculated. This strategy prevents cycling, but there is a
        # remote possibility that it will cause premature termination
        if self.resmax == 0:
            stage = self.L480_second_stage()
        else:
            while (stage := self.L70()) == self.KEEP_STAGE:
                pass

        while stage != self.FINISH:
            stage = self.L70()

        return self.ifull, self.dx

        
    def L70(self):
        optnew = self.resmax if (self.mcon == self.cobyla.m) else -np.dot(self.dx, self.cobyla.a[-1])

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
                return self.L490_termination_chance()

        # If ICON exceeds NACT, then we add the constraint with index IACT(ICON) to
        # the active set. Apply Givens rotations so that the last N-NACT-1 columns
        # of Z are orthogonal to the gradient of the new constraint, a scalar
        # product being set to zero if its nonzero value could be due to computer
        # rounding errors. The array DXNEW is used for working space
        if (self.icon <= self.nact):
            return self.L260()

        kk = self.iact[self.icon]
        self.dxnew = self.cobyla.a[kk].copy()
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
                if self.tot == 0:
                    self.tot = self.sp 
                else:
                    temp = ((self.sp ** 2) + (self.tot ** 2)) ** 0.5
                    alpha = self.sp / temp
                    beta = self.tot / temp
                    self.tot = temp
                    self.z[k], self.z[k + 1] = \
                        (alpha * self.z[k]) + (beta * self.z[k + 1]), \
                        (alpha * self.z[k + 1]) - (beta * self.z[k])
                
        # Add the new constraint if this can be done without a deletion from the
        # active set

        if (self.tot != 0):
            self.nact += 1
            self.zdota[self.nact] = self.tot
            self.vmultc[self.icon] = self.vmultc[self.nact]
            self.vmultc[self.nact] = 0
            return self.L210(kk)

        if (ratio := self.constant_gradient()) < 0:
            return self.L490_termination_chance()

        return self.revise_lagrange_multipliers_reorder_active_cons(kk, ratio)
    

    def constant_gradient(self):
        # The next instruction is reached if a deletion has to be made from the
        # active set in order to make room for the new active constraint, because
        # the new constraint gradient is a linear combination of the gradients of 
        # the old active constraints. Set the elements of VMULTD to the multipliers
        # of the linear combination. Further, set IOUT to the index of the
        # constraint to be deleted, but branch if no suitable index can be found
        ratio = -1
        for k in range(self.nact, -1, -1):
            temp = self.z[k] * self.dxnew
            zdotv = sum(temp)
            zdvabs = sum(abs(temp))

            acca = zdvabs + (0.1 * (abs(zdotv)))
            accb = zdvabs + (0.2 * (abs(zdotv)))
            if ((zdvabs < acca) and (acca < accb)):
                temp = zdotv / self.zdota[k]
                if ((temp > 0) and (self.iact[k] <= (self.cobyla.m - 1))):
                    tempa = self.vmultc[k] / temp
                    if (ratio < 0) or (tempa < ratio):
                        ratio = tempa
                        self.iout = k # TODO: Drop this ???

                if (k >= 1):
                    self.dxnew -= (temp * self.cobyla.a[self.iact[k]])

                self.vmultd[k] = temp
            else:
                self.vmultd[k] = 0
        
        return ratio
    

    def revise_lagrange_multipliers_reorder_active_cons(self, kk, ratio):
        # Revise the Lagrange multipliers and reorder the active constraints so
        # that the one to be replaced is at the end of the list. Also calculate the
        # new value of ZDOTA(NACT) and branch if it is not acceptable
        k = self.nact + 1
        temp = self.vmultc[:k] - (ratio * self.vmultd[:k])
        self.vmultc[:k] = ((temp > 0 ) * temp)

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

        temp = np.dot(self.z[self.nact], self.cobyla.a[kk])
        if (temp == 0):
            return self.L490_termination_chance()
        
        self.zdota[self.nact] = temp
        self.vmultc[self.icon] = 0
        self.vmultc[self.nact] = ratio

        return self.L210(kk)

        
    def L210(self, kk):
        # Update IACT and ensure that the objective function continues to be
        # treated as the last active constraint when MCON>M
        
        self.iact[self.icon] = self.iact[self.nact]
        self.iact[self.nact] = kk
        if ((self.mcon > self.cobyla.m) and (kk != (self.mcon - 1))):
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
            self.iact[k] = kk
            self.vmultc[k], self.vmultc[self.nact] = self.vmultc[self.nact], self.vmultc[k]

        # If stage one is in progress, then set SDIRN to the direction of the next 
        # change to the current vector of variables
        if (self.mcon > self.cobyla.m): # mcon == (m + 1)
            return self.L320()

        kk = self.iact[self.nact]
        temp = (np.dot(self.sdirn, self.cobyla.a[kk]) - 1) / self.zdota[self.nact]
        self.sdirn -= (temp * self.z[self.nact])
        
        return self.L340()


    def L260(self):
        # Delete the constraint that has the index IACT(ICON) from the active set
        if (self.icon < self.nact):
            isave = self.iact[self.icon]
            vsave = self.vmultc[self.icon]
            for k in range(self.icon, self.nact):
                kp = k + 1
                kk = self.iact[kp]
                
                sp = np.dot(self.z[k], self.cobyla.a[kk])
                temp = ((sp ** 2) + (self.zdota[kp] ** 2)) ** 0.5
                alpha = self.zdota[kp] / temp
                beta = sp / temp
                self.zdota[kp], self.zdota[k] = alpha * self.zdota[k], temp

                temp = ((alpha * self.z[kp]) + (beta * self.z[k]))
                self.z[kp] = ((alpha * self.z[k]) - (beta * self.z[kp]))
                self.z[k] = temp
                self.iact[k] = kk
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
        return self.L340()

        
    def L340(self):
        # Calculate the step to the boundary of the trust region or take the step
        # that reduces RESMAX to zero. The two statements below that include the
        # factor 1.0E-6 prevent some harmless underflows that occurred in a test
        # calculation. Further, we skip the step if it could be zero within a
        # reasonable tolerance for computer rounding errors
        dd = (self.cobyla.rho ** 2)
        mask = (abs(self.dx) >= (self.cobyla.rho * 1e-6))
        dd -= sum(self.dx[mask] ** 2)
        sd = np.dot(self.dx, self.sdirn)
        ss = np.dot(self.sdirn, self.sdirn)

        if (dd <= 0):
            return self.L490_termination_chance()
        
        temp = (ss * dd) ** 0.5
        if (abs(sd) >= (temp * 1e-6)):
            temp = ((ss * dd) + (sd ** 2)) ** 0.5
            
        self.step = self.stpful = dd / (temp + sd)
        if(self.mcon == self.cobyla.m):
            acca = self.step + (self.resmax * 0.1)
            accb = self.step + (self.resmax * 0.2)
            if ((self.step >= acca) or (acca >= accb)):
                return self.L480_second_stage()
            
            self.step = min(self.step, self.resmax)
            
        return self.set_dxnew()

        
    def set_dxnew(self):
        # Set DXNEW to the new variables if STEP is the steplength, and reduce 
        # RESMAX to the corresponding maximum residual if stage one is being done.
        # Because DXNEW will be changed during the calculation of some Lagrange
        # multipliers, it will be restored to the following value later

        self.dxnew = self.dx + (self.step * self.sdirn)
        
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
        k = self.nact

        while True:
            temp = (self.z[k] * self.dxnew)
            zdotw = sum(temp)
            zdwabs = sum(abs(temp))

            acca = zdwabs + (0.1 * abs(zdotw))
            accb = zdwabs + (0.2 * abs(zdotw))
            zdotw *= not((zdwabs >= acca) or (acca >= accb))
        
            self.vmultd[k] = zdotw / self.zdota[k]
            
            if k < 1:
                break

            kk = self.iact[k]
            self.dxnew -= self.vmultd[k] * self.cobyla.a[kk]
            k -= 1

        if (self.mcon > self.cobyla.m):
            self.vmultd[self.nact] = max(0, self.vmultd[self.nact])
                        
        # Complete VMULTC by finding the new constraint residuals
        self.dxnew = self.dx + (self.step * self.sdirn)
        if ((self.mcon - 1) > self.nact):
            confval = self.cobyla.orig_con[:-1]
            for k in range(self.nact + 1, self.mcon):
                kk = self.iact[k]
                temp = self.cobyla.a[kk] * self.dxnew
                ssum = self.resmax - confval[kk] + sum(temp)
                ssumabs = self.resmax + abs(confval[kk]) + sum(abs(temp))

                acca = ssumabs + (0.1 * abs(ssum))
                accb = ssumabs + (0.2 * abs(ssum))
                cond = ((ssumabs >= acca) or (acca >= accb))
                self.vmultd[k] = 0 if cond else ssum

        # Calculate the fraction of the step from DX to DXNEW that will be taken
        ratio = 1
        self.icon = -1            
        for k in range(self.mcon):
            if self.vmultd[k] < 0:
                temp = self.vmultc[k] / (self.vmultc[k] - self.vmultd[k])
                if (temp < ratio):
                    ratio = temp
                    self.icon = k

        # Update DX, VMULTC and RESMAX
        temp = 1 - ratio
        self.dx = (temp * self.dx) + (ratio * self.dxnew)
        self.vmultc = (temp * self.vmultc) + (ratio * self.vmultd)
        self.vmultc *= (self.vmultc > 0)

        if (self.mcon == self.cobyla.m):
            self.resmax = resold + (ratio * (self.resmax - resold))
        
        # If the full step is not acceptable then begin another iteration.
        # Otherwise switch to stage two or end the calculation
        if (self.icon > -1):
            return self.KEEP_STAGE

        if (self.step == self.stpful):
            return self.FINISH

        return self.L480_second_stage()


    def L480_second_stage(self):
        self.mcon = self.cobyla.m + 1
        self.icon = self.iact[-1] = self.cobyla.m
        self.vmultc[-1] = 0
        # L60
        self.icount = self.optold = 0
        return self.CHANGE_STAGE


    def L490_termination_chance(self):
        if (self.mcon == self.cobyla.m):
            return self.L480_second_stage()
        
        self.ifull = False
        return self.FINISH
