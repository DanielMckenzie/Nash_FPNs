'''
This file collects some utils for implementing the Payoff-Net architecture, as
described in "What game are we playing?" by Ling, Fang and Kolter. Code is 
adapted from that available at https://github.com/lingchunkai/payoff_learning
'''

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import logging
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Declare Games classes
# ----------------------------------------------------------------------------

class Game(object):
    '''
    Normal form game
    '''
    
    def __init__(self, Pu, Pv):
        self.P = [Pu, Pv]
        
class ZeroSumGame(Game): 
    def __init__(self, P): 
        '''
        Construct a 2-player game with P being the (n x m) payoff matrix.
        The vaue of the game is u^T P v, where u is the minimizing player.
        :param payoff matrix defined by (n x m) numpy array 
        '''
        self.Pz = P
        super(ZeroSumGame, self).__init__(-P, P)
    
    def SetP(self, p):
        self.Pz = p
        

# ----------------------------------------------------------------------------
# Code implementing the differentiable game solver.
# ----------------------------------------------------------------------------
        
class Solver(object):
    def __init__(self, g):
        self.g = g
        self.usize = g.Pz.shape[0]
        self.vsize = g.Pz.shape[1]

    def CheckZeroSum():
        pass

    def Solve():
        pass

class GRELogit(Solver):
    '''
    Solves for quantal response equilibrium.
    '''
    
    def __init__(self, g):
        super(GRELogit, self).__init__(g)
        # g is the game.
        
    def Solve(self, alpha=0.1, beta=0.95, tol=10**-3, epsilon=10**-8, 
              min_t=10**-20, max_it=50):
        '''
        Solve for QRE using Newton's method.
        :param alpha parameter for line search
        :param beta parameter for line search
        :param tol for termination condition
        :param epsilon minimum requirement for log barrier
        :param min_t minimum factor for line-search
        :param max_it maximum number of allowable iterations 
        :return u, v mixed strategy for QRE
        :return dev2, second derivative of KKT matrix
        '''

        def line_search(x, r, nstep):
            t = 0.999
            while (np.any((x+t*nstep)[:-2] < epsilon) or
                   np.linalg.norm(residual(x + t * nstep)) > (1.0-alpha*t) * np.linalg.norm(r)):
                t *= beta
                if t < min_t:
                    break
            return t
        
        def term_cond(r):
            return True if np.linalg.norm(r) < tol else False
        
        def residual(x):
            return residual_(*unpack(x))
        
        def residual_(u, v, mu, vu):
            # Computes residuals associated to KKT conditions.
            # Should all be = 0 when KKT conditions are satisfied.
            ru = np.atleast_2d(np.dot(self.g.Pz, v) + np.log(u) + 1. + mu*np.ones([self.usize])).T
            rv = np.atleast_2d(np.dot(self.g.Pz.T, u) - np.log(v) - 1 + vu * np.ones([self.vsize])).T
            rmu = [[np.sum(u) - 1.]]
            rvu = [[np.sum(v) - 1.]]
            r = np.concatenate([ru, rv, rmu, rvu], axis=0)
            return r
        
        def unpack(x):
            '''
            Break concatenated x into u, v, mu, vu
            :return u, v, mu, vu
            '''
            return np.split(x, np.cumsum([self.usize, self.vsize, 1]))
        
        ## Now comes the actual solver
        
        # Initialize using all ones
        u = np.ones([self.usize]) / self.usize
        v = np.ones([self.vsize]) / self.vsize
        mu, vu = np.array([0.]), np.array([0.])
        
        for it in range(max_it):
            
            logger.debug('Iteration number %d', it)
            logger.debug('(u, v, mu, vu): %s %s %s %s', u, v, mu, vu)
            
            # Second derivatives of KKT conditions
            # Really the Hessian of the Lagrangian
            matu = np.concatenate([np.diag(1./u), self.g.Pz, np.ones([self.usize, 1]), np.zeros([self.usize, 1])], axis=1)
            matv = np.concatenate([self.g.Pz.T, -np.diag(1./v), np.zeros([self.vsize, 1]), np.ones([self.vsize, 1])], axis=1)
            matmu = np.concatenate([np.ones([1, self.usize]), np.zeros([1, self.vsize]), np.array([[0]]), np.array([[0]])], axis=1)
            matvu = np.concatenate([np.zeros([1, self.usize]), np.ones([1, self.vsize]), np.array([[0]]), np.array([[0]])], axis=1)
            mat = np.concatenate([matu, matv, matmu, matvu], axis=0)
            
            # Constants in Newton's method
            r = residual_(u, v, mu, vu)
            logger.debug('Residual norm: %s', np.linalg.norm(r))
            if term_cond(r):
                logger.debug('Residual norm below tol at iteration %d. Terminating', it)
                break
                
            # Compute the Newton direction
            nstep = np.squeeze(np.linalg.solve(mat, -r))
            logger.debug('Newton Step (u, v, mu, vu): %s %s %s %s', *unpack(nstep))
            
            # Do a line search
            x = np.concatenate([u, v, mu, vu], axis=0)
            t = line_search(x, r, nstep)
            x = x + t * nstep
            u, v, mu, vu = unpack(x)
            
            logger.debug('(u, v, mu, vu): %s %s %s %s', u, v, mu, vu)

            if np.all(np.abs(t * nstep) < tol): break
        
        if not self.VerifySolution(u, v, epsilon=0.0001):
            logger.warning('Solution verification failed! (u, v): %s %s', u, v)

        return u, v, mat
    
    def VerifySolution(self, u, v, epsilon=0.01):
        '''
        Verify if (u, v) is a fixed point of logit response
        :param numpy array, mixed strategy of min player
        :param numpy array, mixed strategy of max player
        :param Tolerance
        '''
        U = np.exp(-np.dot(self.g.Pz, v))
        V = np.exp(np.dot(self.g.Pz.T, u))
        ucheck = U/np.sum(U)
        vcheck = V/np.sum(V)
        checkPass = not (np.linalg.norm(ucheck - u) > epsilon or np.linalg.norm(vcheck - v) > epsilon)
        if checkPass:
            logger.debug('Verify solution (u, uCheck, v, vCheck): %s %s %s %s, checkPass: %s', u, ucheck, v, vcheck, checkPass)
        if not checkPass:
            logger.warning('Verify solution (u, uCheck, v, vCheck): %s %s %s %s, checkPass: %s', u, ucheck, v, vcheck, checkPass)
            logger.warning('P: %s', self.g.Pz)

        return checkPass
    
# ---------------------------------------------------------------------------
# Wrap game solver in an autograd-ready function.
# ---------------------------------------------------------------------------

class ZSGSolver(torch.autograd.Function):
    '''
    Custom autograd function. This layer takes as input a payoff matrix and then
    solves a zero-sum game on the forward pass. Differentiates through it on 
    the backward pass.
    '''
    ## This style of autograd Function is no longer supported. 
    # def __init__(self, usize, vsize):
    #    self.usize, self.vsize = usize, vsize
    #    super(ZSGSolver, self).__init__()

    @staticmethod
    def forward(ctx, input, size=3):
        # input is a batch of payoff matrices.
        # Naive handling of batch-size > 1
        input_np = input.numpy()
        batchsize = input_np.shape[0]
        U = np.zeros([batchsize, size], dtype=np.float64)
        V = np.zeros([batchsize, size], dtype=np.float64)
        # dev2s will store the Jacobians.
        dev2s = np.zeros([batchsize, 2*size + 2, 2*size + 2],
                         dtype=np.float64)
        for i in range(batchsize):
            p = np.squeeze(input_np[i, :, :]) # payoff matrix
            game = ZeroSumGame(p)
            logitSolver = GRELogit(game)
            u, v, dev2 = logitSolver.Solve(alpha=0.15, beta=0.90, tol=10**-15, epsilon=0, min_t=10**-20, max_it=1000)
            # [u, v] is the Nash equilibrium corresponding to payoff p
            U[i, :] = u
            V[i, :] = v
            dev2s[i,:,:] = dev2
            
        U, V = torch.Tensor(U), torch.Tensor(V)
        dev2s = torch.Tensor(dev2s)
        ctx.save_for_backward(input, U, V, dev2s)
        ctx.action_size=size
        
        return U, V
    
    @staticmethod
    def backward(ctx, grad_u, grad_v):
        # backprop gradients of u and v thru solver.
        batchsize = grad_u.shape[0]
        P, U, V, dev2s = ctx.saved_tensors
        size = ctx.action_size
        # P, U, V, dev2a = tuple([x.data.numpy() for x in ctx.saved_variables])
        dP = np.zeros([batchsize, size, size], dtype=np.float64)
        for i in range(batchsize):
            # naive handling of batchsize > 1
            u, v = U[i,:], V[i,:]
            p = P[i, :, :]
            
            # Will convert to numpy, do a bunch of computations and then 
            # convert back to pytorch.
            gu, gv = grad_u[i, :].numpy(), grad_v[i, :].numpy()
            d = np.linalg.solve(dev2s[i, :, :], -np.concatenate([gu, gv, [0.0], [0.0]], axis=0))
            # extract the differentials we want
            du = d[:size]
            dv = d[size: 2*size]
            dp = np.outer(du, v) + np.outer(u, dv)
            dP[i, :, :] = dp
            
        return torch.Tensor(dP), None
    
# ---------------------------------------------------------------------------
# Small tool for converting vector to antisymmetric matrix
# --------------------------------------------------------------------------- 
      
def VecToAntiSymMatrix(x, action_size):
    fullsize = x.size()
    nbatchsize = fullsize[0]
    temp = torch.zeros((nbatchsize, action_size, action_size))
    counter = 0
    for i in range(action_size-1):
        for j in range(i+1,action_size):
            temp[:, i, j] = x[:,counter]
            temp[:, j, i] = - x[:,counter]
            counter +=1
    return temp

    