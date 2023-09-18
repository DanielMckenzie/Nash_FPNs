# ----------------------------------------------------------------------------
# The following code implements the Jacobian-Free Backprop (JFB) as described 
# in "JFB: Jacobian-Free Backpropagation for Implicit Models" by Fung 
# et al (AAAI 2022) and available at:
# https://github.com/howardheaton/jacobian_free_backprop
# ----------------------------------------------------------------------------


import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from utils import project_simplex, LinearMonotoneLayer
from Payoff_Net_Utils import ZSGSolver, VecToAntiSymMatrix

classification = torch.tensor
latent_variable = torch.tensor
image = torch.tensor


def forward_implicit(net, d: image, eps=1.0e-3, max_depth=100,
                     depth_warning=False):
    ''' Fixed Point Iteration forward prop
        With gradients detached, find fixed point. During forward iteration,
        u is updated via R(u,Q(d)) and Lipschitz constant estimates are
        refined. Gradient are attached performing one final step.
    '''

    with torch.no_grad():
        net.depth = 0.0
        Qd = net.data_space_forward(d)
        u = torch.ones(Qd.shape, device=net.device())/3
        u_prev = np.Inf*torch.ones(u.shape, device=net.device())
        all_samp_conv = False
        while not all_samp_conv and net.depth < max_depth:
            u_prev = u.clone()
            u = net.latent_space_forward(u, Qd)
            res_norm = torch.max(torch.norm(u - u_prev, dim=1))
            net.depth += 1.0
            all_samp_conv = res_norm <= eps

    if net.depth >= max_depth and depth_warning:
        print("\nWarning: Max Depth Reached - Break Forward Loop\n")

    attach_gradients = net.training
    if attach_gradients:
        Qd = net.data_space_forward(d)
        return net.map_latent_to_inference(
            net.latent_space_forward(u.detach(), Qd))
    else:
        return net.map_latent_to_inference(u).detach()

# ----------------------------------------------------------------------------
# RPS Architecture: This is our proposed Nash-FPN architecture for RPS
# ----------------------------------------------------------------------------

context       = torch.tensor
action        = torch.tensor
weight_matrix = torch.tensor
payoff_matrix = torch.tensor

class NFPN_RPS_Net(nn.Module):
    def __init__(self, action_size=3, context_size=3):
        '''
            NB: action_size = dim of single player's action space
        '''
        super(NFPN_RPS_Net, self).__init__()
        self.fc_1 = nn.Linear(context_size, 2*action_size)
        self.fc_2 = nn.Linear(2*action_size, 2*action_size)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.action_size = action_size
        
    def device(self) -> str:
        return next(self.parameters()).data.device
    
    def name(self):
        '''
            Returns name of model.
        '''
        return 'NFPN_RPS_Net'
    
    def data_space_forward(self, d: context) -> latent_variable:
        '''
            Map context to latent space.
        '''
        Qd = self.leaky_relu(self.fc_1(d))
        return Qd
    
    def latent_space_forward(self, z1, z2: action) -> action:
        '''
        Forward operator. Note this is of the form
            Proj(z - F(z;d))
        '''
        Fxd = self.fc_2(z1 + z2) + z1
        zz = project_simplex(z1 - Fxd, action_size=self.action_size)

        return zz
    
    def forward(self, d: context, eps=1.0e-5, max_depth=100,
                depth_warning=False):
        '''
            Forward propagation using N-FPN. Finds fixed point of PGD-type
            operator.
        '''
        return forward_implicit(self, d, eps=eps, max_depth=max_depth,
                                depth_warning=depth_warning)
        
    def map_latent_to_inference(self, z: action) -> action:
        '''
            Not really necessary, as latent space coincides with output space.
        '''
        return z
        

## Cocoercive variant of the above
class CoCo_NFPN_RPS_Net(nn.Module):
    def __init__(self, latent_step_size, action_size=3, context_size=3):
        '''
            NB: action_size = dim of single player's action space
        '''
        super(CoCo_NFPN_RPS_Net, self).__init__()
        self.fc_1 = nn.Linear(context_size, 2*action_size)
        self.linear = LinearMonotoneLayer(2*action_size, 0.5)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.action_size = action_size
        self.latent_step_size = latent_step_size
        
    def device(self) -> str:
        return next(self.parameters()).data.device
    
    def name(self):
        '''
            Returns name of model.
        '''
        return 'CoCo_NFPN_RPS_Net'
    
    def data_space_forward(self, d: context) -> latent_variable:
        '''
            Map context to latent space.
        '''
        Qd = self.leaky_relu(self.fc_1(d))
        return Qd
    
    def latent_space_forward(self, z1, z2: action) -> action:
        '''
        Forward operator. Note this is of the form
            Proj(z - F(z;d))
        '''
        Fxd = z1 - self.linear(z1 + z2)
        zz = project_simplex(z1 - self.latent_step_size*Fxd, action_size=self.action_size) # need to use smaller step-size for larger problems

        return zz
    
    def forward(self, d: context, eps=1.0e-5, max_depth=100,
                depth_warning=False):
        '''
            Forward propagation using N-FPN. Finds fixed point of PGD-type
            operator.
        '''
        return forward_implicit(self, d, eps=eps, max_depth=max_depth,
                                depth_warning=depth_warning)
        
    def map_latent_to_inference(self, z: action) -> action:
        '''
            Not really necessary, as latent space coincides with output space.
        '''
        return z
        

# ----------------------------------------------------------------------------
# RPS Architecture: This is the architecture described in "What game are we
# playing?" by Ling, Fang and Kolter. Code is adapted from that available at
# https://github.com/lingchunkai/payoff_learning
# ---------------------------------------------------------------------------- 
        
class Payoff_Net(nn.Module):
    '''
        Mildly adapted version of the RPS architecture proposed in "What game
        are we playing?" by Feng, Kolter and Ling.
    '''
    def __init__(self, action_size, context_size=3):
        super(Payoff_Net, self).__init__()
        self.action_size = action_size
        self.nfeatures = context_size
        num_ind_params = int(action_size*(action_size-1)/2)
        self.num_ind_params = num_ind_params
        self.fc1 = nn.Linear(context_size, context_size)
        self.fc2 = nn.Linear(context_size, num_ind_params)
        self.leaky_relu = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        fullsize = x.size()
        nbatchsize = fullsize[0]
        x = self.fc2(self.leaky_relu(self.fc1(x)))
        
        # Convert to payoff matrices
        P = VecToAntiSymMatrix(x, self.action_size)
        
        # Pass to game solver
        u, v = ZSGSolver.apply(P, self.action_size)
        
        return torch.cat((u, v), dim=1)
    