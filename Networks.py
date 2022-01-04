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
from utils import project_simplex
from Payoff_Net_Utils import ZSGSolver

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
        u = torch.zeros(Qd.shape, device=net.device())
        u_prev = np.Inf*torch.ones(u.shape, device=net.device())
        all_samp_conv = False
        while not all_samp_conv and net.depth < max_depth:
            u_prev = u.clone()
            u = net.latent_space_forward(u, Qd)
            res_norm = torch.max(torch.norm(u - u_prev, dim=1))
            net.depth += 1.0
            all_samp_conv = res_norm <= eps

        # if net.training:
        #    net.normalize_lip_const(u_prev, Qd)

    if net.depth >= max_depth and depth_warning:
        print("\nWarning: Max Depth Reached - Break Forward Loop\n")

    attach_gradients = net.training
    if attach_gradients:
        Qd = net.data_space_forward(d)
        return net.map_latent_to_inference(
            net.latent_space_forward(u.detach(), Qd))
    else:
        return net.map_latent_to_inference(u).detach()


def forward_explicit(net, d: image, eps=1.0e-3, max_depth=100,
                     depth_warning=False):
    '''
        Apply Explicit Forward Propagation
    '''

    net.depth = 0.0

    Qd = net.data_space_forward(d)
    u = torch.zeros(Qd.shape, device=net.device())
    Ru = net.latent_space_forward(u, Qd)

    return net.map_latent_to_inference(Ru)


def normalize_lip_const(net, u: latent_variable, v: latent_variable):
    ''' Scale convolutions in R to make it gamma Lipschitz
        It should hold that |R(u,v) - R(w,v)| <= gamma * |u-w| for all u
        and w. If this doesn't hold, then we must rescale the convolution.
        Consider R = I + Conv. To rescale, ideally we multiply R by
            norm_fact = gamma * |u-w| / |R(u,v) - R(w,v)|,
        averaged over a batch of samples, i.e. R <-- norm_fact * R. The
        issue is that ResNets include an identity operation, which we don't
        wish to rescale. So, instead we use
            R <-- I + norm_fact * Conv,
        which is accurate up to an identity term scaled by (norm_fact - 1).
        If we do this often enough, then norm_fact ~ 1.0 and the identity
        term is negligible.
    '''
    noise_u = torch.randn(u.size(), device=net.device())
    noise_v = torch.randn(u.size(), device=net.device())
    w = u.clone() + noise_u
    Rwv = net.latent_space_forward(w, v + noise_v)
    Ruv = net.latent_space_forward(u, v + noise_v)
    R_diff_norm = torch.mean(torch.norm(Rwv - Ruv, dim=1))
    u_diff_norm = torch.mean(torch.norm(w - u, dim=1))
    R_is_gamma_lip = R_diff_norm <= net.gamma * u_diff_norm
    if not R_is_gamma_lip:
        violation_ratio = net.gamma * u_diff_norm / R_diff_norm
        normalize_factor = violation_ratio ** (1.0 / net._lat_layers)
        for i in range(net._lat_layers):
            net.latent_convs[i][0].weight.data *= normalize_factor
            net.latent_convs[i][0].bias.data *= normalize_factor
            net.latent_convs[i][3].weight.data *= normalize_factor
            net.latent_convs[i][3].bias.data *= normalize_factor


# ----------------------------------------------------------------------------
# RPS Architecture: This is our proposed Nash-FPN architecture for RPS
# ----------------------------------------------------------------------------

context       = torch.tensor
action        = torch.tensor
weight_matrix = torch.tensor
payoff_matrix = torch.tensor

class NFPN_RPS_Net(nn.Module):
    def __init__(self, action_size=6, context_size=3):
        super(NFPN_RPS_Net, self).__init__()
        # self.fc_1 = nn.Linear(context_size, 5*action_size)
        # self.fc_2 = nn.Linear(5*action_size, action_size)
        # self.fc_3 = nn.Linear(action_size, action_size)
        self.fc_1 = nn.Linear(context_size, action_size)
        self.fc_2 = nn.Linear(2*action_size, 5*action_size)
        self.fc_3 = nn.Linear(5*action_size, action_size)
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
        # Qd = self.fc_2(self.leaky_relu(self.fc_1(d)))
        Qd = self.fc_1(d)
        return Qd
    
    def latent_space_forward(self, z1, z2: action) -> action:
        '''
        Forward operator. Note this is of the form
            Proj(z - F(z;d))
        '''
        # zz = z1 + z2
        xd = torch.cat((z1, z2), dim=1)
        Fxd = z1 + self.fc_3(self.leaky_relu(self.fc_2(xd)))
        # zz = project_simplex(zz - self.fc_3(self.leaky_relu(zz)))
        zz = project_simplex(z1 - Fxd)

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
# playing?" by Kolter, Fang and Ling. Code is adapted from that available at
# 
# ---------------------------------------------------------------------------- 
        
class Payoff_Net(nn.Module):
    '''
        Mildly adapted version of the RPS architecture proposed in "What game
        are we playing?" by Feng, Kolter and Ling.
    '''
    def __init__(self, size=3, nfeatures=3):
        super(Payoff_Net, self).__init__()
        self.size = 3
        self.usize, self.vsize = size, size
        self.nfeatures = nfeatures
        self.fc1 = nn.Linear(nfeatures, 3, bias=False)
        self.fc1.weight.data = torch.DoubleTensor(np.zeros([size, nfeatures])) # control the initialization?
        
    def forward(self, x):
        fullsize = x.size()
        nbatchsize = fullsize[0]
        temp = torch.DoubleTensor(np.zeros((nbatchsize, 3, 3)))
        x = self.fc1(x)
        
        # now set up to use differentiable game solver
        x = x.view(-1, 3)
        
        temp[:, 0, 1], temp[:, 1, 0] = x[:, 0], -x[:, 0]
        temp[:, 0, 2], temp[:, 2, 0] = -x[:, 1], x[:, 1]
        temp[:, 1, 2], temp[:, 2, 1] = x[:, 2], -x[:, 2]
        
        u, v = ZSGSolver.apply(temp, self.size)
        # u, v = solver(temp)
        
        return torch.cat((u, v), dim=1)  # Different to original code, return u and v strategies as a tuple.
    