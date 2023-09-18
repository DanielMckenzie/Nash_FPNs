import torch
from prettytable import PrettyTable
from time import sleep
import numpy as np
import torch.nn as nn
import torch

class LinearMonotoneLayer(nn.Module):
    def __init__(self, input_dim, monotonicity_param):
        super(LinearMonotoneLayer, self).__init__()

        # Define the weights
        #self.A = nn.utils.spectral_norm(nn.Parameter(torch.randn(input_dim, input_dim)))
        self.A = nn.utils.spectral_norm(nn.Linear(in_features=input_dim, out_features=input_dim))
        self.B = nn.utils.spectral_norm(nn.Linear(in_features=input_dim, out_features=input_dim))
        self.m = monotonicity_param

    def apply_transpose(self, x, weights):
        y = torch.mm(x, weights.t())
        return y

    def forward(self, x):
        #temp = torch.matmul(x, self.A)
        out = self.apply_transpose(self.A(x), self.A.weight)
        out = (1-self.m)*x - out + self.B(x) - self.apply_transpose(x, self.B.weight) #torch.matmul(x, self.B) - torch.matmul(x, self.B.t())
        return out

def project_simplex(y, action_size=3, num_players=2):
    '''
        Takes in x = [u, v], interpreted as a pair of player action profiles.
        returns [proj(u), proj(v)] where proj() denotes projection to prob.
        simplex.
    '''
    num_samples = y.shape[0]
    proj = torch.zeros(y.shape)
    for i in range(num_players):
        ind = [i * action_size + j for j in range(action_size)]
        u = torch.flip(torch.sort(y[:, ind], dim=1)[0], dims=(1,))
        u_sum = torch.cumsum(u, dim=1)
        j = torch.arange(1, action_size + 1, dtype=y.dtype, device=y.device)
        pos_u_expr = u * j + 1.0 - u_sum  > 0 
        pos_u_expr = pos_u_expr.float()
        rho = torch.sum(pos_u_expr, dim=1, keepdim=True)
        rho = rho.long()
        lambd = [(1 - u_sum[sample, rho[sample]-1]) / rho[sample] for sample in range(num_samples)]
        lambd = torch.tensor(lambd)
        lambd = lambd.view(lambd.shape[0], 1)
        proj[:, ind] = torch.clamp(y[:, ind] + lambd, min=0)
    return proj

def model_params(net):
    '''
        Create a nice table showing all model parameters.
    '''
    table = PrettyTable(["Network Component", "# Parameters"])
    num_params = 0
    for name, parameter in net.named_parameters():
        if not parameter.requires_grad:
            continue
        table.add_row([name, parameter.numel()])
        num_params += parameter.numel()
    table.add_row(['TOTAL', num_params])
    return table

def SampleSimplex(num_samples, dim):
    '''
        Sample uniformly from the interior of the probability simplex of dim=dim
    '''
    x = torch.zeros(num_samples, dim+1)
    x[:, :dim-1] = torch.rand(num_samples, dim-1)
    x[:, dim] = 1
    y,_ = torch.sort(x)
    z = y[:, 1:] - y[:, :-1]
    
    return z
    
    
