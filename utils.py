import torch
from prettytable import PrettyTable
from time import sleep
import time
from tqdm import tqdm
import numpy as np


def project_simplex(y, action_size=3, num_players=2):
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

