'''
    This script generates the data (i.e. context, Nash equilibrium pairs) for
    regularized RPS (so the NE can also be thought of as a Quantal Response
    Equilibrium).
'''

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
from utils import project_simplex, SampleSimplex
from Payoff_Net_Utils import VecToAntiSymMatrix

context       = torch.tensor
action        = torch.tensor
weight_matrix = torch.tensor
payoff_matrix = torch.tensor

# ---------------------------------------------------------------------------
# Generate random seed. Fix matrix relating context to payoff matrix.
# ---------------------------------------------------------------------------

action_size = 100
context_size = 3


# The number of independent params in a action_size-by-action_size
# antisym matrix 

num_ind_params = int(action_size*(action_size-1)/2)

# seed = 30
# torch.manual_seed(seed)
W = torch.rand((num_ind_params, context_size))* 10

def sample_context(num_samples: int) -> context:
    return torch.rand(num_samples, 3) 

def create_payoff_matrix(d: context, action_size) -> payoff_matrix:
    ind_params = torch.matmul(W, d.permute(1, 0))
    P = VecToAntiSymMatrix(ind_params.permute(1, 0), action_size)
    return P

# ---------------------------------------------------------------------------
# The game gradient. Note it includes a term from the entropic regularization.
# ---------------------------------------------------------------------------

def F(x: action, d: context, action_size) -> action:
    B = create_payoff_matrix(d, action_size)
    x = x.view(x.shape[0], x.shape[1], 1)
    Fx_1 = B.bmm(x[:, action_size:, :]) + torch.log(x[:, :action_size, :] + 1e-12) + 1
    Bt = B.permute(0, 2, 1)
    Fx_2 = -Bt.bmm(x[:, :action_size, :]) + torch.log(x[:, action_size:, :] + 1e-12) + 1
    Fx = torch.cat((Fx_1, Fx_2), dim=1)
    return Fx.view(Fx.shape[0], Fx.shape[1])

# ---------------------------------------------------------------------------
# Solver. Just PGD-style algorithm with game gradient. Guaranteed to converge 
# as game gradient is monotone.
# ---------------------------------------------------------------------------
    
def get_nash_eq(d: context, action_size, fxd_pt_tol=1e-8, max_iter=20000, 
                step_size=5e-5, debug_mode=False) -> action:
    num_samples = d.shape[0]
    x = torch.ones(num_samples, 2*action_size)/(2*action_size) # initialize at uniform strategy for both players
    conv        = False
    step        = 0
    while not conv and step < max_iter:
        x_prev = x.clone()
        y = project_simplex(x - step_size*F(x, d, action_size), action_size=action_size)
        x = y
        res = torch.max(torch.norm(x - x_prev, dim=1)) 
        step += 1
        conv = res < fxd_pt_tol 
        if step % 5 == 0 and debug_mode:
            fmt_str = "Step {:5d}: |xk - xk_prev| = {:2.2e}   x[5,:] = "
            print(fmt_str.format(step, res) + str(x[5,:]))
        if step % 100 == 0:
            step_size *= 0.5
    return x

# ---------------------------------------------------------------------------
# Functions for creating data. Outputs data loaders
# ---------------------------------------------------------------------------

def create_data(train_batch_size=200, test_batch_size=100,
                train_size=2000, test_size=100):
    '''
        Create and store data.
    '''
    d_context = sample_context(train_size + test_size)
    x_true = get_nash_eq(d_context,action_size, debug_mode=True)
    dataset = TensorDataset(x_true, d_context) # Note non-standard ordering 
    
    train_dataset, test_dataset = random_split(dataset, 
                                               [train_size, test_size])
    state = {
            'W': W,
            'train_dataset': train_dataset,
            'test_dataset': test_dataset,
            'train_size': train_size,
            'test_size': test_size
            }
    save_dir = './data/'
    state_path = save_dir + 'RPS_training_data_QRE' + str(action_size)+ '.pth'
    torch.save(state, state_path)
    
# ---------------------------------------------------------------------------
# Run.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    create_data()

