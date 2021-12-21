'''
    This script generates the data (i.e. context, Nash equilibrium pairs) for 
    regularized RPS (so the NE can also be thought of as a Quantal Response 
    Equilibrium).
'''
import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
from utils import project_simplex

context       = torch.tensor
action        = torch.tensor
weight_matrix = torch.tensor
payoff_matrix = torch.tensor

# ---------------------------------------------------------------------------
# Generate random seed. Fix matrix relating context to payoff matrix.
# ---------------------------------------------------------------------------

seed = 30
torch.manual_seed(seed)
W = torch.rand(3, 3) * torch.tensor([0.5, 10, 20])
W = W.permute(1,0)
print(W)

def sample_context(num_samples: int) -> context:
    return torch.rand(num_samples, 3) 

def create_payoff_matrix(d: context) -> payoff_matrix:
    num_samples = d.shape[0] 
    action_size = d.shape[1]
    Wd = d.mm(W.permute(1,0)) 
    B = torch.zeros(num_samples, action_size, action_size)
    B[:,0,1] = -Wd[:, 0]
    B[:,0,2] =  Wd[:, 1]
    B[:,1,2] = -Wd[:, 2]
    B -= B.permute(0, 2, 1)
    return B

# ---------------------------------------------------------------------------
# The game gradient. Note it includes a term from the entropic regularization.
# ---------------------------------------------------------------------------

def F(x: action, d: context, player_size=3) -> action:
    B = create_payoff_matrix(d)
    x = x.view(x.shape[0], x.shape[1], 1)
    Fx_1 = B.bmm(x[:, player_size:, :]) + x[:, :player_size, :] + 1
    Bt = B.permute(0, 2, 1)
    Fx_2 = -Bt.bmm(x[:, :player_size, :]) - x[:, player_size:, :] - 1
    Fx = torch.cat((Fx_1, Fx_2), dim=1)
    return Fx.view(Fx.shape[0], Fx.shape[1])

# ---------------------------------------------------------------------------
# Solver. Just PGD-style algorithm with game gradient. Guaranteed to converge 
# as game gradient is monotone.
# ---------------------------------------------------------------------------
    
def get_nash_eq(d: context, fxd_pt_tol=1e-5, max_iter=10000, step_size=5e-3,
                action_size=6, debug_mode=False) -> action:
    num_samples = d.shape[0]
    x           = torch.rand(num_samples, action_size)
    conv        = False
    step        = 0
    while not conv and step < max_iter:
        x_prev = x.clone()
        x = project_simplex(x - step_size * F(x, d))
        res = torch.max(torch.norm(x - x_prev, dim=1)) 
        step += 1
        conv = res < fxd_pt_tol 
        if step % 5 == 0 and debug_mode:
            fmt_str = "Step {:5d}: |xk - xk_prev| = {:2.2e}   x[0,:] = "
            print(fmt_str.format(step, res) + str(x[0,:]))
    return x

# ---------------------------------------------------------------------------
# Functions for creating data. Outputs data loaders
# ---------------------------------------------------------------------------

def create_data(train_batch_size=20, test_batch_size=10,
                train_size=100, test_size=10):
    '''
        Ideally we would create the data set and store it, so that all nets 
        are trained on the same data. However I'm not sure how to do this.
        As the random seed is fixed, the current approach is probably OK.
    '''
    d_context = sample_context(train_size + test_size)
    x_true = get_nash_eq(d_context, max_iter=100, debug_mode=True)
    dataset = TensorDataset(x_true, d_context)
    
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
    state_path = save_dir + 'RPS_training_data.pth'
    torch.save(state, state_path)
    
# ---------------------------------------------------------------------------
# Run.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    create_data()

