#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark Nash-FPN against Payoff-Net for symmetric, regularized matrix game
(aka generalized rock-paper-scissors)

"""
import pickle as pkl
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from Networks import NFPN_RPS_Net, Payoff_Net
from Generate_Data import create_data
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time

# ---------------------------------------------------------------------------
# Initialize arrays for recording total train time, final test accuracy.
# ---------------------------------------------------------------------------
num_sizes = 5
num_trials = 3

NFPN_time = np.zeros((num_sizes, num_trials))
Payoff_Net_time = np.zeros((num_sizes, num_trials))

NFPN_acc = np.zeros((num_sizes, num_trials))
Payoff_Net_acc = np.zeros((num_sizes, num_trials))

# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

for b in range(0, num_trials):
    for a in range(5): # vary the dimension of the action space
        # ------------------------------------------------------------------------
        # Some global variables for both models
        # ------------------------------------------------------------------------
        action_size = 5*a + 5 # action size
        max_epochs = 100
        
        # ------------------------------------------------------------------------
        # Fetch data, create data loaders
        # ------------------------------------------------------------------------
        state = torch.load('./data/RPS_training_data_QRE'+str(action_size)+'.pth')
        train_dataset = state['train_dataset']
        test_dataset = state['test_dataset']
        train_size = state['train_size']
        test_size = state['test_size']
    
        train_loader = DataLoader(dataset=train_dataset, batch_size=200,
                                  shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=test_size,
                                 shuffle=False)
        
        # ------------------------------------------------------------------------
        # Initialize N-FPN model
        # ------------------------------------------------------------------------
        model = NFPN_RPS_Net(action_size)
        learning_rate = 1e-3
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, 'min')
        fixed_pt_tol = 1.0e-5
        criterion = nn.MSELoss()
        NFPN_save_str = 'experiment1/NFPN_RPS_data'+str(action_size)+'.pth'
    
        test_loss_hist = []
        train_loss_hist = []
        depth_hist = []
        train_loss_ave = 0
    
        fmt = '[{:4d}/{:4d}]: train loss = {:7.3e} | test_loss = {:7.3e} | depth '
        fmt += '= {:5.1f} | lr = {:5.1e} | fxt_pt_tol = {:5.1e}'
        fmt += '| time = {:4.1f} sec'
    
        # ------------------------------------------------------------------------
        # Train N-FPN
        # ------------------------------------------------------------------------
        print('\nTraining Nash-FPN for RPS with act_size='+str(action_size)+'\n')
        print(model)
        train_start_time = time.time()
        
        for epoch in range(max_epochs):
            print(epoch)
            start_time = time.time()
            num_batches = len(train_loader)
            train_loss = 0
            for x_batch, d_batch in train_loader:
                model.train()
                optimizer.zero_grad()
                x_pred = model(d_batch, eps=fixed_pt_tol)
                loss = criterion(x_pred, x_batch)
                train_loss_ave = 0.95*train_loss_ave + 0.05*loss.item()
                loss.backward()
                optimizer.step()
            
            model.eval()
            for x_batch, d_batch in test_loader: # only one batch in test_loader
                x_pred = model(d_batch, eps=fixed_pt_tol)
                test_loss = criterion(x_pred, x_batch)
    
            scheduler.step(test_loss)
            time_epoch = time.time() - start_time
    
            print(fmt.format(epoch+1, max_epochs, train_loss_ave,
                             test_loss.item(), model.depth,
                             optimizer.param_groups[0]['lr'], fixed_pt_tol,
                             time_epoch))
    
            test_loss_hist.append(test_loss.item())
            train_loss_hist.append(train_loss_ave)
            depth_hist.append(model.depth)
    
            if epoch % 3 == 0 or epoch == max_epochs-1:
                state = {
                        'fixed_pt_tol': fixed_pt_tol,
                        'T_state_dict': model.state_dict(),
                        'test_loss_hist': test_loss_hist,
                        'train_loss_hist': train_loss_hist,
                        'depth_hist': depth_hist
                        }
                torch.save(state, NFPN_save_str)
             
        # ------------------------------------------------------------------------
        # Store data
        # ------------------------------------------------------------------------
        
        total_time = time.time() - train_start_time
        final_acc = test_loss.item()
        NFPN_time[a, b] = total_time
        NFPN_acc[a, b] = final_acc
        
        # ------------------------------------------------------------------------
        # Initialize Payoff-Net model
        # ------------------------------------------------------------------------
        model = Payoff_Net(action_size=action_size)
        learning_rate = 1e-3
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, 'min')
        criterion = nn.MSELoss()
        save_str = 'experiment1/Payoff_Net_RPS_data'+str(action_size)+'.pth'
    
        test_loss_hist = []  
        train_loss_hist = [] 
        train_loss_ave = 0
    
        fmt = '[{:4d}/{:4d}]: train loss = {:7.3e} | test_loss = {:7.3e} ' 
        fmt += '| lr = {:5.1e} | time = {:4.1f} sec'
        
        # ------------------------------------------------------------------------
        # Train Payoff-Net
        # ------------------------------------------------------------------------
        print('\nTraining Payoff-Net for RPS with act_size='+str(action_size)+'\n')
        print(model)
        train_start_time = time.time()
        
        for epoch in range(max_epochs):
            epoch_start_time = time.time()
            model.train()
            for x_batch, d_batch in train_loader:
                optimizer.zero_grad()
                x_pred = model(d_batch)
                loss = criterion(x_pred, x_batch)
                train_loss_ave = 0.95*train_loss_ave + 0.05*loss.item()
                loss.backward()
                optimizer.step()
            
            model.eval()
            for x_batch, d_batch in test_loader:
                x_pred = model(d_batch)
                test_loss = criterion(x_pred, x_batch)
         
            scheduler.step(test_loss)
            time_epoch = time.time() - epoch_start_time
        
            print(fmt.format(epoch+1, max_epochs, train_loss_ave, test_loss.item(),
                         optimizer.param_groups[0]['lr'], time_epoch))
        
            test_loss_hist.append(test_loss.item())
            train_loss_hist.append(train_loss_ave)
        
            if epoch % 10 == 0 or epoch == max_epochs-1:
                state = {
                        'Payoff_Net_State_dict': model.state_dict(),
                        'test_loss_hist': test_loss_hist,
                        'train_loss_hist': train_loss_hist,
                        }
                torch.save(state, save_str)
                
        # ------------------------------------------------------------------------
        # Store data
        # ------------------------------------------------------------------------
        
        total_time = time.time() - train_start_time
        final_acc = test_loss.item()
        Payoff_Net_time[a, b] = total_time
        Payoff_Net_acc[a, b] = final_acc
    

    
    

# ----------------------------------------------------------------------------
# Store results in a pickle file.
# ----------------------------------------------------------------------------
results = {"NFPN_time": NFPN_time,
           "NFPN_acc": NFPN_acc,
           "Payoff_Net_time": Payoff_Net_time,
           "Payoff_Net_acc": Payoff_Net_acc}

myfile = open("experiment1/results_exp1_100epochs.p", "wb")
pkl.dump(results, myfile)
myfile.close()

