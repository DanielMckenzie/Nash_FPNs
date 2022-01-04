#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 13:36:30 2022

@author: danielmckenzie
"""

import torch
import torch.optim as optim
import torch.nn as nn
from Networks import Payoff_Net
from Payoff_Net_Utils import *
from Generate_Data import create_data
from torch.utils.data import DataLoader
import time

# ---------------------------------------------------------------------------
# Fetch data, create data loaders
# ---------------------------------------------------------------------------

state = torch.load('./data/RPS_training_data_QRE.pth')
train_dataset = state['train_dataset']
test_dataset = state['test_dataset']
train_size = state['train_size']
test_size = state['test_size']

train_loader = DataLoader(dataset=train_dataset, batch_size=200, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=test_size, shuffle=False)

# ---------------------------------------------------------------------------
# Create training setup
# ---------------------------------------------------------------------------

model = Payoff_Net()
model = model.to(torch.float) # convert from double precision to single precision
learning_rate = 1e-4
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
fixed_pt_tol = 1.0e-5
criterion = nn.MSELoss()
max_epochs = 100
save_str = 'Payoff_Net_data.pth'

test_loss_hist = []  
train_loss_hist = [] 
train_loss_ave = 0

fmt = '[{:4d}/{:4d}]: train loss = {:7.3e} | test_loss = {:7.3e} ' 
fmt += '| lr = {:5.1e} | fxt_pt_tol = {:5.1e} | time = {:4.1f} sec'

# ---------------------------------------------------------------------------
# Train!
# ---------------------------------------------------------------------------

print('\nTraining Payoff_Net for RPS\n')
print(model)

for epoch in range(max_epochs):
    start_time = time.time()
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
        
    time_epoch = time.time() - start_time
    
    print(fmt.format(epoch+1, max_epochs, train_loss_ave, test_loss.item(),
                     optimizer.param_groups[0]['lr'], time_epoch))
    
    test_loss_hist.append(test_loss.item())
    train_loss_hist.append(loss.item())
    
    if epoch % 10 == 0 or epoch == max_epochs-1:
        state = {
                'Payoff_Net_State_dict': model.state_dict(),
                'test_loss_hist': test_loss_hist,
                'train_loss_hist': train_loss_hist,
                }
        torch.save(state, save_str)
