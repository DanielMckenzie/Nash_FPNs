import torch
import torch.optim as optim
from Networks import NFPN_RPS_Net
from Generate_Data import create_data
from torch.utils.data import DataLoader
import time

# ---------------------------------------------------------------------------
# Fetch data, create data loaders
# ---------------------------------------------------------------------------

state = torch.load('./data/RPS_training_data.pth')
train_dataset = state['train_dataset']
test_dataset = state['test_dataset']
train_size = state['train_size']
test_size = state['test_size']

train_loader = DataLoader(dataset=train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=test_size, shuffle=False)

# ---------------------------------------------------------------------------
# Create training setup
# ---------------------------------------------------------------------------

model = NFPN_RPS_Net()
learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
fixed_pt_tol = 1.0e-6
criterion = nn.MSELoss()
max_epochs = 1000
save_str = 'NFPN_RPS_data.pth'

test_loss_hist = []  
train_loss_hist = [] 
depth_hist = [] 
train_loss_ave = 0

fmt  = '[{:4d}/{:4d}]: train loss = {:7.3e} | test_loss = {:7.3e} | depth ' 
fmt += '= {:5.1f} | lr = {:5.1e} | fxt_pt_tol = {:5.1e} | time = {:4.1f} sec' 

# ---------------------------------------------------------------------------
# Train!
# ---------------------------------------------------------------------------

print('\nTraining Nash-FPN for RPS')

for epoch in range(max_epochs):
    start_time = time.time()
    for x_batch, d_batch in train_loader:
        model.train()
        optimizer.zero_grad()
        x_pred = model(d_batch, eps=fixed_pt_tol)
        loss = criterion(x_pred, x_batch)
        train_loss_ave = 0.95*train_loss_ave + 0.05*loss.item()
        loss.backward()
        optimizer.step()

    for x_batch, d_batch in test_loader:
        x_pred = model(d_batch, eps=fixed_pt_tol)
        test_loss = criterion(x_pred, x_batch)

    time_epoch = time.time() - start_time

    print(fmt.format(epoch+1, max_epochs, train_loss_ave, test_loss.item(),
                     model.depth, optimizer.param_groups[0]['lr'],
                     fixed_pt_tol, time_epoch))

    test_loss_hist.append(test_loss.item())
    train_loss_hist.append(loss.item())
    depth_hist.append(model.depth)

    if epoch % 10 == 0 or epoch == max_epochs-1:
        state = {
                'fixed_pt_tol': fixed_pt_tol,
                'T_state_dict': model.state_dict(),
                'test_loss_hist': test_loss_hist,
                'train_loss_hist': train_loss_hist,
                'depth_hist': depth_hist
                }
        torch.save(state, save_str)
