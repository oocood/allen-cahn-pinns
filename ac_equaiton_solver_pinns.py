# due to failure in direct in combining data loss and pde loss, 
# according to tutorial on github, this program is written by
# fuyi li at 12 12 2024 Thursday in a more easy-reading way

import torch
import torch.autograd as autograd
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from pyDOE import lhs # latin Hypercupper_bounde sampling
# import scipy.io
import time

# setting default data type and random seed
torch.set_default_dtype(torch.float)
torch.manual_seed(42)
np.random.seed(42)

# device configuration
# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
# print(device)

# setting total epoch num
steps = 100000
lr = 0.1 # learning rate
layers = np.array([2, 50, 50, 50, 50, 50, 50, 50, 50, 50, 1]) # 5 hidden layers

# setting data points and pde residual points
N_u = 500 # total data points including boundary and initial condition
N_f = 10000  # collcation points for calculating pde loss
kappa = 0.01 # diffuse boundary width for alen-cahn equation

# generate data
start_time = time.time()
data_path = './allen_cahn_data/imex1e-05.txt'
data = np.loadtxt(data_path).reshape((-1, 256))
x_length = 2
x_nodenum = 256
x = x_length/x_nodenum*np.arange(-x_nodenum/2, x_nodenum/2, 1)
total_time = 1
delta_t = 1e-5
steps_per_saved = 100
t = np.arange(0, total_time, delta_t)[::steps_per_saved]

# prepare data
X, T = np.meshgrid(x, t)
x_test = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))

# domain bounds
lower_bound = x_test[0]
upper_bound = x_test[-1]
print(upper_bound.shape)
u_true = data.flatten()[:,None]

# training data
# initial data -1<=x<=1 and t=0
initial_x = np.hstack((X[0, :][:, None], T[0, :][:, None]))
initial_u = data[0, :][:, None]

# boundary data x=-1, 0<t<100
lower_x = np.hstack((X[:, 0][:, None], T[:, 0][:, None]))
lower_u = data[:, 0][:, None]

# boundary data x=1, 0<t<100
upper_x = np.hstack((X[:, -1][:, None], T[:, -1][:, None]))
upper_u = data[:, -1][:, None]

print(initial_x.shape, lower_x.shape, upper_x.shape)
# x_train = np.vstack([initial_x, lower_x, upper_x])
# u_train = np.vstack([initial_u, lower_u, upper_u])

# choose random 3/4 initial and boundary points for training
idx_initial = np.random.choice(initial_x.shape[0], int(initial_x.shape[0]*0.75), replace=False)
idx_boundary = np.random.choice(upper_bound.shape[0], int(upper_bound.shape[0]*0.75), replace=False)
x_train_initial = initial_x[idx_initial, :]
x_train_lowerbound = lower_x[idx_boundary, :]
x_train_upperbound = upper_x[idx_boundary, :]

u_train_initial = initial_u[idx_initial, :]
u_train_lowerbound = lower_u[idx_boundary, :]
u_train_upperbound = upper_u[idx_boundary, :]

# collocation points in domain
# using latin hypercupper_bounde sampling
x_train_f = lower_bound + (upper_bound - lower_bound)*lhs(2, N_f)
x_train_f = np.vstack((x_train_f, x_train_initial, x_train_lowerbound, x_train_upperbound))

# change data into tensor type
x_train_f = torch.from_numpy(x_train_f).float()

x_train_initial = torch.from_numpy(x_train_initial).float()
u_train_initial = torch.from_numpy(u_train_initial).float()

x_train_lowerbound = torch.from_numpy(x_train_lowerbound).float()
u_train_lowerbound = torch.from_numpy(u_train_lowerbound).float()

x_train_upperbound = torch.from_numpy(x_train_upperbound)
u_train_lowerbound = torch.from_numpy(u_train_upperbound)

x_test = torch.from_numpy(x_test).float()
u = torch.from_numpy(u_true).float()

# Define neural network
class PINNs(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.activation = nn.Tanh() # setting activation function
        self.loss_func = nn.MSELoss(reduction='mean')

        # define network layers
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        # self.iter = 0

        # Xavier normal initialization
        for i in range(len(layers)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0) 
            nn.init.zeros_(self.linears[i].bias.data) # set zero biases for each layer

    def forward(self, x):
        if torch.is_tensor(x) !=True:
            x = torch.from_numpy(x)
        # preprocessing input
        # print(upper_bound.shape)
        upper_bound_device = torch.from_numpy(upper_bound).float()
        lower_bound_device = torch.from_numpy(lower_bound).float()
        x = (x - lower_bound_device)/(upper_bound_device - lower_bound_device)
        tmp = x.float()
        for i in range(len(layers)-2):
            tmp1 = self.linears[i](tmp)
            tmp = self.activation(tmp1)
        y_pred = self.linears[-1](tmp)
        return y_pred
    
    def loss_ic(self, x, y): # initial condition loss
        loss_ic = self.loss_func(self.forward(x), y)
        return loss_ic 
    
    def loss_bc(self, x_lower, x_upper): # boundary condition loss
        loss_bc = self.loss_func(self.forward(x_lower), self.forward(x_upper))
        return loss_bc
    
    def loss_pde(self, x): 
        x_copy = x.clone()
        x_copy.requires_grad = True
        u = self.forward(x_copy)
        u_x_t = autograd.grad(u, x_copy, torch.ones([x_copy.shape[0], 1]), create_graph= True, retain_graph=True)[0]
        u_xx_tt = autograd.grad(u_x_t, x_copy, torch.ones(x_copy.shape), create_graph=True, retain_graph=True)[0]
        u_x = u_x_t[:, [0]] # special slice instead of u_x_t[:, 0] it return n*1 unsqueeze
        u_t = u_x_t[:, [1]]
        u_xx = u_xx_tt[:, [0]]
        f = u_t - kappa**2*u_xx + u**3 - u
        loss_pde = self.loss_func(f, torch.zeros_like(f))
        return loss_pde
    
    def loss(self, x_intial, y_initial, x_lower, x_upper, x_pde):
        loss_ic = self.loss_ic(x_intial, y_initial)
        loss_bc = self.loss_bc(x_lower, x_upper)
        loss_pde = self.loss_pde(x_pde)
        return loss_ic + loss_bc + loss_pde
    
    def closure(self):
        optimizer_lbfgs.zero_grad()
        loss = self.loss(x_train_initial, u_train_initial, x_train_lowerbound, x_train_upperbound, x_train_f)
        loss.backward()
        return loss
    
    def test(self):
        u_pred = self.forward(x_test)
        error_vec = torch.norm((u - u_pred),2)/torch.norm(u, 2)
        u_pred = u_pred.reshape((-1, 256))
        return error_vec, u_pred
    
# create network object
pinnsnet = PINNs(layers)
print(pinnsnet)
params = list(pinnsnet.parameters())

# define optimizer method
optimizer_adam = torch.optim.Adam(pinnsnet.parameters(), lr=0.001)
optimizer_lbfgs = torch.optim.LBFGS(pinnsnet.parameters(), lr=0.0001, 
                              max_iter=100000, max_eval=None,
                              tolerance_grad=1e-11,
                              tolerance_change=1e-11,
                              history_size = 100,
                              line_search_fn = 'strong_wolfe')
num_epoches = 4000
save_interval = 100
save_dir = './pinns_1d_allen_cahn_model/'
for epoch in range(num_epoches):
    if epoch<=3000:
        optimizer_adam.zero_grad()
        loss = pinnsnet.loss(x_train_initial, u_train_initial, x_train_lowerbound, x_train_upperbound, x_train_f)
        loss.backward()
        optimizer_adam.step()
    else:
        # optimizer_lower_boundfgs.zero_grad()
        # loss = pinnsnet.loss(x_train_b, u_train_b, x_train_f)
        # loss.backward()
        optimizer_lbfgs.step(pinnsnet.closure)
    if epoch % 50 == 0:
        error_vec, _ = pinnsnet.test()
        print(loss, error_vec)
    if (epoch + 1) % save_interval == 0:
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': pinnsnet.state_dict(),
            'optimizer': optimizer_lbfgs.state_dict(),
        }
        filename = f'{save_dir}/checkpoint_{epoch+1}.pth'
        torch.save(checkpoint, filename)
        print(f'Model saved at epoch {epoch+1}: {filename}')

# start training
# optimizer.step(pinnsnet.closure)
elapsed = time.time() - start_time
print(f'Training time is: {elapsed:.4f} s!')

# accuacy test
with torch.no_grad():
    error_vec, u_pred = pinnsnet.test()
    u_pred = u_pred.numpy()
    # u_pred = u_pred.reshape((-1,256))
    plt.imshow(u_pred.T, cmap='coolwarm')
    plt.xlabel('t axis')
    plt.ylabel('x axis')
    plt.title('pinns result')
    plt.colorbar()
    plt.savefig('./pinns_1d_allen_cahn_model/train_result.png', dpi=300)
print(f'Test error is: {error_vec:.5f}')
