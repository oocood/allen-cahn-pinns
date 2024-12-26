# to overcome the issue of converging to trival solution, trying to use warm up through training pinns under simple para in pde
# then record the weights of the layers and restart the training

import torch
import matplotlib.pyplot as plt
import os
import re
import torchvision
import numpy as np
import os
import math
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset, TensorDataset
from torchvision import datasets, transforms
import torch.nn.functional as F
import random
import time 
from pyDOE import lhs

# super para
kappa = 0.01
total_time = 1.0 # when the performance is good  increase the time and restart training

# setting device to cuda if it's available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# setting random seed and deafult data type
torch.set_default_dtype(torch.float)
torch.manual_seed(1234)
np.random.seed(1234)

# free energy function
def free_energy_grad_calculate(c):
    # print(c.item())
    return 20*c**2*(c-1) + 20*c*(c-1)**2 + torch.log(c) - torch.log(1-c)

# Define Pde dataset
class PDEdataset(Dataset): # create dataset of inner domain for calculating pde loss
    def __init__(self, input):
        self.inputs = input
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, idx):
        return self.inputs[idx]

# Define test dataset
class Testdataset(Dataset): # create test data set from pde numerical solution
    def __init__(self, x_test_input, x_test_target):
        self.input = x_test_input
        self.target = x_test_target
    def __len__(self):
        return len(self.target)
    def __getitem__(self, idx):
        return self.input[idx], self.target[idx]

# Define neural network
class PINNs(nn.Module): # The features mat shape is N*3 
    def __init__(self, layers):
        super(PINNs, self).__init__()
        self.layers = layers
        self.loss_func = nn.MSELoss()
        self.activation_func = nn.Tanh()
        self.linear_layer= nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])

        # Xavier normal initialization
        for i in range(len(layers)-1):
            nn.init.xavier_normal_(self.linear_layer[i].weight.data, gain=1.0) 
            nn.init.zeros_(self.linear_layer[i].bias.data) # set zero biases for each layer
        print('neural network has been created!')

    # Define forward formula
    def forward(self, xyt):
        if torch.is_tensor(xyt)!=True:
            xyt = torch.from_numpy(xyt).float()
        input = xyt
        for i in range(len(self.layers)-2):
            tmp = self.linear_layer[i](input)
            input = self.activation_func(tmp)
        output = self.linear_layer[-1](input)
        return output
    
    def loss_pde(self, x_train):
        if not x_train.requires_grad:
            x_train = x_train.clone().detach().requires_grad_(True)
        u = self.forward(x_train)
        u_x_y_t = autograd.grad(u, x_train, torch.ones_like(u, device=device), create_graph=True, retain_graph=True)[0]
        u_xx_yy_tt = autograd.grad(u_x_y_t, x_train, torch.ones_like(u_x_y_t, device=device), create_graph=True, retain_graph=True)[0]
        # u_x = u_x_y_t[:, [0]]
        # u_y = u_x_y_t[:, [1]]
        u_t = u_x_y_t[:, [2]]
        u_xx = u_xx_yy_tt[:, [0]]
        u_yy = u_xx_yy_tt[:, [1]]
        f = u_t - kappa**2*(u_xx+u_yy)+(u**3-u) # pde form 5 using aiming to accerate phase separation
        pde_residual = f
        # pde_residual_grad = autograd.grad(f, x_train, torch.ones_like(f, device=device), create_graph=True, retain_graph=True)[0]
        pde_loss = self.loss_func(pde_residual, torch.zeros_like(pde_residual, device=device))
        return pde_loss

    def loss_bc(self, x_left, x_right, x_lower, x_upper):
        if not x_left.requires_grad:
            x_left = x_left.clone().detach().requires_grad_(True)
        if not x_right.requires_grad:
            x_right = x_right.clone().detach().requires_grad_(True)
        if not x_lower.requires_grad:
            x_lower = x_lower.clone().detach().requires_grad_(True)
        if not x_upper.requires_grad:
            x_upper = x_upper.clone().detach().requires_grad_(True)
        u_left = self.forward(x_left)
        u_right = self.forward(x_right)
        u_lower = self.forward(x_lower)
        u_upper = self.forward(x_upper)
        u_left_grad = autograd.grad(u_left, x_left, torch.ones_like(u_left, device=device), create_graph=True, retain_graph=True)[0]
        u_left_grad_x = u_left_grad[:,[0]]
        u_right_grad = autograd.grad(u_right, x_right, torch.ones_like(u_left, device=device), create_graph=True, retain_graph=True)[0]
        u_right_grad_x = u_right_grad[:,[0]]
        u_lower_grad = autograd.grad(u_lower, x_lower, torch.ones_like(u_lower, device=device), create_graph=True, retain_graph=True)[0]
        u_lower_grad_y = u_lower_grad[:, [1]]
        u_upper_grad = autograd.grad(u_upper, x_upper, torch.ones_like(u_lower, device=device), create_graph=True, retain_graph=True)[0]
        u_upper_grad_y = u_upper_grad[:, [1]]
        loss_left_right = self.loss_func(u_left, u_right)
        loss_lower_upper = self.loss_func(u_lower, u_upper)
        loss_grad_left_right = self.loss_func(u_left_grad_x, u_right_grad_x)
        loss_grad_lower_upper = self.loss_func(u_lower_grad_y, u_lower_grad_y)
        loss_bc = loss_left_right + loss_lower_upper + loss_grad_left_right + loss_grad_lower_upper
        return loss_bc

    def loss_ic(self, x_initial, u_initial):
        u_initial = u_initial[:, None]
        u_pred_initial = self.forward(x_initial)
        # print(u_pred_initial.shape, u_initial.shape)
        loss_ic = self.loss_func(u_pred_initial, u_initial)
        return loss_ic
    
    def loss(self, x_train, x_initial, u_initial, x_left, x_right, x_lower, x_upper):
        loss_pde = self.loss_pde(x_train)
        loss_ic = self.loss_ic(x_initial, u_initial)
        # loss_ic = 0.0 # split out ic loss to see if converge
        loss_bc =  self.loss_bc(x_left, x_right, x_lower, x_upper)
        loss = 2*loss_pde + 10*loss_ic + loss_bc # increase weight of initial loss
        # loss = loss_pde + loss_bc
        return loss, loss_ic, loss_bc
    
    def test(self, x_test, u_exact): # x_test is data points for test the accuracy of the model, u_exact is the exact numerical solution
        with torch.no_grad():
            # print(x_test.shape, u_exact.shape)
            u_test_pred = self.forward(x_test)
            loss_test = torch.norm((u_test_pred - u_exact), 2)/torch.norm(u_exact, 2)
            # print(torch.max(u_test_pred))
        return loss_test
    
    def load_checkpoint(self, checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # load model state dict
        pinnsnet.load_state_dict(checkpoint['state_dict'])
        
        start_epoch = checkpoint['epoch']
        
        print(f"Loaded checkpoint at epoch {start_epoch}.")
        return start_epoch


# using re to extract number from file name
def extract_number(file_name):
    match = re.search(r'(\d+)\.txt$', file_name)
    if match:
        return int(match.group(1))
    else:
        return -1

def sort_files_by_number(file_list):
    return sorted(file_list, key=extract_number)

# read data from path saved by numerical pde solver
start_time = time.time()
x_data_path = './ac_data_para_1/x_axis.txt'
y_data_path = './ac_data_para_1/y_axis.txt'
u_data_path =  './ac_data_para_1/imex/'
# imag_path = './pinns_initial_imag/'
u_chaotic_files = os.listdir(u_data_path)
total_slice = int(len(np.arange(0, total_time, 1e-5))/100) # calculate total slice of file due to total time 
u_sorted_files = sort_files_by_number(u_chaotic_files)[0:total_slice] # sorted file by number before .txt
u_sorted_files = [os.path.join(u_data_path, i) for i in u_sorted_files]

# load and slice for proper scale data num for training
x_data = np.loadtxt(x_data_path)
y_data = np.loadtxt(y_data_path)
# t_data = np.loadtxt(t_data_path)
t_data = np.arange(0, total_time, 1e-5)[::100] # this comes from allen cahn calculation and steps per saved is 100
t_data_sliced = t_data[::20] # reduce t every 30 points, only about 30 t elements left
u_sorted_files_sliced = u_sorted_files[::20] # u result sliced along with t
print(t_data_sliced)
print(u_sorted_files_sliced)

# prepare data for training
x_mesh, y_mesh= np.meshgrid(x_data, y_data)
x_flatten = x_mesh.flatten()
y_flatten = y_mesh.flatten()
all_x = []
all_y = []
all_t = []
all_u = []
for i in range(len(t_data_sliced)): # change multi demension data into 1d
    u_tmp = np.loadtxt(u_sorted_files_sliced[i]).reshape((256, -1))
    u_tmp_flatten = u_tmp.flatten()
    all_u.extend(u_tmp_flatten)
    all_x.extend(x_flatten)
    all_y.extend(y_flatten)
    t_flatten = np.full_like(u_tmp_flatten, t_data_sliced[i])  # fullfill shape of u_tmp_flatten using t
    all_t.extend(t_flatten)
input_data = np.hstack((np.array(all_x)[:, None], np.array(all_y)[:, None], np.array(all_t)[:, None])) # change data into n*3 matrix
target_data = all_u
# print(input_data, target_data)
input_tensor = torch.tensor(input_data, dtype=torch.float32, requires_grad=True) # change data type to tensor
target_tensor = torch.tensor(target_data, dtype=torch.float32).unsqueeze(1)

# create internal dataset using hypercube sampling, using to calculate pde loss
upper_point = np.array([1., 1., total_time]) 
lower_point = np.array([-1., -1., 0.])
x_train_f = lower_point + (upper_point - lower_point)*lhs(3, int(0.005*len(all_u)))
x_train_f = np.vstack([x_train_f, input_data[::10]])


# generate boundary points, due to periodical boundary condition, generate two pairs of symmetric data points 
print(x_data.shape, y_data.shape, t_data_sliced.shape)
random_choice_num = int(0.75*len(x_data))
print(random_choice_num)
idx_bound = np.random.choice(x_data.shape[0], random_choice_num, replace=False)
idy_bound = np.random.choice(y_data.shape[0], random_choice_num, replace=False)
idt_bound = np.random.choice(t_data_sliced.shape[0], int(0.75*(t_data_sliced.shape[0])), replace=False)
initial_xy_x_mesh, initial_xy_y_mesh = np.meshgrid(x_data, y_data)
initial_xy_x_mesh_flatten = initial_xy_x_mesh.flatten()[:, None]
initial_xy_y_mesh_flatten = initial_xy_y_mesh.flatten()[:, None]
initial_xy = np.hstack((initial_xy_x_mesh_flatten, initial_xy_y_mesh_flatten, np.full_like(initial_xy_y_mesh_flatten, 0))) # initial condition data
initial_u = np.loadtxt(u_sorted_files_sliced[0]).reshape((-1, 256)).flatten()         

# following are bound condition: bound_xt_lower: y=-1, -1<=x<=1, 0<t<1, bound_xt_upper: y=1, -1<=x<=1, 0<t<1, bound_yt_lower, -1<=y<=1, x=-1, 0<t<1 ...
bound_xt_x_mesh, bound_xt_t_mesh = np.meshgrid(x_data[idx_bound], t_data_sliced[idt_bound])
bound_xt_x_mesh_flatten = bound_xt_x_mesh.flatten()[:, None]
bound_xt_t_mesh_flatten = bound_xt_t_mesh.flatten()[:, None]
bound_xt_lower = np.hstack((bound_xt_x_mesh_flatten, np.full_like(bound_xt_x_mesh_flatten, -1), bound_xt_t_mesh_flatten))
bound_xt_upper = np.hstack((bound_xt_x_mesh_flatten, np.full_like(bound_xt_x_mesh_flatten, 1), bound_xt_t_mesh_flatten))

bound_yt_y_mesh, bound_yt_t_mesh = np.meshgrid(y_data[idy_bound], t_data_sliced[idt_bound])
bound_yt_y_mesh_flatten = bound_yt_y_mesh.flatten()[:, None]
bound_yt_t_mesh_flatten = bound_yt_t_mesh.flatten()[:, None]
bound_yt_lower = np.hstack((np.full_like(bound_yt_y_mesh_flatten, -1), bound_yt_y_mesh_flatten, bound_yt_t_mesh_flatten))
bound_yt_upper = np.hstack((np.full_like(bound_yt_t_mesh_flatten, 1), bound_yt_y_mesh_flatten, bound_yt_t_mesh_flatten))

# change data into tensor form
x_train_f = torch.from_numpy(x_train_f).float()
initial_xy = torch.from_numpy(initial_xy).float().to(device)
initial_u = torch.from_numpy(initial_u).float().to(device)
bound_xt_lower = torch.from_numpy(bound_xt_lower).float().to(device)
bound_xt_upper = torch.from_numpy(bound_xt_upper).float().to(device)
bound_yt_lower = torch.from_numpy(bound_yt_lower).float().to(device)
bound_yt_upper = torch.from_numpy(bound_yt_upper).float().to(device)
print(x_train_f.shape)
print(initial_xy.shape)
print(bound_xt_lower.shape)
print(bound_yt_lower.shape)

# create data loader for epoch iteration
train_batch_size = 10000
test_batch_size = 40000
PDE_dataset = PDEdataset(x_train_f)
PDE_Dataloader = DataLoader(PDE_dataset, batch_size=train_batch_size, num_workers=8, shuffle=True)
Test_dataset = Testdataset(input_tensor, target_tensor)
Test_Dataloader = DataLoader(Test_dataset, batch_size=test_batch_size, num_workers=8, shuffle=True)


# define neural net, loss and para updata formula set
layers = [3, 100, 100, 100, 100, 100, 100, 100, 100, 1] # enough depth neural network
pinnsnet = PINNs(layers=layers).to(device)
start_epoch = pinnsnet.load_checkpoint('./ac_2d_model_para_1/checkpoint_200.pth')
optimizer_adam = optim.Adam(pinnsnet.parameters(), lr=0.0001)
switch_epoch = 10000
criterion = nn.MSELoss()

# start iteration loop
num_epoches = 20000
# best_val_loss = float('inf')
test_interval = 20 # test mode accuracy every 5 epoches
save_interval = 50
save_dir = './ac_2d_model_para_1'
print('Iteration begins!')
for epoch in range(num_epoches):
    epoch = epoch + start_epoch
    running_loss = 0.0
    for _, x_train in enumerate(PDE_Dataloader):
        x_train = x_train.to(device)
        if epoch<3000:
            optimizer_adam.zero_grad()
            loss, loss_ic, loss_bc = pinnsnet.loss(x_train, initial_xy, initial_u, bound_xt_lower, bound_xt_upper, bound_yt_lower, bound_yt_upper)
            loss.backward()
            optimizer_adam.step()
        else:
            continue
        # print('start to test!')
    if (epoch+1) % 10 == 0:
        print(f'Present training epoch:{epoch+1}, loss is {loss:.10f}')
    if epoch % test_interval == 0:
        error = 0.
        batch_num = 0
        with torch.no_grad():
            for idx1, (input, target) in enumerate(Test_Dataloader):
                input, target = input.to(device), target.to(device)
                # print(input.shape, target.shape)
                error += pinnsnet.test(input, target)
                batch_num += 1
            average_error = error/batch_num
        print(loss, loss_ic, loss_bc, average_error)
        # print('start to save model!')
    if (epoch + 1) % save_interval == 0:
        with torch.no_grad():
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': pinnsnet.state_dict(),
                # 'optimizer': optimizer_lbfgs.state_dict(),
            }
            filename = f'{save_dir}/checkpoint_{epoch+1}.pth'
            torch.save(checkpoint, filename)
        print(f'Model saved at epoch {epoch+1}: {filename}')

# end training
end_time = time.time()
print(f'All the calculation has been done, total cost time is: {end_time - start_time:.4f} s')