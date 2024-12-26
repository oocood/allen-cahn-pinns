import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

layers = [3, 100, 100, 100, 100, 100, 100, 100, 100, 1] # enough depth neural network

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
    
   
# create object from network
pinnsnet = PINNs(layers)
optimizer = optim.Adam(pinnsnet.parameters(), lr=0.001)

# load pre-trained model
checkpoint_path = './ac_2d_model_para_5/checkpoint_2600.pth'
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

# load model parameters
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint  # 如果检查点只包含 state_dict
pinnsnet.load_state_dict(state_dict)

# set model to evaluation mode
pinnsnet.eval()

# choose generate mesh
x_length = 2
y_length = x_length
x_nodenum = 256
y_nodenum = x_nodenum
x = x_length/x_nodenum*np.arange(-x_nodenum/2, x_nodenum/2, 1)
y = y_length/y_nodenum*np.arange(-y_nodenum/2, y_nodenum/2, 1)
x_mesh, y_mesh = np.meshgrid(x, y)
x_flatten = x_mesh.flatten()
y_flatten = y_mesh.flatten()
t = 0.01
t_flatten = np.full_like(x_flatten, t)
input_data = np.column_stack((x_flatten, y_flatten, t_flatten))
input_data_tensor = torch.tensor(input_data, dtype=torch.float32, requires_grad=False)
with torch.no_grad():
    u = pinnsnet(input_data_tensor)
u_mesh = u.numpy()
u_mesh = u_mesh.reshape(x_mesh.shape)
print(np.max(u_mesh))
plt.imshow(u_mesh)
plt.colorbar()
plt.show()
# plt.savefig('./ac_2d_model_para_5/pinns_result_2d_allen_cahn_t_0.5.png', dpi=300)
print('All the calculation has been done!')