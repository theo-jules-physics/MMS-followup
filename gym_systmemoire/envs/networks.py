import torch
from torch import nn
        
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
        
def simple_mlp(shape, activation=nn.ReLU(), bias=True):
    # Function that produces a FC network with specified sizes (from input to output) and 
    # non linear activation.
    model = nn.Sequential()
    k = 0
    model.add_module(f'fc_{k+1}', nn.Linear(shape[0], shape[1], bias=bias))
    for j in range(len(shape)-2):
        model.add_module(f'relu_{k+1}', activation)
        k += 1
        model.add_module(f'fc_{k+1}', nn.Linear(shape[j+1], shape[j+2], bias=bias))
    model.apply(init_weights)
    return model

class Policy(nn.Module):
    def __init__(self, shape, threshold):
        super(Policy, self).__init__()
        self.model = simple_mlp(shape)
        self.threshold = threshold
        self.tanh_act = nn.Tanh()
        
    def forward(self, x):
        return self.tanh_act(self.model(x)) * self.threshold

class Qfunc(nn.Module):
    def __init__(self, shape):
        super(Qfunc, self).__init__()
        self.model = simple_mlp(shape)

    def forward(self, x):
        return self.model(x)