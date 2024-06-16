import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from InvPendulumNCPNets import InvPendulumPolicyNet

torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# state_dict = torch.load("./saved/InvPendulumPPO_NCP_pi_5.pth")

net = InvPendulumPolicyNet().to(device=device)
# net.load_state_dict(state_dict)

test_data = torch.rand(5,1,4).to(device)

print(net.rnn(test_data))

o0, h0 = net.rnn(test_data[0])
h0[:11] = 0
o1, h1 = net.rnn(test_data[1], h0)
h1[:11] = 0
o2, h2 = net.rnn(test_data[2], h1)
h2[:11] = 0
o3, h3 = net.rnn(test_data[3], h2)
h3[:11] = 0
o4, h4 = net.rnn(test_data[4], h3)

print(o0, o1, o2, o3, o4)
print(h0, h1, h2, h3, h4)

# (tensor([[[ 0.2098, -0.5889]],
#
#         [[ 0.4980, -0.8826]],
#
#         [[ 0.5771, -0.9335]],
#
#         [[ 0.5580, -0.9241]],
#
#         [[ 0.5128, -0.9037]]], device='cuda:0', grad_fn=<StackBackward0>), tensor([[-0.3797,  0.2220,  0.5318, -0.1151,  0.0429, -0.5964,  0.5433, -0.9198,
#          -0.5900,  0.4832,  0.9019, -0.0752,  0.1489, -0.0430, -0.0980,  0.7112,
#           0.5917, -0.2400,  0.5128, -0.9037]], device='cuda:0',
#        grad_fn=<CatBackward0>))