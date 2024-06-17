import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from InvPendulumNCPNets import InvPendulumPolicyNet

# state_dict = torch.load("./saved/InvPendulumPPO_NCP_pi_6.pth")
net = InvPendulumPolicyNet()
# net.load_state_dict(state_dict)

state_dict = net.state_dict()

wiring = net.rnn.wiring

print("point")

sparse_0 = state_dict["rnn.rnn_cell.layer_0.sparsity_mask"]
f1_0 = state_dict["rnn.rnn_cell.layer_0.ff1.weight"] * sparse_0
sparse_1 = state_dict["rnn.rnn_cell.layer_1.sparsity_mask"]
f1_1 = state_dict["rnn.rnn_cell.layer_1.ff1.weight"] * sparse_1
sparse_2 = state_dict["rnn.rnn_cell.layer_2.sparsity_mask"]
f1_2 = state_dict["rnn.rnn_cell.layer_2.ff1.weight"] * sparse_2


def plot_weight(data):
    if (isinstance(data, torch.Tensor)):
        data = data.cpu().detach().numpy()

    ax = plt.subplot(111)
    ax.grid(True)
    ax.pcolormesh(np.arange(data.shape[1]), np.arange(data.shape[0]), data)
    plt.show()


plot_weight(f1_0)
plot_weight(f1_1)
plot_weight(f1_2)

plot_weight(sparse_0)
plot_weight(sparse_1)
plot_weight(sparse_2)

print("end")
