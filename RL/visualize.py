import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from InvPendulumNCPNets import InvPendulumPolicyNet


net = InvPendulumPolicyNet()
net.load_state_dict(torch.load("./saved/InvPendulumPPO_NCP_pi_12.pth"))

wiring = net.rnn.wiring

sns.set_style("white")
plt.figure(figsize=(6, 4))
legend_handles = wiring.draw_graph(draw_labels=True, neuron_colors={"command": "tab:cyan"})
plt.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(1, 1))
sns.despine(left=True, bottom=True)
plt.tight_layout()
plt.show()