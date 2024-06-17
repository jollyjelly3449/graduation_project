import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from InvPendulumNCPNets import InvPendulumPolicyNet
from SetCustomConnection import *


net = InvPendulumPolicyNet()
SetCustomConnectionFromText(net.rnn, "./sparsity.txt")
MatchWiring(net.rnn)

wiring = net.rnn.wiring

sns.set_style("white")
plt.figure(figsize=(6, 4))
legend_handles = wiring.draw_graph(draw_labels=True, neuron_colors={"command": "tab:cyan"})
plt.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(1, 1))
sns.despine(left=True, bottom=True)
plt.tight_layout()
plt.show()