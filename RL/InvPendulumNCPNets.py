import numpy as np
import torch
import torch.nn as nn
from ncps.wirings import AutoNCP
from ncps.torch import CfC
from RLFramework.net import *


# version 0: CfC(4, 50), ReLU(), Linear(50,2), train freq 2000 step
# version 1: CfC(4, AutoNCP(16, 2)), train freq 200 step, 1 episode
# version 2: CfC(4, AutoNCP(8, 2)), train same
# version 2_1: network same as 2, reward changed

class InvPendulumPolicyNet(PolicyNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, hx=torch.zeros((1, 16)), init_state=np.array([1]), **kwargs)

        self.model = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        # version 0
        # self.rnn = CfC(input_size=4, units=50, batch_first=False) #nn.RNN(input_size=4, hidden_size=128, num_layers=3)
        # self.output = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Linear(50, 2)
        # )

        # version 1
        # wiring = AutoNCP(16, 2)

        # version 2
        wiring = AutoNCP(16, 2)
        self.rnn = CfC(input_size=4, units=wiring, batch_first=False)

    def forward(self, x):
        # x = self.model(x)
        if x.shape[0] != 1:
            outputs = []
            last_index = 0

            for i in range(1, x.shape[0]):
                if i == x.shape[0] - 1:
                    i += 1

                if i == x.shape[0] or self.init_state[i]:
                    _x = x[last_index:i].reshape((i - last_index, 1, -1))
                    last_index = i
                    hx = torch.zeros((1, 16)).to(x)

                    _x, _ = self.rnn(_x, hx)
                    _x = _x.reshape((_x.shape[0], -1))
                    outputs.append(_x)

            x = torch.cat(outputs)

        else:
            self.init_state = self.init_state * 0

            x = x.reshape((1, 1, -1))
            hx = self.hx.to(x)
            x, hx = self.rnn(x, hx)
            self.hx = hx
            x = x.reshape(1, -1)

        # x = self.output(x)

        return x


class InvPendulumValueNet(ValueNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = nn.Sequential(
            nn.Linear(4,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,1)
        )

    def forward(self, x):
        if len(x.shape) == 1:
            unbatched = True
            x = x.reshape((1, -1))
        else:
            unbatched = False

        x = self.model(x)

        if unbatched:
            x = x.reshape(-1)

        return x
