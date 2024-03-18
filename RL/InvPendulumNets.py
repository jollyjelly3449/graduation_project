import numpy as np
import torch
import torch.nn as nn
from RLFramework.net import *


class InvPendulumPolicyNet(PolicyNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, hx=torch.zeros((3, 1, 128)), init_state=np.array([1]), **kwargs)

        self.model = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

        self.rnn = nn.RNN(input_size=4, hidden_size=128, num_layers=3)
        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128, 2)
        )

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
                    hx = torch.zeros((3, 1, 128)).to(x)

                    _x, _ = self.rnn(_x, hx)
                    _x = _x.reshape((_x.shape[0], -1))
                    outputs.append(_x)

            x = torch.cat(outputs)
            x = self.output(x)

        else:
            self.init_state = self.init_state * 0

            x = x.reshape((1, 1, -1))
            hx = self.hx.to(x)
            x, hx = self.rnn(x, hx)
            self.hx = hx
            x = x.reshape(1, -1)
            x = self.output(x)

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
