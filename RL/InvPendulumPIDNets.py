import numpy as np
import torch
import torch.nn as nn
from RLFramework.net import *
from tensor_logger import TensorLogger


class InvPendulumPolicyNet(PolicyNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 24)
        )

        # self.logger = TensorLogger("./RL/data/mlp_v0/", slots=["state", "first", "second", "output"])

        # self.rnn = nn.RNN(input_size=4, hidden_size=128, num_layers=3)
        # self.output = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Linear(128, 2)
        # )

    def forward(self, x):
        y = self.model(x)
        return y


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

