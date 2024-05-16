import numpy as np
import torch
import torch.nn as nn
from RLFramework.net import *
from tensor_logger import TensorLogger


class InvPendulumPolicyNet(PolicyNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, sum=np.array([0,0]), **kwargs)

        self.model = nn.Sequential(
            nn.Linear(4, 5),
            nn.ReLU(),
            nn.Linear(5,3),
            nn.ReLU(),
            nn.Linear(3,2)
        )

        self.logger = TensorLogger("./RL/data/mlp_v0/", slots=["state", "output"])

        # self.rnn = nn.RNN(input_size=4, hidden_size=128, num_layers=3)
        # self.output = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Linear(128, 2)
        # )

    def forward(self, x):
        print("pid!")
        self.logger.append(state=x)
        x = self.model(x)
        # linear1, relu1, linear2, relu2, linear3 = self.model.children()
        self.sum = self.sum * 0.9 + x[0, :2].detach().cpu().numpy() * 0.02
        # x = self.model(x)
        # x = (x * torch.tensor([0.30257425, 2.2337036,  0.3474976,  0.3868848]).to(x)).sum()
        # x = (x * torch.tensor([-0.567695, 10.017549,  4.235533, 2.7435687]).to(x)).sum()
        # x = (x * torch.tensor([0.7174523, 2.5525918, 0.04922304, 0.26647952]).to(x)).sum() + self.sum.T @ np.array([-3.6145527, 0.90431637])
        # x = torch.tensor([x,x]).reshape(2)
        # self.logger.append(first=x)
        #
        # x = linear2(relu1(x))
        # self.logger.append(second=x)
        #
        # x = linear3(relu2(x))
        self.logger.append(output=x)

        # if x.shape[0] != 1:
        #     outputs = []
        #     last_index = 0
        #
        #     for i in range(1, x.shape[0]):
        #         if i == x.shape[0] - 1:
        #             i += 1
        #
        #         if i == x.shape[0] or self.init_state[i]:
        #             _x = x[last_index:i].reshape((i - last_index, 1, -1))
        #             last_index = i
        #             hx = torch.zeros((3, 1, 128)).to(x)
        #
        #             _x, _ = self.rnn(_x, hx)
        #             _x = _x.reshape((_x.shape[0], -1))
        #             outputs.append(_x)
        #
        #     x = torch.cat(outputs)
        #     x = self.output(x)
        #
        # else:
        #     self.init_state = self.init_state * 0
        #
        #     x = x.reshape((1, 1, -1))
        #     hx = self.hx.to(x)
        #     x, hx = self.rnn(x, hx)
        #     self.hx = hx
        #     x = x.reshape(1, -1)
        #     x = self.output(x)

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

