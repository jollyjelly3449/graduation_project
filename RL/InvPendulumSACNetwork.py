from RLFramework.Network import *
from tensor_logger import TensorLogger


class InvPendulumPolicyNetwork(Network):
    def __init__(self):
        super().__init__()
        self.first = nn.Sequential(
            nn.BatchNorm1d(4),
            nn.Linear(4, 256)
        )

        self.second = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(256, 256),
        )

        self.last = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(256, 2)
        )

        self.logger = TensorLogger("./RL/data/", slots=["state", "first", "second", "last"])

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0)

        self.apply(init_weights)

    def sample_action(self, policy, test=False):
        if len(policy.shape) == 1:
            unbatched = True
            policy = policy.reshape((1, -1))
        else:
            unbatched = False

        if not test:
            dist = torch.distributions.normal.Normal(policy[:, :1], torch.exp(policy[:, 1:]))
            action = dist.rsample()

            logprobs = dist.log_prob(action)
            action = torch.tanh(action)

            logprobs -= torch.log(3 * (1 - action.pow(2)) + 1e-6)
            logprobs = torch.sum(logprobs, dim=1, keepdim=True)
            action = action * 3
        else:
            action = policy[:, :1]
            action = torch.tanh(action)

            action = action * 3

            logprobs = None



        #print(f"pocliy : {policy}")
        #print(f"action : {action}, logprob : {logprobs}")

        if unbatched:
            action = action.reshape(-1)

        return action, logprobs

    def forward(self, x):
        if len(x.shape) == 1:
            unbatched = True
            x = x.reshape((1, -1))
        else:
            unbatched = False

        if unbatched:
            self.logger.append(state=x)

        x = self.first(x)
        if unbatched:
            self.logger.append(first=x)

        x = self.second(x)
        if unbatched:
            self.logger.append(second=x)

        x = self.last(x)
        if unbatched:
            self.logger.append(last=x)

        if unbatched:
            x = x.reshape(-1)

        return x

class InvPendulumNCPPolicyNetwork(Network):
    def __init__(self):
        super().__init__()

        self.head = nn.Sequential(
            nn.BatchNorm1d(4),
            nn.Linear(4, 256),
            nn.ReLU()
        )
        self.model = nn.RNN(input_size=256, hidden_size=128)
        self.tail = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128, 2)
        )

        self.logger = Logger("./RL/data/", slots=["state", "first", "second", "last"])

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0)

        self.apply(init_weights)

    def sample_action(self, policy, test=False):
        if len(policy.shape) == 1:
            unbatched = True
            policy = policy.reshape((1, -1))
        else:
            unbatched = False

        policy, h0 = policy[:, :2], policy[:, 2:]

        if not test:
            dist = torch.distributions.normal.Normal(policy[:, :1], torch.exp(policy[:, 1:]))
            action = dist.rsample()

            logprobs = dist.log_prob(action)
            action = torch.tanh(action)

            logprobs -= torch.log(3 * (1 - action.pow(2)) + 1e-6)
            logprobs = torch.sum(logprobs, dim=1, keepdim=True)
            action = action * 3
        else:
            action = policy[:, :1]
            action = torch.tanh(action)

            action = action * 3

            logprobs = None

        #print(f"pocliy : {policy}")
        #print(f"action : {action}, logprob : {logprobs}")

        action = torch.concatenate([action, h0], dim=1)

        if unbatched:
            action = action.reshape(-1)

        return action, logprobs

    def forward(self, x):
        if len(x.shape) == 1:
            unbatched = True
            x = x.reshape((1, -1))
        else:
            unbatched = False

        x, h0 = x[:, :4], x[:, 4:]

        if unbatched:
            self.logger.append(state=x)

        print(x.shape)

        x = self.head(x)
        x = x.reshape(1, x.shape[0], -1)
        h0 = h0.reshape(1, h0.shape[0], -1).contiguous()
        x, hn = self.model(x, h0)
        x = x.reshape(x.shape[1], -1)
        hn = hn.reshape(hn.shape[1], -1)
        x = self.tail(x)

        x = torch.concat([x, hn], dim=1)

        if unbatched:
            x = x.reshape(-1)

        return x

class InvPendulumQNetwork(Network):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d(5),
            nn.Linear(5, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0)

        self.apply(init_weights)

    def forward(self, x):
        if len(x.shape) == 1:
            unbatched = True
            x = x.reshape((1, -1))
        else:
            unbatched = False

        x = torch.concat([x[:, :4], x[:, 128+4:128+5]], dim=1)

        x = self.model(x)

        if unbatched:
            x = x.reshape(-1)

        return x


class InvPendulumValueNetwork(Network):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d(4),
            nn.Linear(4, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0)

        self.apply(init_weights)

    def forward(self, x):
        if len(x.shape) == 1:
            unbatched = True
            x = x.reshape((1, -1))
        else:
            unbatched = False

        x, h0 = x[:, :4], x[:, 4:]

        x = self.model(x)

        if unbatched:
            x = x.reshape(-1)

        return x
