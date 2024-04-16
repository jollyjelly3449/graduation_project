import torch
import RLFramework as rl


class PidGaussian(rl.exp.Gaussian):
    def __init__(self, state_dim=4, mask=torch.ones(12), action_space: rl.space.Continuous = None):
        self.state_dim = state_dim
        self.mask = mask

        super().__init__(action_space=action_space)

    def get_policy_shape(self, action_space: rl.space.Continuous):
        # p, i, d mean for all states, p, i, d std for all states, p, i, d values for all states.
        return (3, self.state_dim * 3, *action_space.shape)

    def explore(self, x, numpy=False):
        coeff, logprob = super().explore(x)
        action = torch.sum(coeff * x[:, 2, :] * self.mask.to(x))

        return action, logprob

    def greedy(self, x, numpy=False):
        coeff = x[:, 0, :]
        action = torch.sum(coeff * x[:, 2, :] * self.mask.to(x))

        return action, 0
