import numpy as np
import gymnasium as gym
import RLFramework as rl
from gymnasium.wrappers import RecordVideo


class GymnasiumEnvironment(rl.Environment):
    def __init__(self, env_name: str, gamma=0.99, seed=None, **kwargs):
        self.env = gym.make(env_name, **kwargs)
        observation_space = self.convert_space(self.env.observation_space)
        action_space = rl.space.Continuous(
                            upper=np.array([0.02,0.02,0.02,0.02,
                                            0.001, 0.001, 0.001, 0.001,
                                            50,50,50,50]),
                            lower=-np.array([0.02,0.02,0.02,0.02,
                                            0.001, 0.001, 0.001, 0.001,
                                            50,50,50,50])
                        )

        super().__init__(observation_space=observation_space, action_space=action_space)

        self.seed = seed
        self.gym_reward = 0

        self.discount_factor = gamma

        self.episode_reward = 0
        self.discount_reward = 0

        self.last_state = np.zeros(4)
        self.state_sum = np.zeros(4)
        self.init = True

        self.mask = np.array([1, 1, 0, 0])

        self.reset_params()
        self.action = 0
        self.last_action = 0
        self.energy = 0

    def convert_space(self, gym_space):
        if isinstance(gym_space, gym.spaces.Discrete):
            return rl.space.Discrete(gym_space.n)
        elif isinstance(gym_space, gym.spaces.Box):
            return rl.space.Continuous(
                upper=np.array(gym_space.high),
                lower=np.array(gym_space.low)
            )
        else:
            raise NotImplementedError

    def update(self, state, action):
        print(action[[0, 1]], action[[4, 5]], action[[8, 9]])
        d = state - self.last_state
        i = self.state_sum + state

        i *= 0
        if self.init:
            self.init = False
            d *= 0

        action = np.tanh(np.sum((state * action[:4] + i * action[4:8] + d * action[8:]) * self.mask).reshape(1,)) * 3
        # kp = np.array([-0.3291951, -0.10543705])
        # kd = np.array([-0.27146307,  0.24464034])
        # action = np.array([np.tanh(np.dot(state[:2], kp) + np.dot(d[:2], kd))])

        print(action)

        self.action = action

        self.last_state = state
        self.state_sum += state

        observation, reward, terminated, truncated, info = self.env.step(action)

        self.gym_reward = reward
        end = terminated or truncated

        return observation, end

    def reward(self, state, action, next_state):
        # print(self.gym_reward)
        self.episode_reward += self.gym_reward - 2 * abs(state[0])
        self.discount_reward = self.discount_reward * self.discount_factor + self.gym_reward

        # print(state, self.gym_reward - state[0] ** 2)

        self.energy = self.energy * 0.9 + (self.action - self.last_action) ** 2
        self.last_action = self.action

        return self.gym_reward - 2 * abs(state[0]) - self.energy / 100

    def reset_params(self):
        self.gym_reward = 0
        self.episode_reward = 0
        self.discount_reward = 0

        self.action = 0
        self.last_action = 0
        self.energy = 0

        self.init = True

        self.last_state = np.zeros(4)
        self.state_sum = np.zeros(4)

        observation, info = self.env.reset(seed=self.seed)
        self.init_state(observation)