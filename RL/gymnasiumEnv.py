import numpy as np
import gymnasium as gym
import RLFramework as rl
from gymnasium.wrappers import RecordVideo


class GymnasiumEnvironment(rl.Environment):
    def __init__(self, env_name: str, gamma=0.99, seed=None, **kwargs):
        self.env = gym.make(env_name, **kwargs)
        observation_space = self.convert_space(self.env.observation_space)
        action_space = self.convert_space(self.env.action_space)

        super().__init__(observation_space=observation_space, action_space=action_space)

        self.seed = seed
        self.gym_reward = 0

        self.discount_factor = gamma

        self.episode_reward = 0
        self.discount_reward = 0

        self.reset_params()

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
        observation, reward, terminated, truncated, info = self.env.step(action)

        self.gym_reward = reward
        end = terminated or truncated

        return observation, end

    def reward(self, state, action, next_state):
        # print(self.gym_reward)
        self.episode_reward += self.gym_reward - 2 * abs(state[0])
        self.discount_reward = self.discount_reward * self.discount_factor + self.gym_reward

        # print(state, self.gym_reward - state[0] ** 2)

        return self.gym_reward - 2 * abs(state[0])

    def reset_params(self):
        self.gym_reward = 0
        self.episode_reward = 0
        self.discount_reward = 0

        observation, info = self.env.reset(seed=self.seed)
        self.init_state(observation)
