import torch
import numpy as np

import RLFramework as rl
from gymnasiumPIDEnv import GymnasiumEnvironment
from InvPendulumPIDNets import *


env = GymnasiumEnvironment("InvertedPendulum-v4", render_mode='human')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy = InvPendulumPolicyNet(
    exploration=rl.exp.Gaussian()
).to(device=device)

value = InvPendulumValueNet(
    use_target=True,
    tau=0.01
).to(device=device)

agent = rl.Agent(policy=policy)

trainer = rl.Trainer(
    agent=agent,
    env=env,
    optimizers=[
        rl.optim.TargetValueOptim(lr=1e-3, epoch=30, batch_size=64, gamma=0.99, level=10),
        rl.optim.ClippedSurrogatePolicyOptim(lr=3e-4, epoch=30, batch_size=64, gamma=0.99, epsilon=0.2,
                                    lamda=0.95, entropy_weight=0.005, use_target_v=False, random_sample=False)
    ],
    logger=rl.utils.Logger(realtime_plot=True, rewards={"reward_sum": "env.episode_reward"}, window_size=800),
    pi=policy,
    v=value,
    memory=rl.traj.VolatileMemory()
)

trainer.load("./RL/saved/InvPendulumPPO_PID", version=1)

trainer.add_interval(trainer.train, episode=4, step=2000)
trainer.add_interval(value.update_target_network, step=1)


def random_position():
    pos = np.tanh(np.random.randn()) * 0.5

    env.env.unwrapped.set_state(np.array([pos,0]), np.array([0,0]))


def random_action():
    trainer.force_action(np.array([np.tanh(np.random.randn() / 3) * 3]))


# trainer.add_interval(random_action, step=200)
# trainer.add_interval(random_position, step=500)

trainer.run(test_mode=True)

env.env.close()

# trainer.save("./RL/saved/InvPendulumPPO_PID", version=1)
# policy.rnn.rnn_cell.logger.save("./RL/data/v5_2/")
# policy.logger.save("./RL/data/v8/")
