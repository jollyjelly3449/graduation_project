import torch
import numpy as np

import RLFramework as rl
from gymnasiumEnv import GymnasiumEnvironment
from PendulumNCPNets import *

env = GymnasiumEnvironment("Pendulum-v1", render_mode="human")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy = PendulumPolicyNet(
    exploration=rl.exp.Gaussian()
).to(device=device)

value = PendulumValueNet(
    use_target=True,
    tau=0.01
).to(device=device)

agent = rl.Agent(policy=policy)

trainer = rl.Trainer(
    agent=agent,
    env=env,
    optimizers=[
        rl.optim.TargetValueOptim(lr=1e-3, epoch=10, batch_size=64, gamma=0.99, level=10),
        rl.optim.ClippedSurrogatePolicyOptim(lr=1e-3, epoch=10, gamma=0.99, epsilon=0.2,
                                    lamda=0.95, entropy_weight=0.005, use_target_v=False, random_sample=False)
    ],
    logger=rl.utils.Logger(realtime_plot=True, rewards={"reward_sum": "env.episode_reward",
                                               "decay_reward": "env.discount_reward"}, window_size=800),
    pi=policy,
    v=value,
    memory=rl.traj.VolatileMemory()
)

trainer.load("./RL/saved/PendulumPPO_NCP", version=0)

trainer.add_interval(trainer.train, episode=4, step=200)
trainer.add_interval(value.update_target_network, step=1)


def random_action():
    trainer.force_action(np.array([np.tanh(np.random.randn() / 3) * 3]))


# trainer.add_interval(random_action, step=50)

trainer.run(test_mode=True)

# trainer.save("./RL/saved/PendulumPPO_NCP", version=0)
# policy.rnn.rnn_cell.logger.save("./RL/data/pendulum_v0/")
# policy.logger.save()
