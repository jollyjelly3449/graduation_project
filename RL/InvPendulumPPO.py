import torch
import numpy as np

from RLFramework import *
from gymnasiumEnv import GymnasiumEnvironment
from InvPendulumNCPNets import *

env = GymnasiumEnvironment("InvertedPendulum-v4", render_mode="human")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy = InvPendulumPolicyNet(
    exploration=Gaussian()
).to(device=device)

value = InvPendulumValueNet(
    use_target=True,
    tau=0.01
).to(device=device)

agent = RLAgent(policy=policy)

trainer = RLTrainer(
    agent=agent,
    env=env,
    optimizers=[
        TargetValueOptim(lr=1e-3, epoch=30, batch_size=64, gamma=0.99, level=10),
        ClippedSurrogatePolicyOptim(lr=3e-4, epoch=30, gamma=0.99, epsilon=0.2,
                                    lamda=0.95, entropy_weight=0.005, use_target_v=False, random_sample=False)
    ],
    logger=Logger(realtime_plot=True, rewards={"reward_sum": "env.episode_reward",
                                               "decay_reward": "env.discount_reward"}, window_size=800),
    pi=policy,
    v=value,
    memory=VolatileMemory()
)

trainer.load("./RL/saved/InvPendulumPPO_NCP", version="6")

trainer.add_interval(trainer.train, episode=4, step=200)
trainer.add_interval(value.update_target_network, step=1)


def random_action():
    trainer.force_action(np.array([np.tanh(np.random.randn() / 3) * 3]))


trainer.add_interval(random_action, step=50)

trainer.run(test_mode=True)

# trainer.save("./RL/saved/InvPendulumPPO_NCP", version=6)
# policy.rnn.rnn_cell.logger.save("./RL/data/v6/")
