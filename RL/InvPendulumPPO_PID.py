import torch
import numpy as np

import RLFramework as rl
from gymnasiumEnv import GymnasiumEnvironment
from InvPendulumNets import *

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
                                             lamda=0.95, entropy_weight=0.005, use_target_v=True, random_sample=False)
    ],
    logger=rl.utils.Logger(realtime_plot=False,
                           rewards={"reward_sum": "env.episode_reward", "decay_reward": "env.discount_reward"},
                           window_size=800),
    pi=policy,
    v=value,
    memory=rl.traj.VolatileMemory()
)

trainer.load("./saved/InvPendulumPPO_wEnergy", version=1)

trainer.add_interval(trainer.train, episode=10, step=5000)
trainer.add_interval(value.update_target_network, step=1)

# rand_seed = np.array([-1.27929788, -0.01545778, -0.90460136, 1.66914875, 0.48203259,
#                       1.21632999, -0.71795415, -1.70283474, -0.86681215, 1.29019525,
#                       0.21657702, -0.80695598, 0.47239043, 0.56756948, 1.11533704,
#                       0.42035474, 0.20958234, -0.36796657, 0.49331923, -1.69551741,
#                       -0.46741346, -0.63564577, 0.5440203, -1.53089434, -0.17548649,
#                       -0.77096377, 0.84693949, 1.05275414, -0.53125902, -0.14560132,
#                       -1.28680704, 0.57232057, -1.19497653, -0.3248494, -0.52062441,
#                       -1.69564467, -0.74890727, -0.28669131, -0.69677672, -0.25049393,
#                       1.37985749, 0.84314428, 0.29748118, -1.24769623, -0.68179482,
#                       -0.37824065, 0.87129586, -0.07298444, 0.39774986, -0.57173387,
#                       2.17734823, -0.04733047, -0.7912065, 0.48530209, -1.29612143,
#                       -1.03005574, -0.64861522, 1.18571507, -0.62688723, 1.62584252,
#                       -1.6671286, 0.4556164, -0.17024315, -0.27623393])

index = 0


def random_position():
    global index
    pos = np.tanh(np.random.randn()) * 0.5
    index += 1

    env.env.unwrapped.set_state(np.array([pos, 0]), np.array([0, 0]))
    # env.init = True


def random_action():
    global index
    trainer.force_action(np.array([np.tanh(np.random.randn() / 3) * 3]))
    index += 1


# trainer.add_interval(random_action, step=250)
# trainer.add_interval(random_position, step=450)

trainer.run(test_mode=True)
# print("average reward:")
# print(sum(trainer.logger.plots["rewards"]["reward_sum"][-5:]) / 5)

env.env.close()

# trainer.save("./saved/InvPendulumPPO_wEnergy", version=1)
# policy.rnn.rnn_cell.logger.save("./RL/data/v5_2/")
# policy.logger.save("./data/e_v1_pid_fixed/")

# version 7 : reward penalty is action ** 2 * 0.9
# version 8 : reward penalty is energy / 100
# PID 12 : 0.5, -, 50
# PID 13 : 0.1, -, 50
