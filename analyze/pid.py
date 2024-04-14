import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


base_addr = "../RL/data/v5_1"

state = np.load(f"{base_addr}/state.npy")
first = np.load(f"{base_addr}/sensory.npy")
second = np.load(f"{base_addr}/command.npy")
last = np.load(f"{base_addr}/output.npy")

# pos = state[:5000, 0, 0]
# angle = state[:5000, 0, 1]

dt = 0.02

p = state[1:, 0].copy()
i = state[1:, 0].copy()
d = (state[1:, 0].copy() - state[:-1, 0]) / dt

# p = angle[1:].copy()
# i = angle[1:].copy()
# d = (angle[1:].copy() - angle[:-1])
#
# pp = pos[1:].copy() / 1000
# pi = pos[1:].copy() / 1000
# pd = (pos[1:].copy() - pos[:-1]) / 1000

sum = 0
for index in range(i.shape[0]):
    sum = sum + p[index] * dt
    i[index] = sum

# plt.plot(p)
# plt.plot(i)
# plt.plot(d)
plt.plot(last[1:, 0, 0] / 100)

true = last[1:, 0, 0] / 100

# k_p = np.zeros(4)
# k_i = np.zeros(4)
# k_d = np.zeros(4)

k_p = np.array([-5.96247708e-05,  1.10553364e-04, -4.69711884e-04,  1.05417399e-03])
k_i = np.array([-1.16411609e-05,  3.13050638e-06, -3.43430322e-05,  6.57691314e-05])
k_d = np.array([-0.00094579,  0.00209476,  0.00036493,  0.00023698])

# k_pp = 0
# k_pi = 0
# k_pd = 0

for _ in range(1000000):
    pred = p @ k_p + i @ k_i + d @  k_d

    loss = np.mean(0.5 * (pred - true) ** 2)

    grad_p = p.T @ (pred-true) * loss
    grad_i = i.T @ (pred-true) * loss
    grad_d = d.T @ (pred-true) * loss

    print(k_p, k_i, k_d)

    lr = 4e-3

    k_p -= grad_p * lr
    k_i -= grad_i * lr
    k_d -= grad_d * lr

    print(loss)

plt.plot(pred)
plt.show()

