import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


base_addr = "../RL/data/v6"

state = np.load(f"{base_addr}/state.npy")
# first = np.load(f"{base_addr}/sensory.npy")
# second = np.load(f"{base_addr}/command.npy")
last = np.load(f"{base_addr}/output.npy")

# pos = state[:5000, 0, 0]
# angle = state[:5000, 0, 1]

dt = 0.02

p = state[1:1000, 0, :2].copy()
i = state[1:1000, 0, :2].copy()
d = state[1:1000, 0, 2:].copy()

# p = angle[1:].copy()
# i = angle[1:].copy()
# d = (angle[1:].copy() - angle[:-1])
#
# pp = pos[1:].copy() / 1000
# pi = pos[1:].copy() / 1000
# pd = (pos[1:].copy() - pos[:-1]) / 1000

sum = 0
for index in range(i.shape[0]):
    sum = sum * 0.9 + p[index] * dt
    i[index] = sum

# plt.plot(p)
# plt.plot(i)
# plt.plot(d)

true = last[1:1000, 0, 0]
plt.plot(true)

# k_p = np.zeros(2)
# k_i = np.zeros(2)
# k_d = np.zeros(2)

k_p = np.array([0.10915335, 1.08076929])
k_i = np.array([-0.00489503,  0.01869387])
k_d = np.array([0.1660981,  0.28244076])

x = np.concatenate([p, d], axis=1)
y = true

w = np.linalg.solve(x.T@x, x.T@y)

print(w)

# k_pp = 0
# k_pi = 0
# k_pd = 0
# [0.29978308 2.21818968] [-1.20167279e+00  7.12130646e-04] [0.34494459 0.38541534]
# [0.30257425 2.2337036  0.3474976  0.3868848 ]
# for _ in range(1000000):
#     pred = p @ k_p + d @  k_d
#
#     loss = np.mean(0.5 * (pred - true) ** 2)
#
#     grad_p = p.T @ (pred-true) * loss
#     grad_i = i.T @ (pred-true) * loss
#     grad_d = d.T @ (pred-true) * loss
#
#     print(k_p, k_i, k_d)
#
#     lr = 1e-1
#
#     k_p -= grad_p * lr
#     k_i -= grad_i * lr
#     k_d -= grad_d * lr
#
#     print(loss)

# plt.plot(pred)
plt.show()

