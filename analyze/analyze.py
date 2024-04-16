import numpy as np
import matplotlib.pyplot as plt
from pca import PCA
from plot import plot


base_addr = "../RL/data/v8"

# state = np.load(f"{base_addr}/state.npy")
# first = np.load(f"{base_addr}/sensory.npy")
# second = np.load(f"{base_addr}/command.npy")
# last = np.load(f"{base_addr}/output.npy")
state = np.load(f"{base_addr}/state.npy")
first = np.load(f"{base_addr}/first.npy")
# second = np.load(f"{base_addr}/second.npy")
last = np.load(f"{base_addr}/output.npy")

print(state.shape, first.shape, last.shape)
# print(state.shape, first.shape, second.shape, last.shape)

# components, eigenvalues = PCA(first[:5000].reshape(-1, 3), n_components=3)

# print(components.shape, eigenvalues.shape)

# plt.bar(np.arange(7) + 1, eigenvalues.real)
# plt.show()

# second_elements = first.reshape(-1, 7) @ components.T

plot(np.arange(100,5000), state[100:5000, 0, 0, 1], first[100:5000, 0, 0, 0], ylim=(-0.6, 0.6))
plot(np.arange(100,5000), state[100:5000, 0, 0, 1], first[100:5000, 0, 0, 1], ylim=(-0.6, 0.6))
plot(np.arange(100,5000), state[100:5000, 0, 0, 1], first[100:5000, 0, 0, 2], ylim=(-0.6, 0.6))
# plot(np.arange(100,5000), state[100:5000, 0, 0, 1], first[100:5000, 0, 0, 3], ylim=(-0.6, 0.6))
# plot(np.arange(100,5000), state[100:5000, 0, 0, 1], first[100:5000, 0, 0, 4], ylim=(-0.6, 0.6))
# plot(np.arange(100,5000), state[100:5000, 0, 0, 1], first[100:5000, 0, 0, 5], ylim=(-0.6, 0.6))
# plot(np.arange(100,5000), state[100:5000, 0, 0, 1], first[100:5000, 0, 0, 6], ylim=(-0.6, 0.6))

# plt.scatter(state[:5000, 0, 0], second[:5000, 0, 0])
# plt.scatter(state[:5000, 0, 3], second[:5000, 0, 1])
# plt.scatter(state[:5000, 0, 3], second[:5000, 0, 2])
# plt.scatter(state[:5000, 0, 2], second[:5000, 0, 3])
# plt.scatter(state[:5000, 0, 3], second[:5000, 0, 4])
# plt.scatter(state[:5000, 0, 2], second[:5000, 0, 5])
# plt.scatter(state[:5000, 0, 3], second[:5000, 0, 6])
# plt.scatter(state[:5000, 0, 0], second[:5000, 0, 7])
# plt.scatter(state[:5000, 0, 0], second[:5000, 0, 8])
# plt.scatter(state[:5000, 0, 0], second[:5000, 0, 9])
# plt.scatter(state[:5000, 0, 0], second[:5000, 0, 10])


plt.show()

# D : 0, 2, 5
# P : 1, 3, 6