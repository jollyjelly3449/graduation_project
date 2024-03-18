import numpy as np
import matplotlib.pyplot as plt
from pca import PCA
from plot import plot


first = np.load("../RL/test_data/first.npy")
second = np.load("../RL/test_data/second.npy")
last = np.load("../RL/test_data/last.npy")
state = np.load("../RL/test_data/state.npy")

print(first.shape, second.shape, last.shape, state.shape)

#pca = PCA(n_components=8)

components, eigenvalues = PCA(second.reshape(-1,256), n_components=8)

print(components.shape, eigenvalues.shape)

plt.bar(np.arange(8) + 1, eigenvalues.real)
plt.show()

second_elements = second.reshape(-1, 256) @ components.T

plot(np.arange(1000), state[:1000, 0, 0], second_elements[:1000, 1], ylim=(-0.1, 0.2))