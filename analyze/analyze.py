import numpy as np
import matplotlib.pyplot as plt
from pca import PCA


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

