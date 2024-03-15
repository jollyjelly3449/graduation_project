import numpy as np

from sklearn.preprocessing import StandardScaler


def PCA(X, n_components=8):
    X=StandardScaler().fit_transform(X)
    cov_matrix = np.cov(X.T)
    print("Covariance matrix== /n", cov_matrix)

    eigen_values, eigen_vectors = np.linalg.eig(
        cov_matrix)  # each column of W is the Principal Component ie the orthogonal axis and ^ tells us how much we have to stretch along each axis
    print("Eigen values \n", eigen_values, "\n")
    print("Eigen vectors \n", eigen_vectors)

    eigen_vec, s, v = np.linalg.svd(X.T)
    print("Eigen Vectors \n", eigen_vec)

    for val in eigen_values:
        print(val)

    return eigen_vec[:n_components], eigen_values[:n_components]

