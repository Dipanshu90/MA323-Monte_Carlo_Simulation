import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import gaussian_kde

def box_muller(n):
    samples = []

    for i in range(n):
        u1 = np.random.uniform()
        u2 = np.random.uniform()

        R = math.sqrt(-2 * math.log(u1))
        theta = 2 * math.pi * u2
        z1 = R * math.cos(theta)
        z2 = R * math.sin(theta)
        samples.append([z1, z2])

    return samples

def cholesky(MU, SIGMA, samples):
    sigma1 = math.sqrt(SIGMA[0,0])
    sigma2 = math.sqrt(SIGMA[1,1])

    rho = SIGMA[0,1] / (sigma1*sigma2)

    A = np.array([[sigma1, 0],
                 [rho*sigma2, math.sqrt(1 - rho**2)*sigma2]])
    X = (MU + np.matmul(A, samples.T)).T

    return X

def plot(X, x1, y1, z1, a):
    plt.figure(figsize=(12, 8))
    plt.hist2d(X[:, 0], X[:, 1], bins=50, cmap="tab20c")
    plt.colorbar()
    plt.title(f'Contour Plot for a = {a}')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.hist2d(X[:, 0], X[:, 1], bins=50, cmap="tab20c")
    plt.colorbar()
    plt.contour(x1, y1, z1.reshape(x1.shape), levels=20, alpha=0.5, colors="black")
    plt.title(f'Contour Plot for a = {a}')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

def main():
    Z = np.array(box_muller(10000))
    
    a_val = np.array([-0.5, 0, 0.5, 1])

    for a in a_val:
        MU = np.array([5, 8]).reshape(2,-1)
        SIGMA = np.array([[1, 2*a], [2*a, 4]])

        X = cholesky(MU, SIGMA, Z)

        np.random.seed(15658)
        actual_dist = np.random.multivariate_normal(MU.ravel(), SIGMA, size=10000)

        X1, X2 = actual_dist[:, 0], actual_dist[:, 1]
        kde = gaussian_kde([X1, X2])
        x1, y1 = np.meshgrid(np.linspace(X1.min(), X1.max(), 100), np.linspace(X2.min(), X2.max(), 100))
        z1 = kde(np.vstack([x1.flatten(), y1.flatten()]))

        plot(X,x1,y1,z1,a)

    return 0

if __name__ == "__main__":
    main()