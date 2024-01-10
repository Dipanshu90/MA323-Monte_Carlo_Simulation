import numpy as np
import matplotlib.pyplot as plt

np.random.seed(69)

def muller(N):
    samples_1 = []
    samples_2 = []
    while len(samples_1) < N:
        u1 = np.random.uniform()
        u2 = np.random.uniform()
        r = np.sqrt(-2 * np.log(u1))
        t = 2 * u2 * np.pi
        z1 = r*np.cos(t)
        z2 = r*np.sin(t)
        samples_1.append(z1)
        samples_2.append(z2)


    return samples_1,samples_2

def generate_from_mix_distribution(n, K, q, mu, sigma):
    dist = []
    normaldist, _ = muller(n)
    for i in range (n):
        U = np.random.uniform()

        for k in range(1, K + 1):
            if q[k - 1] < U <= q[k]:
                dist.append(mu[k-1] + sigma[k-1] * normaldist[i])

    return dist

def main():
    print("####### Question - 1 #######")

    K = 3
    pi = [1/2, 1/3, 1/6]
    mu =[-1 ,0,1]
    sigma =[1/4,1,1/2]

    q = [0]  # q0
    for i in range(K):
        qk = q[-1] + pi[i]
        q.append(qk)

    q[-1]=1

    dist = generate_from_mix_distribution(5000, K, q, mu, sigma)
    mean_dist = np.mean(dist)
    variance_dist = np.var(dist)

    print(f"Mean of the generated distribution is {mean_dist}")
    print(f"Variance of the generated distribution is {variance_dist}\n")
    
    print("####### Question - 2 #######")
    num_paths = 10
    num_steps = 5000
    T = 5.0
    dt = T / num_steps

    sample_paths = np.zeros([num_paths, 1 + num_steps])

    for i in range(num_paths):
        temp = muller(num_steps)
        Z = temp[0]
        for j in range(num_steps):
            sample_paths[i, j+1] = sample_paths[i, j] + np.sqrt(dt) * Z[j]

    t = np.linspace(0, T, num_steps+1)

    plt.figure(figsize=(12, 6))
    for i in range(num_paths):
        plt.plot(t, sample_paths[i], label=f"Path {i+1}")

    plt.title("Sample Paths of Standard Brownian Motion")
    plt.xlabel("Time")
    plt.ylabel("W(t)")
    plt.grid(True)
    plt.legend()
    plt.show()

    W2 = np.mean(sample_paths[:,int(2/dt)])
    W2v = np.var(sample_paths[:,int(2/dt)])
    W5v = np.var(sample_paths[:,int(5/dt)])
    W5 = np.mean(sample_paths[:,int(5/dt)])


    print(f"Estimated E[W(2)]:{W2}")
    print(f"Estimated E[W(5)]:{W5}")
    print(f"Estimated Variance of W(2):{W2v}")
    print(f"Estimated Variance of W(5):{W5v}\n")

    print("####### Question - 3 #######")

    mu = 0.06
    sigma = 0.3

    sample_paths = np.zeros((num_paths, num_steps+1))
    sample_paths[:, 0] = 5.0

    for i in range(num_paths):
        temp = muller(num_steps)
        Z = temp[0]
        for j in range(num_steps):
            sample_paths[i, j+1] = sample_paths[i, j] + mu * dt + sigma * np.sqrt(dt) * Z[j]

    t = np.linspace(0, T, num_steps+1)

    plt.figure(figsize=(12, 6))
    for i in range(num_paths):
        plt.plot(t, sample_paths[i], label=f"Path {i+1}")

    plt.title("Sample Paths of Brownian Motion with Drift")
    plt.xlabel("Time")
    plt.ylabel("X(t)")
    plt.grid(True)
    plt.legend()
    plt.show()

    X2 = np.mean(sample_paths[:, int(2 / dt)])

    X5 = np.mean(sample_paths[:, int(5 / dt)])

    print("Estimated E[X(2)]:", X2)
    print("Estimated E[X(5)]:", X5)

if __name__ == "__main__":
    main()