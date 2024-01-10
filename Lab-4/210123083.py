import numpy as np
import matplotlib.pyplot as plt
import math
import time

np.random.seed(86)

def normal(mean, var, samples):
    std = math.sqrt(var)
    y = (1/(np.sqrt(2 * math.pi) * std) * np.exp(-0.5 * ((samples - mean) / std) ** 2))
    return y

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

def marsaglia(n):
    cnt_r, cnt_total = 0, 0
    cnt_a = 0
    samples = []
    while cnt_a < n:
        u1 = np.random.uniform()
        u2 = np.random.uniform()

        cnt_total += 1
        u1 = 2 * u1 - 1
        u2 = 2 * u2 - 1

        if u1 ** 2 + u2 ** 2 > 1:
            cnt_r += 1
            continue

        x = -2 * math.log(u1 ** 2 + u2 ** 2) / (u1 ** 2 + u2 ** 2)
        x = np.sqrt(x)

        z1 = u1 * x
        z2 = u2 * x

        cnt_a += 1
        samples.append([z1, z2])
    
    return samples, round(cnt_r/cnt_total, 6)

def plot(mode, mean, var, samples, rejection_constant):
    actual_samples = np.linspace(mean - 7.5, mean + 7.5, 2000)
    actual_samplesy = normal(mean, var, actual_samples)

    print(f"Mean and Variance for N(0,1) samples :- \n")
    print(f"The mean of this sample is {mean}")
    print(f"The variance of this sample is {var}\n")

    plt.hist(samples, bins=80, density=True)
    plt.plot(actual_samples, actual_samplesy, c='r')
    plt.show()

    samples_1 = samples * math.sqrt(5)
    mean = np.mean(samples_1)
    var = np.var(samples_1)

    actual_samples = np.linspace(mean - 7.5, mean + 7.5, 2000)
    actual_samplesy = normal(mean, var, actual_samples)

    print(f"Mean and Variance for N(0,5) samples :- \n")
    print(f"The mean of this sample is {mean}")
    print(f"The variance of this sample is {var}\n")

    plt.hist(samples_1, bins=80, density=True)
    plt.plot(actual_samples, actual_samplesy, c='r')
    plt.show()

    samples_1 = samples * math.sqrt(5) + 5
    mean = np.mean(samples_1)
    var = np.var(samples_1)

    actual_samples = np.linspace(mean - 7.5, mean + 7.5, 2000)
    actual_samplesy = normal(mean, var, actual_samples)

    print(f"Mean and Variance for N(5,5) samples :- \n")
    print(f"The mean of this sample is {mean}")
    print(f"The variance of this sample is {var}")
    if mode == 0:
        print()

    plt.hist(samples_1, bins=80, density=True)
    plt.plot(actual_samples, actual_samplesy, c='r')
    plt.show()

    if mode == 1:
        print(f"The proportion of values rejected is :- {rejection_constant}\n")

def print_time(mode, n, time):
    if mode == 0:
        print(f"The time for box muller of sample size {n} is :- {time}")

    elif mode == 1:
        print(f"The time for marsaglia of sample size {n} is :- {time}")

def main():
    # box muller
    t1 = time.time()
    s1 = np.array(box_muller(100))
    t2 = time.time()

    print_time(0, 100, t2-t1)

    t1 = time.time()
    s2 = np.array(box_muller(10000))
    t2 = time.time()

    print_time(0, 10000, t2-t1)

    samples1 = s1[:, 0]
    samples2 = s2[:, 0]
    
    mean1 = np.mean(samples1)
    var1 = np.var(samples1)

    mean2 = np.mean(samples2)
    var2 = np.var(samples2)

    plot(0, mean1, var1, samples1, 0)
    plot(0, mean2, var2, samples2, 0)

    # marsiglia
    t1 = time.time()
    [s1, rejection_rate1] = marsaglia(100)
    t2 = time.time()

    print_time(1, 100, t2-t1)

    t1 = time.time()
    [s2, rejection_rate2] = marsaglia(10000)
    t2 = time.time()

    print_time(1, 10000, t2-t1)

    samples1 = np.array(s1)[:, 0]
    samples2 = np.array(s2)[:, 0]

    mean1 = np.mean(samples1)
    var1 = np.var(samples1)

    mean2 = np.mean(samples2)
    var2 = np.var(samples2)

    plot(1, mean1, var1, samples1, rejection_rate1)
    plot(1, mean2, var2, samples2, rejection_rate2)

if __name__ == "__main__":
    main()