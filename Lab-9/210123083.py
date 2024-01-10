import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
sample_size = 10000

def exp_dist(theta):
    u = np.random.uniform(size = len(theta))
    x = -theta * np.log(u)
    return x

def find_prob(deadline, E10):
    cnt = 0
    for i in range(len(E10)):
        if E10[i] > deadline:
            cnt += 1
    return cnt / len(E10)

def std_dev(E10):
    x_bar = np.mean(E10)
    var = 0

    for x in E10:
        var += (x - x_bar) ** 2
    var /= (len(E10) - 1)
    std = np.sqrt(var)
    return std

def plot_hist(E10):
    plt.hist(E10, bins=100)
    plt.title("Histogram of obtained distribution of E10")
    plt.show()

def hx(lambda_, T):
    E10 = T[0] + T[9] + max(T[3] + T[1], T[7] + T[2], T[8] + max(T[4] + T[1], T[5] + T[2], T[6] + T[2]))
    if E10 > 70:
        return E10, np.power(4, 10) * np.exp((-3/4) * np.dot(lambda_, T)), np.power(4, 10) * np.exp((-3/4) * np.dot(lambda_, T))
    else:
        return E10, 0, np.power(4, 10) * np.exp((-3/4) * np.dot(lambda_, T))
    
def hx_for_part_f(lambda_1, lambda_2, k, T):
    E10 = T[0] + T[9] + max(T[3] + T[1], T[7] + T[2], T[8] + max(T[4] + T[1], T[5] + T[2], T[6] + T[2]))

    if E10 > 70:
        return E10, np.power(k, 4) * np.exp(np.dot(lambda_1, T)) / np.exp(np.dot(lambda_2, T)), np.power(k, 4) * np.exp(np.dot(lambda_1, T)) / np.exp(np.dot(lambda_2, T))
    else:
        return E10, 0, np.power(k, 4) * np.exp(np.dot(lambda_1, T)) / np.exp(np.dot(lambda_2, T))

def main():
    mean_times = np.array([4,4,2,5,2,3,2,3,2,2])

    # simple monte-carlo for estimating mean E10
    E10 = np.zeros(sample_size)
    for i in range(sample_size):
        T = exp_dist(mean_times)
        E10[i] = T[0] + T[9] + max(T[3] + T[1], T[7] + T[2], T[8] + max(T[4] + T[1], T[5] + T[2], T[6] + T[2]))
    estimated_E10 = np.mean(E10)
    print(f"The estimated value of E10 taking n = 10000 is :- {estimated_E10}")

    plot_hist(E10)

    deadline = 70 # in days
    print(f"The approximate probability that project misses the deadline of {deadline} days is :- {find_prob(deadline, E10)}")
    print(f"The approximate value of standard deviation for deadline - {deadline} days is :- {std_dev(E10)}\n")

    print("############ 1 (e) ############")

    hx_array = np.zeros(sample_size)
    E10_list = np.zeros(sample_size)
    expected_sample_size = np.zeros(sample_size)
    count = 0
    lambda_ = 1 / mean_times

    for i in range(sample_size):
        X = exp_dist(mean_times * 4)
        E10_list[i], hx_array[i], expected_sample_size[i] = hx(lambda_, X)
        count += hx_array[i]

    summation = np.sum(expected_sample_size)
    sqsum = np.sum(expected_sample_size ** 2)
    estimated_mean = np.mean(E10_list)
    print(f"The estimated value of hx taking n = 10000 is :- {estimated_mean}")

    print(f"The approximate probability that project misses the deadline of {deadline} days is :- {count / sample_size}")
    print(f"The approximate value of standard deviation for deadline - {deadline} days is :- {np.std(hx_array)}")
    print(f"Expected Sample Size = {summation**2 / sqsum}\n")

    plot_hist(E10_list)

    print("############ 1 (f) ############")
    K = np.array([3.0, 4.0, 5.0])

    for k in K:
        hx_array = np.zeros(sample_size)
        E10_list = np.zeros(sample_size)
        expected_sample_size = np.zeros(sample_size)
        count = 0
        lambda_1 = 1 / mean_times
        lambda_2 = 1 / mean_times

        for i in range(sample_size):
            X = mean_times
            for j in range(10):
                if j == 0 | j == 1 | j == 3 | j == 9:
                    X[j] *= k
                    lambda_2[j] = 1 / X[j]
            X = exp_dist(X)
            E10_list[i], hx_array[i], expected_sample_size[i] = hx_for_part_f(lambda_1, lambda_2, k, X)
            count += hx_array[i]

        summation = np.sum(expected_sample_size)
        sqsum = np.sum(expected_sample_size ** 2)
        estimated_mean = np.mean(E10_list)
        print(f"The estimated value of hx taking n = 10000 is :- {estimated_mean}")

        print(f"The approximate probability that project misses the deadline of {deadline} days is :- {count / sample_size}")
        print(f"The approximate value of standard deviation for deadline - {deadline} days is :- {np.std(hx_array)}")
        print(f"Expected Sample Size = {summation**2 / sqsum}\n")

    return 0

if __name__ == "__main__":
    main()