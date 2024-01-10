import numpy as np
import matplotlib.pyplot as plt
import cmath
import math

# function for Linear Congruence Generator
def LCG(a, b, m, x0, n):
    arr=[]
    for i in range(n+1):
        x0 = (a*x0 + b)%m
        arr.append(x0/m)
    return arr

# function using the recursion to generate values
def recfun(n, arr):
    for i in range(len(arr) - 1, n+1):
        x = arr[i-17] - arr[i-5]
        if x < 0:
            x += 1
        arr.append(x)
    return arr

# function to plot histogram
def plot_hist(arr, n, i):
    a = np.array(arr)
    bins_array = np.arange(0, 1, 0.01)
    plt.hist(a, bins = bins_array)
    plt.title(f"N={n}")
    plt.savefig(f"plot_hist{i}")
    #plt.xticks(bins_array)
    plt.show()

# function to plot (Ui, Ui+1)
def plot_U(arr, n):
    arr1 = arr.copy()
    arr2 = arr.copy()

    arr1.pop()
    arr2.pop(0)

    plt.figure(figsize=(16, 8))
    plt.scatter(arr1, arr2)
    plt.savefig(f"N={n}")
    plt.show()

def ECDF_exp(N, theta):
    np.random.seed(69)
    u = np.random.uniform(size = N)
    samples = list()
    
    for x in u:
        samples.append(-theta * math.log(1 - x))
    
    sorted_samples = np.sort(samples)
    # print(np.arange(len(sorted_samples)))

    y_axis = np.arange(len(sorted_samples))/float(len(sorted_samples) - 1)
    
    plt.title(f'Empirical Cumulative Distribution Function of X ( sample count = {N} )')
    plt.ylabel('Probability - P(X <= x)')
    plt.xlabel('X (generated values)')
    plt.plot(sorted_samples, y_axis, color='blue')
    plt.show()

    print("Sample count =", N)
    print("Mean =", np.mean(samples))
    print("Variance =", np.var(samples))
    print()

def Actual_CDF_exp(rounds, theta):
    x = np.linspace(0, 20, rounds)
    y = 1 - np.exp(-x/theta)
    plt.title(f"Actual CDF of F(x) \n Number of rounds in Simulation = {rounds}", fontsize=20)
    plt.xlabel("x-values", fontsize=20)
    plt.ylabel("F(x)-values", fontsize=20)
    plt.plot(x,y, color='blue')
    plt.show()

def ECDF_arcsin(N):
    np.random.seed(69)
    u = np.random.uniform(size = N)
    samples = list()
    
    for x in u:
        samples.append(math.sin(math.pi*x/2) ** 2)
    
    sorted_samples = np.sort(samples)
    # print(np.arange(len(sorted_samples)))

    y_axis = np.arange(len(sorted_samples))/float(len(sorted_samples) - 1)
    
    plt.title(f'Empirical Cumulative Distribution Function of X ( sample count = {N} )')
    plt.ylabel('Probability - P(X <= x)')
    plt.xlabel('X (generated values)')
    plt.plot(sorted_samples, y_axis, color='blue')
    plt.show()

    print("Sample count =", N)
    print("Mean =", np.mean(samples))
    print("Variance =", np.var(samples))
    print()

def Actual_CDF_arcsin(rounds):
    x = np.linspace(0, 1, rounds)
    y = (2/math.pi)*np.arcsin(np.sqrt(x))
    plt.title(f"Actual CDF of F(x) \n Number of rounds in Simulation = {rounds}", fontsize=20)
    plt.xlabel("x-values", fontsize=20)
    plt.ylabel("F(x)-values", fontsize=20)
    plt.plot(x,y, color='blue')
    plt.show()

# Que - 1
arr1 = LCG(5644, 32, 48486, 5616, 17)
#print(arr1)

arr2 = arr1.copy()
recfun(1000, arr2)
# print(arr1)
# print(arr2)

arr3 = arr2.copy()
recfun(10000, arr3)

arr4 = arr3.copy()
recfun(100000, arr4)

plot_hist(arr2, 1000, 1)
plot_hist(arr3, 10000, 2)
plot_hist(arr4, 100000, 3)

plot_U(arr2, 1000)
plot_U(arr3, 10000)
plot_U(arr4, 100000)


# Que - 2
samples = [10, 100, 1000, 10000, 100000]
theta = cmath.pi/2
actual_mean = theta
actual_var = theta ** 2
print(actual_mean, " ", actual_var)
for sample in samples:
    ECDF_exp(sample, theta)
    Actual_CDF_exp(sample, theta)

# Que - 3
samples = [10, 100, 1000, 10000, 100000]
for sample in samples:
    ECDF_arcsin(sample)
    Actual_CDF_arcsin(sample)

# Que - 4
d_U = np.arange(1, 10000, 2)
N = 100000
h = np.zeros(len(d_U))
q = [i/5000 for i in range(1,5001)]
ud = np.random.uniform(size=N)
for u in ud:
    for j in range(1,5000):
        if q[j-1] < u and u <= q[j]:
            h[j] += 1
            break

plt.bar(x=d_U, height=h)
plt.title("Sample size : 100000")
plt.title("K")
plt.ylabel("Frequency")
plt.show()
