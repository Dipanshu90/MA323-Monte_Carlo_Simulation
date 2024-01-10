import numpy as np
import matplotlib.pyplot as plt
import math

#taking beta = 1
def f2x(x, alpha):
    return (1/math.gamma(alpha))*(x ** (alpha-1))*(np.e ** -x)

def gx(x, alpha):
    A = 1/alpha + 1/np.e
    if 0 < x < 1:
        return (x ** (alpha-1))/A
    elif x >= 1:
        return (np.e ** -x)/A
    else: 
        return 0
    
def Ginv(u, alpha):
    A = 1/alpha + 1/np.e
    if 0 < u < 1/(alpha*A):
        return (alpha*A*u) ** (1/alpha)
    elif 1/(alpha*A) <= u < 1:
        return -np.log(A) - np.log(1-u)
    else: 
        return 0
    
def pdf_f2x(c, alpha):
    count = 0
    samples = list()
    while count < 10000:
        u1 = np.random.uniform()
        x = Ginv(u1, alpha)
        u2 = np.random.uniform()
        if c*gx(x, alpha)*u2 <= f2x(x, alpha) and 0 < x < 1:
            samples.append(x)
            count += 1
    return samples

# f(x) = 20*x*(1-x)^3
def fx(x):
    return 20*x*(1-x) ** 3

def pdf_fx(c):
    count = 0
    itr = 1
    samples = list()
    itr_list = list()
    while count < 10000:
        x = np.random.uniform()
        u = np.random.uniform()
        if u <= fx(x)/c:
            samples.append(x)
            itr_list.append(itr)
            itr = 1
            count += 1
        else:
            itr += 1
    return samples, itr_list

def main():
    np.random.seed(69)

    #Q - 1
    X = np.arange(0, 1, 0.01)
    f = [fx(i) for i in X]
    C = [max(f), 6, 9]

    for c in C:
        print(f"For c = {c} :- \n")
        samples, itr_list = pdf_fx(c)
        samples = np.array(samples)
        itr_list = np.array(itr_list)
        print(f"Sample mean is :- {np.mean(samples)}")
        print(f"Expectation of pdf of f(x) is :- {1.00/3.00}\n")
        
        s = np.sum(np.fromiter((fx(sample) for sample in samples), float))
        prob = np.sum(fx(samples[np.logical_and(samples >= 0.25 , samples <= 0.75)]) / s)

        print(f"P(0.25 <= X <= 0.75) = {prob}")
        print("Actual Value = 0.617188\n")

        print(f"The average number of iterations for c = {c} are :- {np.mean(itr_list)}")
        print(f"Average number of iterations for generating random numbers :- {C[0]}\n")

        if c == 2.109375:
            plt.hist(samples, density=True, bins=100, color="red", alpha=0.5)
            x_val = np.linspace(0, 1, 100)
            plt.plot(x_val, fx(x_val), color="blue")
            plt.show()

    #Q - 2
    alpha = 0.5
    A = 1/alpha + 1/np.e
    c = A/math.gamma(alpha)

    print(f"The rejection constant for alpha = {alpha} is {c}")
    samples = pdf_f2x(c, alpha)
    samples = np.array(samples)

    plt.hist(samples, density=True, bins=100, color="red", alpha=0.5)
    x_val = np.linspace(0.001, 1.001, 1000)
    plt.plot(x_val, f2x(x_val,alpha), color="blue", label='Target PDF')
    plt.show()

if __name__ == "__main__":
    main()