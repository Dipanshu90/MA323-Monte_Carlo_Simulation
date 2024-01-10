import numpy as np

# Calculated the integral of e^sqrt(x)
def exact_val(x):
    return 2 * np.exp(np.sqrt(x)) * (np.sqrt(x) - 1)

def monte_carlo_estimator(M):
    np.random.seed(69)
    U_samples = np.random.rand(M)  # Generate M random samples from U(0,1)
    Y_samples = np.exp(np.sqrt(U_samples))  # Calculate Y_i
    I_M = np.mean(Y_samples)  # Calculate I_M

    # Calculate the value of S_n.
    Si = 0
    mui = Y_samples[0]

    for i in range(1, M):
        diff = Y_samples[i] - mui
        mui = mui + diff / (i + 1)
        Si = Si + (np.square(diff) * i) / (i + 1)

    S_n = np.sqrt(Si / (M - 1))

    # Determine the confidence interval for I_M.
    ci = ((I_M - 1.96 * (S_n / np.sqrt(M))).round(6), (I_M + 1.96 * (S_n / np.sqrt(M))).round(6))

    # Determining the exact value of I and comparing with estimated value.
    lower_lt = 0 
    upper_lt = 1
    I = exact_val(upper_lt) - exact_val(lower_lt)
    print(f"Sample size: {M}")
    print(f"Exact value(I): {I}, estimated value(I_M): {I_M.round(6)}")

    return ci

def main():
    M_values = [10**2, 10**3, 10**4, 10**5]

    for val in M_values:
        ci = monte_carlo_estimator(val)
        print(f"Confidence interval of 95%: {ci}")
        print(f"Interval length: {(ci[1] - ci[0]).round(6)}\n")

    return 0

if __name__ == "__main__":
    main()