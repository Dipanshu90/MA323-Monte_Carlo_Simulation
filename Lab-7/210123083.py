import numpy as np
import math

def main():
    np.random.seed(69)

    M_vals = [100,1000,10000,100000]
    delta = 1.96
    for M in M_vals:
        X = list()
        
        m = int(M/2)
        for i in range(m):
            u = np.random.uniform(0,1)
            x = (math.exp(math.sqrt(u))+ math.exp(math.sqrt(1-u)))/2
            X.append(x)
        Im = np.mean(X)
        sm_sq = 0
        for i in range(m):
            sm_sq += (X[i] - Im)**2
        sm_sq /= (m-1)
        sm = math.sqrt(sm_sq)
    
        L = Im - (delta*sm/math.sqrt(m))
        U = Im + (delta*sm/math.sqrt(m))

        print(f"For M = {M}")
        print(f"Im \t\t\t= {Im.round(6)}")
        print(f"Confidence Interval \t= [{L.round(6)}, {U.round(6)}]")
        print(f"Variance \t\t= {sm_sq.round(6)}")
        print(f"Interval Length \t= {(U-L).round(6)}")
        print()

if __name__ == "__main__":
    main()