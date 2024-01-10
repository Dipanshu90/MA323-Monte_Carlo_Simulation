import numpy as np
import matplotlib.pyplot as plt 

np.random.seed(68)

def Van_der_Corput(N, base):
    Van_der_Corputseq = list()
    for i in range(N):
        num = i
        s_val=0
        p = -1
        while(num!=0):
            temp = num%base
            num = (int)(num/base)
            s_val = s_val + (temp * (base ** p))
            p = p-1
        Van_der_Corputseq.append(s_val)
    return Van_der_Corputseq

# function for Linear Congruence Generator
def LCG(a, b, m, x0, n): 
    #arr1=[x0]
    arr2=[]
    for i in range(n+1):
        x0 = (a*x0 + b)%m
        #arr1.append(x0)
        arr2.append(x0/m)
    return arr2

def plot_comparision(seq1, seq2, lcg1, lcg2):
    fig = plt.figure(figsize= (15, 10))
    plot_1 = fig.add_subplot(221)
    plot_1.hist(seq1, bins=25, rwidth=0.75)
    plot_1.set_xlabel('Values')
    plot_1.set_ylabel('Frequency')
    plot_1.set_title('Van der Corput sequence(100 values)')

    plot_2 = fig.add_subplot(222)
    plot_2.hist(lcg1, bins=25, rwidth=0.75)
    plot_2.set_title('LCG(100 values)')
    plot_2.set_ylabel('Frequency')
    plot_2.set_xlabel('Values')

    plot_3 = fig.add_subplot(223)
    plot_3.hist(seq2, bins=25, rwidth=0.75)
    plot_3.set_title('Van der Corput sequence(100000 values)')
    plot_3.set_ylabel('Frequency')
    plot_3.set_xlabel('Values')

    plot_4 = fig.add_subplot(224)
    plot_4.hist(lcg2, bins=25, rwidth=0.75)
    plot_4.set_xlabel('Values')
    plot_4.set_ylabel('Frequency')
    plot_4.set_title('LCG(100000 values)')

    plt.show()


def main():
    print("###### Question 1 ######")
    seq_25 = Van_der_Corput(25,2) #the first 25 values of the Van der Corput sequence
    print(seq_25) #printing the values

    seq_1000 = Van_der_Corput(1000,2)
    xi = seq_1000[0:999]
    xi1 = seq_1000[1:1000]

    plt.scatter(xi,xi1) #plotting the (x(i),x(i+1)) points on a graph
    plt.title('(x(i),x(i+1)) for 1000 values of Van der Corput sequence')
    plt.xlabel('x(i) values')
    plt.ylabel('x(i+1) values')
    plt.show()

    #Taking 100 and 100000 values of the Van der Corput sequence
    seq_100 = Van_der_Corput(100,2)
    seq_100000 = Van_der_Corput(100000,2)

    print("\n###### Question 2 ######")
    print('LCG used: x(i+1)=(1229*x(i)+13)%4096 with x(0)=67')
    #Taking 100 and 100000 values from the LCG
    lcg100 = LCG(1229, 13, 4096, 67, 100)
    lcg100000 = LCG(1229, 13, 4096, 67, 100000)

    plot_comparision(seq_100, seq_100000, lcg100, lcg100000)

    #Genrating the Halton sequence xi= (φ2(i), φ3(i)) for 100 and 100000 values and plotting

    φ2_100 = Van_der_Corput(100,2)
    φ2_100000 = Van_der_Corput(100000,2)
    φ3_100 = Van_der_Corput(100,3)
    φ3_100000 = Van_der_Corput(100000,3)

    plt.scatter(φ2_100, φ3_100) #plotting the (φ2(i), φ3(i)) points on a graph for 100 values
    plt.title('(φ2(i), φ3(i)) for 100 values of Halton sequence')
    plt.xlabel('φ2(i) values')
    plt.ylabel('φ3(i) values')
    plt.show()

    plt.scatter(φ2_100000, φ3_100000) #plotting the (φ2(i), φ3(i)) points on a graph for 100000 values
    plt.title('(φ2(i), φ3(i)) for 100000 values of Halton sequence')
    plt.xlabel('φ2(i) values')
    plt.ylabel('φ3(i) values')
    plt.show()

if __name__ == "__main__":
    main()