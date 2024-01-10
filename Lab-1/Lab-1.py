#import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# function for Linear Congruence Generator
def LCG(a, b, m, x0, n): 
    arr1=[x0]
    arr2=[]
    for i in range(n+1):
        x0 = (a*x0 + b)%m
        arr1.append(x0)
        arr2.append(x0/m)
    return [arr1, arr2]

# function for checking distinct values
def checkDistinct(res):
    return len(set(res))

# Question - 1 -------------------------------------------------------

# Task - 1. (a)
for i in range(11):
    res = LCG(6, 0, 11, i, 35)
    print("a = 6, b = 0, m = 11, x0 = ", i)
    for j in res[0]:
        print(j, end = " ")
    print("")
    print("distinct values = ", checkDistinct(res[0]))
    print("")

print("")

# Task - 1. (b)
for i in range(11):
    res = LCG(3, 0, 11, i, 35)
    print("a = 3, b = 0, m = 11, x0 = ", i)
    for j in res[0]:
        print(j, end = " ")
    print("")
    print("distinct values = ", checkDistinct(res[0]))
    print("")


# Question - 2 -------------------------------------------------------

# Task - 2. (i)
for i in range(5):
    x1 = LCG(1597, 0, 244944, (i+1)*1000, 10000)
    data = pd.DataFrame(x1[1])
    data['cuts'] = pd.cut(data[0], [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1])
    values = data["cuts"].value_counts().sort_index()
    print(values.to_string(index=False))
    print("")
    plt.figure(figsize=(16, 8))
    values.plot.bar()
    label = "a = 1597, m = 244944, x0 = " + str((i+1)*1000)
    plt.title(label)
    plt.savefig(f"task_2-1-bar-{i+1}")
    plt.show()

print("")

# Task - 2. (ii)
for i in range(5):
    x1 = LCG(51749, 0, 244944, (i+1)*10000, 10000)
    data = pd.DataFrame(x1[1])
    data['cuts'] = pd.cut(data[0], [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1])
    values = data["cuts"].value_counts().sort_index()
    print(values.to_string(index=False))
    print("")
    plt.figure(figsize=(16, 8))
    values.plot.bar()
    label = "a = 51749, m = 244944, x0 = " + str((i+1)*10000)
    plt.title(label)
    plt.savefig(f"task_2-2-bar-{i+1}")
    plt.show()

# Question - 3 -------------------------------------------------------

x2 = LCG(1229, 1, 2048, 4000, 10000)
arr1 = x2[1].copy()
arr2 = x2[1].copy()

arr1.pop(0)
arr2.pop()

plt.figure(figsize=(16, 8))
plt.scatter(arr2, arr1)
plt.savefig("task_3-scatter_plot")
plt.show()