#Question 1
import numpy as np
import statistics

np.random.seed(42)

K = 3
π = [1/2, 1/3, 1/6]
mu =[-1 ,0,1]
sigma =[1/4,1,1/2]


def muller(num_samples):
  samples1 = []
  samples2 = []
  while len(samples1) < num_samples:
      U1 = np.random.uniform(0,1)
      U2 = np.random.uniform(0,1)
      R=np.sqrt(-2* np.log(U1))
      t=2*U2*np.pi
      Z1=R*np.cos(t)
      Z2=R*np.sin(t)
      samples1.append(Z1)
      samples2.append(Z2)


  return samples1,samples2

q = [0]  # q0
for i in range(K):
    qk = q[-1] + π[i]
    q.append(qk)

q[-1]=1


def generate_from_mixture_distribution(n):
    dist =[]
    normaldist=[]
    normaldist,_=muller(n)
    i = 0
    for i in range (n):
      U = np.random.uniform(0, 1)

      for k in range(1, K + 1):
          if q[k - 1] < U <= q[k]:
            dist.append(mu[k-1]+sigma[k-1]*normaldist[i])

    return dist

meandist= np.mean(generate_from_mixture_distribution(5000))
vardist= statistics.variance(generate_from_mixture_distribution(5000))

print(f"Mean of the generated distribution is {meandist}")
print(f"Variance of the generated distribution is {vardist}")

#Question 2
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

num_paths = 10
num_steps = 5000
T = 5.0
dt = T / num_steps

def muller(num_samples):
  samples1 = []
  samples2 = []
  while len(samples1) < num_samples:
      U1 = np.random.uniform(0,1)
      U2 = np.random.uniform(0,1)
      R=np.sqrt(-2* np.log(U1))
      t=2*U2*np.pi
      Z1=R*np.cos(t)
      Z2=R*np.sin(t)
      samples1.append(Z1)
      samples2.append(Z2)


  return samples1


sample_paths = np.zeros([num_paths,1+num_steps])


for i in range(num_paths):
    Z = muller(num_steps)
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
plt.show()

W2 = np.mean(sample_paths[:,int(2/dt)])
W2v = statistics.variance(sample_paths[:,int(2/dt)])
W5v = statistics.variance(sample_paths[:,int(5/dt)])
W5 = np.mean(sample_paths[:,int(5/dt)])


print("Estimated E[W(2)]:", W2)
print("Estimated E[W(5)]:", W5)
print("Estimated Variance of W(2):", W2v)
print("Estimated Variance of W(5):", W5v)

# Question 3
import numpy as np
import matplotlib.pyplot as plt
import statistics
np.random.seed(42)

num_paths = 10
num_steps = 5000
T = 5.0
dt = T / num_steps
mu = 0.06
sigma = 0.3

def muller(num_samples):
  samples1 = []
  samples2 = []
  while len(samples1) < num_samples:
      U1 = np.random.uniform(0,1)
      U2 = np.random.uniform(0,1)
      R=np.sqrt(-2* np.log(U1))
      t=2*U2*np.pi
      Z1=R*np.cos(t)
      Z2=R*np.sin(t)
      samples1.append(Z1)
      samples2.append(Z2)


  return samples1


sample_paths = np.zeros((num_paths, num_steps+1))
sample_paths[:, 0] = 5.0

for i in range(num_paths):
    Z = muller(num_steps)
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
# X2var= statistics.variance(sample_paths[:, int(2 / T * num_steps)])
X5 = np.mean(sample_paths[:, int(5 / dt)])
# X5var= statistics.variance(sample_paths[:, int(5 / T * num_steps)])
print("Estimated E[X(2)]:", X2)
print("Estimated E[X(5)]:", X5)
# print("Estimated V[X(2)]:", X2var)
# print("Estimated V[X(5)]:", X5var)
