import numpy as np
from matplotlib import pyplot as plt
import random
random.seed(10)
delta = 0.05
n = 20000
d = 10000
beta = np.zeros(d)
X = np.empty((n,d))
inv_XTX = np.zeros((d,d))
y = np.random.normal(0, 1, n)
for i in range(n): 
    index = (i%d)+1
    e = np.zeros(d)
    e[index-1] = 1.0
    X[i,:] = np.sqrt(index)*e

XTX = X.T @ X
ci = np.zeros(d)

for i in range(d):
    inv_XTX[i,i] = 1/(XTX[i,i])
    ci[i] =np.sqrt(2*inv_XTX[i,i]*np.log(2/delta))

beta_hat = inv_XTX @ X.T @ y

count = 0
for i in range(d): 
    if np.abs(beta_hat[i])> ci[i]:
        count += 1
x = np.linspace(1,d,d)
plt.figure(figsize=(20,20))
plt.plot(x,beta_hat, '.')
plt.fill_between(x, (beta-ci), (beta+ci), color='orange')
plt.xlabel("i")
plt.ylabel("beta_hat[i]")
plt.title("Confidence Interval for Least Squares Estimator Beta")
plt.show()
print(count)