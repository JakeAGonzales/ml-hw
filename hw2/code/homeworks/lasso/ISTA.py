from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from utils import problem


@problem.tag("hw2-A")
def step(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float, eta: float
) -> Tuple[np.ndarray, float]:
    
    a = 2*np.sum(X*X, axis=0)

    n, d = X.shape
    b = np.mean(y-np.dot(X, weight))

    for k in range(d): 
        x_k = X[:, k]
        weight[k] = 0
        c_k = 2*np.dot(x_k, y - (b + np.dot(X, weight)))
        if c_k < -_lambda:
            weight[k] = (c_k + _lambda) / a[k]
        elif c_k > _lambda: 
            weight[k] = (c_k - _lambda)/ a[k]

    return weight, bias 

@problem.tag("hw2-A")
def loss(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float
) -> float:
    #L-1 (Lasso) regularized MSE loss.
    loss_fn = np.linalg.norm(X @ weight + bias - y)**2 + _lambda * np.linalg.norm(weight, 1) 

    return loss_fn

@problem.tag("hw2-A", start_line=5)
def train(
    X: np.ndarray,
    y: np.ndarray,
    _lambda: float = 0.01,
    eta: float = 0.001,
    convergence_delta: float = 1e-4,
    start_weight: np.ndarray = None,
    start_bias: np.ndarray = None,
) -> Tuple[np.ndarray, float]:
    
    a = 2*np.sum(X*X, axis=0)

    x, d = X.shape

    
    if start_weight is None:
        start_weight = np.zeros(d)
        start_bias = 0
    old_w = start_weight + np.inf
    

    while convergence_criterion(start_weight, old_w, None, None, convergence_delta) is False: 
        old_w = np.copy(start_weight)
        w = start_weight
        train = step(X, y, w, None, _lambda, eta)

    return train

@problem.tag("hw2-A")
def convergence_criterion(
    weight: np.ndarray, old_w: np.ndarray, bias: float, old_b: float, convergence_delta: float
) -> bool:
    max_absolute_change = np.linalg.norm(weight - old_w, ord=np.inf) 
    
    if max_absolute_change < convergence_delta: 
        return True
    else: 
        return False


@problem.tag("hw2-A")
def main():
    
    sigma = 1
    n = 500
    d = 1000
    k = 100
    iter = 5
    # Gaussian noise for epsilon
    eps_noise = np.random.normal(0, sigma**2, size=n) 
    x = np.random.normal(size = (n,d))
    w = np.zeros((d, ))

    for j in range(1, k+1): 
        w[j-1] = j/k
    
    y = x @ w + eps_noise

    lambda_max = np.max(2 * np.sum(x.T * (y - np.mean(y)), axis=0))
    lambda_iter = [lambda_max/(2**i) for i in range (iter)] 

    fdr, tpr = [], []
    delta = 1e-5
    eta = 0.001

    # keep nonnzeros count 
    non_zeros = []

    for lambda_reg in lambda_iter:
        weight = train(x, y, lambda_reg, eta, delta, None, None)[0]
        not_Zeros = np.count_nonzero(weight)
        non_zeros.append(not_Zeros) 
        correct_nonZeros  = np.count_nonzero(weight[:k])
        incorrect_nonZeros = np.count_nonzero(weight[k+1:])

        try: 
            fdr.append(incorrect_nonZeros / not_Zeros)
        except ZeroDivisionError: 
            fdr.append(0) 

        tpr.append(correct_nonZeros/k)


    # first plot 
    plt.figure(figsize=(10,5))
    plt.plot(lambda_iter, non_zeros)
    plt.xscale('log')
    plt.title('Non-Zero Count v. Lambda')
    plt.xlabel('Lambda')
    plt.ylabel('Non-Zero Weight Elements')
    plt.show()

    # second plot
    plt.figure(figsize = (10,5))
    plt.plot(fdr, tpr)
    plt.title('TPR v. FDR') 
    plt.xlabel('False Discovery Rate')
    plt.ylabel('True Positive Rate')
    plt.show() 



if __name__ == "__main__":
    main()
