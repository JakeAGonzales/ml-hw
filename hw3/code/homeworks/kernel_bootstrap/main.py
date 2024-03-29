from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


def f_true(x: np.ndarray) -> np.ndarray:
    return 4 * np.sin(np.pi * x) * np.cos(6 * np.pi * x ** 2)


@problem.tag("hw3-A")
def poly_kernel(x_i: np.ndarray, x_j: np.ndarray, d: int) -> np.ndarray:
    """Polynomial kernel.

    Given two indices a and b it should calculate:
    K[a, b] = (x_i[a] * x_j[b] + 1)^d

    Args:
        x_i (np.ndarray): An (n,) array. Observations (Might be different from x_j).
        x_j (np.ndarray): An (m,) array. Observations (Might be different from x_i).
        d (int): Degree of polynomial.

    Returns:
        np.ndarray: A (n, m) matrix, where each element is as described above (see equation for K[a, b])

    Note:
        - It is crucial for this function to be vectorized, and not contain for-loops.
            It will be called a lot, and it has to be fast for reasonable run-time.
        - You might find .outer functions useful for this function.
            They apply an operation similar to xx^T (if x is a vector), but not necessarily with multiplication.
            To use it simply append .outer to function. For example: np.add.outer, np.divide.outer
    """
    K = (np.outer(x_i, x_j) + 1)**d

    return K


@problem.tag("hw3-A")
def rbf_kernel(x_i: np.ndarray, x_j: np.ndarray, gamma: float) -> np.ndarray:
    """Radial Basis Function (RBF) kernel.

    Given two indices a and b it should calculate:
    K[a, b] = exp(-gamma*(x_i[a] - x_j[b])^2)

    Args:
        x_i (np.ndarray): An (n,) array. Observations (Might be different from x_j).
        x_j (np.ndarray): An (m,) array. Observations (Might be different from x_i).
        gamma (float): Gamma parameter for RBF kernel. (Inverse of standard deviation)

    Returns:
        np.ndarray: A (n, m) matrix, where each element is as described above (see equation for K[a, b])

    Note:
        - It is crucial for this function to be vectorized, and not contain for-loops.
            It will be called a lot, and it has to be fast for reasonable run-time.
        - You might find .outer functions useful for this function.
            They apply an operation similar to xx^T (if x is a vector), but not necessarily with multiplication.
            To use it simply append .outer to function. For example: np.add.outer, np.divide.outer
    """
    K = np.exp(-gamma * (np.subtract.outer(x_i, x_j)) ** 2)

    return K


@problem.tag("hw3-A")
def train(
    x: np.ndarray,
    y: np.ndarray,
    kernel_function: Union[poly_kernel, rbf_kernel],  # type: ignore
    kernel_param: Union[int, float],
    _lambda: float,
) -> np.ndarray:
    """Trains and returns an alpha vector, that can be used to make predictions.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        kernel_function (Union[poly_kernel, rbf_kernel]): Either poly_kernel or rbf_kernel functions.
        kernel_param (Union[int, float]): Gamma (if kernel_function is rbf_kernel) or d (if kernel_function is poly_kernel).
        _lambda (float): Regularization constant.

    Returns:
        np.ndarray: Array of shape (n,) containing alpha hat as described in the pdf.
    """
    n = len(x) 
    K = kernel_function(x, x, kernel_param) # Either poly_kernel or rbf_kernel
    alpha = np.linalg.inv((K + _lambda * np.eye(n,n))).dot(y) 

    return alpha

@problem.tag("hw3-A", start_line=1)
def cross_validation(
    x: np.ndarray,
    y: np.ndarray,
    kernel_function: Union[poly_kernel, rbf_kernel],  # type: ignore
    kernel_param: Union[int, float],
    _lambda: float,
    num_folds: int,
) -> float:
    """Performs cross validation.

    In a for loop over folds:
        1. Set current fold to be validation, and set all other folds as training set.
        2, Train a function on training set, and then get mean squared error on current fold (validation set).
    Return validation loss averaged over all folds.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        kernel_function (Union[poly_kernel, rbf_kernel]): Either poly_kernel or rbf_kernel functions.
        kernel_param (Union[int, float]): Gamma (if kernel_function is rbf_kernel) or d (if kernel_function is poly_kernel).
        _lambda (float): Regularization constant.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or 10 for 10-fold CV.

    Returns:
        float: Average loss of trained function on validation sets across folds.
    """
    fold_size = len(x) // num_folds
    error = []
    start = 0
    end = fold_size

    for i in range(num_folds): 
        x_k = x[start : end]
        y_k = y[start : end]
        i_arr = np.array(range(start, end, 1)) 
        x_train = np.delete(x, i_arr, 0)
        y_train = np.delete(y, i_arr, 0)

        start += fold_size
        end += fold_size

        alpha = train(x_train, y_train, kernel_function, kernel_param, _lambda)
        y_predicted = alpha.dot((kernel_function(x_train, x_k, kernel_param)))
        error_k = np.mean((np.subtract(y_k, y_predicted))**2) 
        error.append(error_k)

    avg_loss = np.mean(error) 

    return avg_loss


@problem.tag("hw3-A")
def rbf_param_search(
    x: np.ndarray, y: np.ndarray, num_folds: int
) -> Tuple[float, float]:
    """
    Parameter search for RBF kernel.

    There are two possible approaches:
        - Grid Search - Fix possible values for lambda, loop over them and record value with the lowest loss.
        - Random Search - Fix number of iterations, during each iteration sample lambda from some distribution and record value with the lowest loss.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        num_folds (int): Number of folds. It should be len(x) for LOO.

    Returns:
        Tuple[float, float]: Tuple containing best performing lambda and gamma pair.

    Note:
        - You do not really need to search over gamma. 1 / (median(dist(x_i, x_j)^2) for all unique pairs x_i, x_j in x
            should be sufficient for this problem. That being said you are more than welcome to do so.
        - If using random search we recommend sampling lambda from distribution 10**i, where i~Unif(-5, -1)
        - If using grid search we recommend choosing possible lambdas to 10**i, where i=linspace(-5, -1)
    """
    n = len(x)
    diff = []

    for i in range(n): 
        for j in range(i + 1, n): 
            diff.append((x[i] - x[j])**2)

    gamma = 1 / np.median(diff)
    gamma_sample = np.random.normal(gamma, 1, 50) 

    error_min = float("inf")
    lambda_min = None
    gamma_min = None 

    lambda_sample = 10 ** np.linspace(-5, -1, num=100) 

    for lambda_ in lambda_sample:
        for gamma in gamma_sample:
            error = cross_validation(x, y, rbf_kernel, lambda_, gamma, num_folds) 

            if error < error_min: 
                lambda_min = lambda_
                gamma_min = gamma
                error_min = error

    return [lambda_min, gamma_min]


@problem.tag("hw3-A")
def poly_param_search(
    x: np.ndarray, y: np.ndarray, num_folds: int
) -> Tuple[float, int]:
    """
    Parameter search for Poly kernel.

    There are two possible approaches:
        - Grid Search - Fix possible values for lambdas and ds.
            Have nested loop over all possibilities and record value with the lowest loss.
        - Random Search - Fix number of iterations, during each iteration sample lambda, d from some distributions and record value with the lowest loss.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or 10 for 10-fold CV.

    Returns:
        Tuple[float, int]: Tuple containing best performing lambda and d pair.

    Note:
        - You can use gamma = 1 / median((x_i - x_j)^2) for all unique pairs x_i, x_j in x) for this problem. 
          However, if you would like to search over other possible values of gamma, you are welcome to do so.
        - If using random search we recommend sampling lambda from distribution 10**i, where i~Unif(-5, -1)
            and d from distribution [5, 6, ..., 24, 25]
        - If using grid search we recommend choosing possible lambdas to 10**i, where i=linspace(-5, -1)
            and possible ds to [5, 6, ..., 24, 25]
    """
    n = len(x)

    error_min = float("inf")
    lambda_min = None
    d_min = 0

    lambda_sample = 10 ** np.linspace(-5, -1, num=100)
    polynomial_sample = np.arange(5, 25) 

    for lambda_ in lambda_sample:
        for polynomial in polynomial_sample: 
            error = cross_validation(x, y, poly_kernel, lambda_, polynomial, num_folds)

            if error < error_min: 
                lambda_min = lambda_
                d_min = polynomial
                error_min = error

    return[lambda_min, d_min]

@problem.tag("hw3-A", start_line=1)
def main():
    """
    Main function of the problem

    It should:
        A. Using x_30, y_30, rbf_param_search and poly_param_search report optimal values for lambda (for rbf), gamma, lambda (for poly) and d.
            Note that x_30, y_30 has been loaded in for you. You do not need to use (x_300, y_300) or (x_1000, y_1000).
        B. For both rbf and poly kernels, train a function using x_30, y_30 and plot predictions on a fine grid

    Note:
        - In part b fine grid can be defined as np.linspace(0, 1, num=100)
        - When plotting you might find that your predictions go into hundreds, causing majority of the plot to look like a flat line.
            To avoid this call plt.ylim(-6, 6).
    """
    (x_30, y_30), (x_300, y_300), (x_1000, y_1000) = load_dataset("kernel_bootstrap")
    
    rbf_params = rbf_param_search(x_30, y_30, 10) 
    print(" \n RBF Params: \n  ", rbf_params)
    polynomial_params = poly_param_search(x_30, y_30, 10) 
    #print("\n Poly Params: \n ", polynomial_params)

    fine_grid = np.linspace(0, 1, num=100) 

    alpha_rbf = train(x_30, y_30, rbf_kernel, rbf_params[1], rbf_params[0])
    alpha_rbf_predicted = alpha_rbf.dot(rbf_kernel(x_30, fine_grid, rbf_params[1]))

    alpha_poly = train(x_30, y_30, poly_kernel, polynomial_params[1], polynomial_params[0])
    alpha_poly_predicted = alpha_poly.dot(poly_kernel(x_30, fine_grid, polynomial_params[1]))

    y_true = f_true(fine_grid) 

    plt.scatter(x_30, y_30, color='forestgreen', label='Data') 
    plt.plot(fine_grid, y_true, label= 'True Function')
    plt.plot(fine_grid, alpha_rbf_predicted, label='RBF Kernel Prediction')
    #plt.plot(fine_grid, alpha_poly_predicted, label='Poly Kernel Prediction')
    plt.title("RBF Kernel") 
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
