from typing import List, Tuple

import numpy as np

from utils import problem


@problem.tag("hw4-A")
def calculate_centers(
    data: np.ndarray, classifications: np.ndarray, num_centers: int
) -> np.ndarray:
    """
    Sub-routine of Lloyd's algorithm that calculates the centers given datapoints and their respective classifications/assignments.
    num_centers is additionally provided for speed-up purposes.

    Args:
        data (np.ndarray): Array of shape (n, d). Training data set.
        classifications (np.ndarray): Array of shape (n,) full of integers in range {0, 1, ...,  num_centers - 1}.
            Data point at index i is assigned to classifications[i].
        num_centers (int): Number of centers for reference.
            Might be usefull for pre-allocating numpy array (Faster that appending to list).

    Returns:
        np.ndarray: Array of shape (num_centers, d) containing new centers.
    """
    # num_centers/k = 10 
    _index = np.argsort(classifications)
    data = data[_index]
    classifications = classifications[_index] 

    _, index = np.unique(classifications, return_counts=False, return_index=True) 
    clusters = np.split(data, index[1:] )

    centers = np.array([np.mean(cluster, axis=0) for cluster in clusters])

    return centers



@problem.tag("hw4-A")
def cluster_data(data: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Sub-routine of Lloyd's algorithm that clusters datapoints to centers given datapoints and centers.

    Args:
        data (np.ndarray): Array of shape (n, d). Training data set.
        centers (np.ndarray): Array of shape (k, d). Each row is a center to which a datapoint can be clustered.

    Returns:
        np.ndarray: Array of integers of shape (n,), with each entry being in range {0, 1, 2, ..., k - 1}.
            Entry j at index i should mean that j^th center is the closest to data[i] datapoint.
    """
    
    distances = [np.sqrt(np.sum((data - center)**2, axis=1)) for center in centers]
    total_distances = np.asarray(distances).T
    assignment = np.argmin(total_distances, axis=1) 

    return assignment


def calculate_error(data: np.ndarray, centers: np.ndarray) -> float:
    """ This method has been implemented for you.
    
    Calculates error/objective function on a provided dataset, with trained centers.

    Args:
        data (np.ndarray): Array of shape (n, d). Dataset to evaluate centers on.
        centers (np.ndarray): Array of shape (k, d). Each row is a center to which a datapoint can be clustered.
            These should be trained on training dataset.

    Returns:
        float: Single value representing mean objective function of centers on a provided dataset.
    """
    distances = np.zeros((data.shape[0], centers.shape[0]))
    for idx, center in enumerate(centers):
        distances[:, idx] = np.sqrt(np.sum((data - center) ** 2, axis=1))
        
    error = np.mean(np.min(distances, axis=1))

    return error



def kmeans_plusplus(data: np.ndarray, num_centers: int
) -> np.ndarray: 
    

    



    return

def random_centers(data: np.ndarray, num_centers: int
) -> np.ndarray: 
    
    random_centers = data[np.random.permutation(np.arange(len(data)))[:num_centers]]

    return random_centers


@problem.tag("hw4-A")
def lloyd_algorithm(
    data: np.ndarray, num_centers: int, epsilon: float = 10e-3
) -> Tuple[np.ndarray, List[float]]:
    """Main part of Lloyd's Algorithm.

    Args:
        data (np.ndarray): Array of shape (n, d). Training data set.
        num_centers (int): Number of centers to train/cluster around.
        epsilon (float, optional): Epsilon for stopping condition.
            Training should stop when max(abs(centers - previous_centers)) is smaller or equal to epsilon.
            Defaults to 10e-3.

    Returns:
        np.ndarray: Tuple of 2 numpy arrays:
            Element at index 0: Array of shape (num_centers, d) containing trained centers.
            Element at index 1: List of floats of length # of iterations
                containing errors at the end of each iteration of lloyd's algorithm.
                You should use the calculate_error() function that has been implemented for you.

    Note:
        - For initializing centers please use the first `num_centers` data points.
    """
    
    init_centers = random_centers(data, num_centers)
    init_clusters = cluster_data(data, init_centers) 

    centers = calculate_centers(data, init_clusters, num_centers) 

    center_difference = np.max(np.abs(centers - init_centers)) 
 
    errors_list = []

    # run unitl objective error is less than epsilon = 10e-3

    while center_difference > epsilon: 
        print(center_difference) 
        init_centers = centers
        classifications = cluster_data(data, init_centers) 

        centers = calculate_centers(data, classifications, num_centers) 
        center_difference = np.max(np.abs(centers - init_centers)) 

        error = calculate_error(data, centers) 
        errors_list.append(error) 

    return centers, errors_list


