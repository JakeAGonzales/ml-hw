if __name__ == "__main__":
    from k_means import lloyd_algorithm  # type: ignore
else:
    from .k_means import lloyd_algorithm

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem

import os


@problem.tag("hw4-A", start_line=1)
def main():
    """Main function of k-means problem

    Run Lloyd's Algorithm for k=10, and report 10 centers returned.

    NOTE: This code might take a while to run. For debugging purposes you might want to change:
        x_train to x_train[:10000]. CHANGE IT BACK before submission.
    """
    (x_train, _), _ = load_dataset("mnist")
    
    # number of cluster centers k=10
    num_centers = 10
    centers, errors_list = lloyd_algorithm(x_train, num_centers) 

    path = os.path.abspath(__file__) 
    fig_path = os.path.join(path, 'figs/figure.png')

    for k in range(num_centers): 
        image = centers[k].reshape(28,28) 
        fig = plt.figure(k)
        plt.imshow(image, cmap='Greys')
        plt.show()


if __name__ == "__main__":
    main()
