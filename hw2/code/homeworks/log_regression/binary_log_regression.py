from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem

# When choosing your batches / Shuffling your data you should use this RNG variable, and not `np.random.choice` etc.
RNG = np.random.RandomState(seed=446)
Dataset = Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]


def load_2_7_mnist() -> Dataset:
    
    (x_train, y_train), (x_test, y_test) = load_dataset("mnist")
    train_idxs = np.logical_or(y_train == 2, y_train == 7)
    test_idxs = np.logical_or(y_test == 2, y_test == 7)

    y_train_2_7 = y_train[train_idxs]
    y_train_2_7 = np.where(y_train_2_7 == 7, 1, -1)

    y_test_2_7 = y_test[test_idxs]
    y_test_2_7 = np.where(y_test_2_7 == 7, 1, -1)

    return (x_train[train_idxs], y_train_2_7), (x_test[test_idxs], y_test_2_7)


class BinaryLogReg:
    @problem.tag("hw2-A", start_line=4)
    def __init__(self, _lambda: float = 1e-3):

        self._lambda: float = _lambda
        # Fill in with matrix with the correct shape
        self.weight: np.ndarray = None  # type: ignore
        self.bias: float = 0.0
        # raise NotImplementedError("Your Code Goes Here")

    @problem.tag("hw2-A")
    def mu(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        
        mu = 1/(1 + np.exp(-y * (self.bias + np.matmul(X, self.weight))))
        
        return mu

    @problem.tag("hw2-A")
    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        
        n = y.size
        J = (1/n) * np.sum(np.log(1 + np.exp(-y*(self.bias + np.matmul(X, self.weight))))) + self._lambda * np.sum(self.weight**2)
         
        return J


    @problem.tag("hw2-A")
    def gradient_J_weight(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:

        n = y.size
        mu_vec = self.mu(X, y)
        
        J_weight = (1/n) * np.matmul(np.multiply((mu_vec-1), y), X) + (2 * self._lambda * self.weight)

        return J_weight
        

    @problem.tag("hw2-A")
    def gradient_J_bias(self, X: np.ndarray, y: np.ndarray) -> float:
        
        n = y.size
        mu_i = self.mu(X, y)
        J_bias = (1/n) * (np.dot(mu_i, y) - np.sum(y))
        
        return J_bias

    @problem.tag("hw2-A")
    def predict(self, X: np.ndarray) -> np.ndarray:

        predict = np.sign(self.bias + np.matmul(X, self.weight))

        return predict

    @problem.tag("hw2-A")
    def misclassification_error(self, X: np.ndarray, y: np.ndarray) -> float:

        n = y.size
        missclassified = np.where(y != self.predict(X))[0].size
        missclassified = missclassified/n
        
        return missclassified

    @problem.tag("hw2-A")
    def step(self, X: np.ndarray, y: np.ndarray, learning_rate: float = 1e-4):
        
        self.weight = self.weight - learning_rate * self.gradient_J_weight(X, y)
        self.bias = self.bias - learning_rate * self.gradient_J_bias(X, y)

    @problem.tag("hw2-A", start_line=7)
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        learning_rate: float = 1e-2,
        epochs: int = 30,
        batch_size: int = 100,
    ) -> Dict[str, List[float]]:
        
        num_batches = int(np.ceil(len(X_train) // batch_size))
        result: Dict[str, List[float]] = {
            "train_losses": [],  # You should append to these lists
            "train_errors": [],
            "test_losses": [],
            "test_errors": [],
        }
        n, d = X_train.shape
        self.weight = np.zeros(d)
        for epoch in range(epochs):
            for batch in range(num_batches):
                batch_i = RNG.choice(np.arange(n), size=batch_size)
                self.step(X_train[batch_i, :], y_train[batch_i], learning_rate=learning_rate)

            # Compute the losses and errors
            result["train_losses"].append(self.loss(X_train, y_train))
            result["train_errors"].append(self.misclassification_error(X_train, y_train))
            result["test_losses"].append(self.loss(X_test, y_test))
            result["test_errors"].append(self.misclassification_error(X_test, y_test))

        return result
    
def plot(history, learning_rate):
    
    # plot function for errors and losses

    # losses
    fig, (ax1, ax2) = plt.subplots(nrows=2)
    ax1.plot(history["train_losses"], label="Train")
    ax1.plot(history["test_losses"], label="Test")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.set_title(f'Learning Rate: {learning_rate}')
    ax1.legend()

    # error
    ax2.plot(history["train_errors"], label="Train")
    ax2.plot(history["test_errors"], label="Test")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Misclassification Error")
    ax2.legend()


    plt.show()
    
if __name__ == "__main__":
    model = BinaryLogReg(_lambda=0.1)
    (x_train, y_train), (x_test, y_test) = load_2_7_mnist()
    
    # GD 
    GD_lr = 0.1
    history_gd = model.train(x_train, y_train, x_test, y_test, learning_rate=GD_lr , epochs=30, batch_size=x_train.shape[0])
    plot(history_gd, learning_rate=GD_lr )

    # SGD Batch Size 1
    SGD_lr_b1 = 0.0001
    history_sgd_batch1 = model.train(x_train, y_train, x_test, y_test, learning_rate=SGD_lr_b1 , epochs=30, batch_size=1)
    plot(history_sgd_batch1, learning_rate=SGD_lr_b1 )

     # SGD Batch Size 100
    SGD_lr_b100 = 0.001
    history_sgd_batch100 = model.train(x_train, y_train, x_test, y_test, learning_rate=SGD_lr_b100, epochs=30, batch_size=100)
    plot(history_sgd_batch100, learning_rate=SGD_lr_b100)

