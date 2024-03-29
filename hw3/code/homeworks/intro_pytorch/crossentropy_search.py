if __name__ == "__main__":
    from layers import LinearLayer, ReLULayer, SigmoidLayer, SoftmaxLayer
    from losses import CrossEntropyLossLayer
    from optimizers import SGDOptimizer
    from train import plot_model_guesses, train
else:
    from .layers import LinearLayer, ReLULayer, SigmoidLayer, SoftmaxLayer
    from .optimizers import SGDOptimizer
    from .losses import CrossEntropyLossLayer
    from .train import plot_model_guesses, train

from typing import Any, Dict

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem

RNG = torch.Generator()
RNG.manual_seed(446)


@problem.tag("hw3-A")
def crossentropy_parameter_search(
    dataset_train: TensorDataset, dataset_val: TensorDataset
) -> Dict[str, Any]:
    """
    Main subroutine of the CrossEntropy problem.
    It's goal is to perform a search over hyperparameters, and return a dictionary containing training history of models, as well as models themselves.

    Models to check (please try them in this order):
        - Linear Regression Model
        - Network with one hidden layer of size 2 and sigmoid activation function after the hidden layer
        - Network with one hidden layer of size 2 and ReLU activation function after the hidden layer
        - Network with two hidden layers (each with size 2)
            and Sigmoid, ReLU activation function after corresponding hidden layers
        - Network with two hidden layers (each with size 2)
            and ReLU, Sigmoid activation function after corresponding hidden layers
    NOTE: Each model should end with a Softmax layer due to CrossEntropyLossLayer requirement.

    Notes:
        - Try using learning rate between 1e-5 and 1e-3.
        - When choosing the number of epochs, consider effect of other hyperparameters on it.
            For example as learning rate gets smaller you will need more epochs to converge.
        - When searching over batch_size using powers of 2 (starting at around 32) is typically a good heuristic.
            Make sure it is not too big as you can end up with standard (or almost) gradient descent!

    Args:
        dataset_train (TensorDataset): Dataset for training.
        dataset_val (TensorDataset): Dataset for validation.

    Returns:
        Dict[str, Any]: Dictionary/Map containing history of training of all models.
            You are free to employ any structure of this dictionary, but we suggest the following:
            {
                name_of_model: {
                    "train": Per epoch losses of model on train set,
                    "val": Per epoch losses of model on validation set,
                    "model": Actual PyTorch model (type: nn.Module),
                }
            }
    """
    train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=32, shuffle=True)
    lr = 0.005

    model_history = dict()
    criterion = CrossEntropyLossLayer()

    linearlayer = nn.Sequential(LinearLayer(2, 2, generator=RNG), SoftmaxLayer())
    optimizer = SGDOptimizer(params=linearlayer.parameters(), lr=lr)
    train_result = train(train_loader, linearlayer, criterion, optimizer, val_loader, 100)
    model_history['linear'] = { "train": train_result["train"], "val": train_result["val"], "model": linearlayer}

    sigmoid_1hiddenlayer = nn.Sequential(
        LinearLayer(2,2, generator=RNG),
        SigmoidLayer(),
        LinearLayer(2,2, generator=RNG),
        SoftmaxLayer()
    )
    optimizer = SGDOptimizer(params=sigmoid_1hiddenlayer.parameters(), lr=lr)
    train_result = train(train_loader, sigmoid_1hiddenlayer, criterion, optimizer, val_loader, 100)
    model_history['sigmoid_1hidden'] = { "train": train_result["train"], "val": train_result["val"], "model": sigmoid_1hiddenlayer}

    relu_1hiddenlayer = nn.Sequential(LinearLayer(2,2, generator=RNG), ReLULayer(), LinearLayer(2,2, generator=RNG),
                                      SoftmaxLayer())
    optimizer = SGDOptimizer(params=relu_1hiddenlayer.parameters(), lr=lr)
    train_result = train(train_loader, relu_1hiddenlayer, criterion, optimizer, val_loader, 100)
    model_history['relu_1hidden'] = { "train": train_result["train"], "val": train_result["val"], "model": relu_1hiddenlayer}

    sigmoid_relu_2hiddenlayer = nn.Sequential(LinearLayer(2,2, generator=RNG), SigmoidLayer(), LinearLayer(2,2, generator=RNG),
                                              ReLULayer(), LinearLayer(2,2, generator=RNG), SoftmaxLayer())
    optimizer = SGDOptimizer(params=sigmoid_relu_2hiddenlayer.parameters(), lr=lr)
    train_result = train(train_loader, sigmoid_relu_2hiddenlayer, criterion, optimizer, val_loader, 100)
    model_history['sigmoid_relu_2hidden'] = { "train": train_result["train"], "val": train_result["val"], "model": sigmoid_relu_2hiddenlayer}

    relu_sigmoid_2hiddenlayer = nn.Sequential(LinearLayer(2,2, generator=RNG), ReLULayer(), LinearLayer(2,2, generator=RNG),
                                              SigmoidLayer(), LinearLayer(2,2, generator=RNG), SoftmaxLayer())
    optimizer = SGDOptimizer(params=relu_sigmoid_2hiddenlayer.parameters(), lr=lr)
    train_result = train(train_loader, relu_sigmoid_2hiddenlayer, criterion, optimizer, val_loader, 100)
    model_history['relu_sigmoid_2hidden'] = { "train": train_result["train"], "val": train_result["val"], "model": relu_sigmoid_2hiddenlayer}

    return model_history

@problem.tag("hw3-A")
def accuracy_score(model, dataloader) -> float:
    """Calculates accuracy of model on dataloader. Returns it as a fraction.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): Dataloader for CrossEntropy.
            Each example is a tuple consiting of (observation, target).
            Observation is a 2-d vector of floats.
            Target is an integer representing a correct class to a corresponding observation.

    Returns:
        float: Vanilla python float resprenting accuracy of the model on given dataset/dataloader.
            In range [0, 1].

    Note:
        - For a single-element tensor you can use .item() to cast it to a float.
        - This is similar to MSE accuracy_score function,
            but there will be differences due to slightly different targets in dataloaders.
    """
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            obs, target = data
            outputs = model(obs)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return correct / total


@problem.tag("hw3-A", start_line=7)
def main():
    """
    Main function of the Crossentropy problem.
    It should:
        1. Call crossentropy_parameter_search routine and get dictionary for each model architecture/configuration.
        2. Plot Train and Validation losses for each model all on single plot (it should be 10 lines total).
            x-axis should be epochs, y-axis should me Crossentropy loss, REMEMBER to add legend
        3. Choose and report the best model configuration based on validation losses.
            In particular you should choose a model that achieved the lowest validation loss at ANY point during the training.
        4. Plot best model guesses on test set (using plot_model_guesses function from train file)
        5. Report accuracy of the model on test set.

    Starter code loads dataset, converts it into PyTorch Datasets, and those into DataLoaders.
    You should use these dataloaders, for the best experience with PyTorch.
    """
    (x, y), (x_val, y_val), (x_test, y_test) = load_dataset("xor")

    dataset_train = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    dataset_val = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
    dataset_test = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))

    ce_configs = crossentropy_parameter_search(dataset_train, dataset_val)

    loss_min = float("inf")
    model_name_min = None
    model_min = None

    for config in ce_configs.items():
        x = range(100)
        train = config[1]['train']
        val = config[1]['val']

        model_name = config[0]
        model = config[1]["model"]

        plt.plot(x , train, label = model_name + "_train")
        plt.plot(x , val, label = model_name + "_val")
        loss_model = min(val)
        
        if loss_model < loss_min:
            loss_min = loss_model
            model_name_min = model_name
            model_min = model

    
    plt.ylabel("Crossentropy Loss")
    plt.xlabel("Epoch")  
    plt.legend()
    plt.show()

    print("Model with lowest loss: ", model_name_min)

    plot_model_guesses(DataLoader(dataset_test), model_min, model_name_min)
    acc = accuracy_score(model_min, DataLoader(dataset_test))
    print("Model Accuracy: ", acc)

if __name__ == "__main__":
    main()
