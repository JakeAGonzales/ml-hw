import math
from typing import List
import matplotlib.pyplot as plt
import torch
from torch.distributions import Uniform
from torch.nn import Module
from torch.nn.functional import cross_entropy, relu
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from utils import load_dataset, problem

class F1(Module):
    def __init__(self, h: int, d: int, k: int):
        super().__init__()
        self.alpha0 = 1 / np.sqrt(d)
        self.alpha1 = 1 / np.sqrt(h)
        self.w0 = torch.nn.Parameter(Uniform(-self.alpha0, self.alpha0).sample(sample_shape=torch.Size([h, d])))
        self.w1 = torch.nn.Parameter(Uniform(-self.alpha1, self.alpha1).sample(sample_shape=torch.Size([k, h])))
        self.b0 = torch.nn.Parameter(Uniform(-self.alpha0, self.alpha0).sample(sample_shape=torch.Size([1, h])))
        self.b1 = torch.nn.Parameter(Uniform(-self.alpha1, self.alpha1).sample(sample_shape=torch.Size([1, k])))
        
        self.params = [self.w0, self.w1, self.b0, self.b1] 

        for param in self.params:
          param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        n,_ = x.shape
        b0 = self.b0.repeat(n,1)
        b1 = self.b1.repeat(n,1)
        x = torch.matmul(x, self.w0.T) + b0
        x = torch.nn.functional.relu(x)
        x = torch.matmul(x, self.w1.T) + b1

        return x
        
class F2(Module):
    def __init__(self, h0: int, h1: int, d: int, k: int):
        super().__init__()
        self.alpha0 = 1 / np.sqrt(d)
        self.alpha1 = 1 / np.sqrt(h0)
        self.alpha2 = 1 / np.sqrt(h1)
        self.w0 = torch.nn.Parameter(Uniform(-self.alpha0, self.alpha0).sample(sample_shape=torch.Size([h0, d])))
        self.w1 = torch.nn.Parameter(Uniform(-self.alpha1, self.alpha1).sample(sample_shape=torch.Size([h1, h0])))
        self.w2 = torch.nn.Parameter(Uniform(-self.alpha2, self.alpha2).sample(sample_shape=torch.Size([k, h1])))
        self.b0 = torch.nn.Parameter(Uniform(-self.alpha0, self.alpha0).sample(sample_shape=torch.Size([1, h0])))
        self.b1 = torch.nn.Parameter(Uniform(-self.alpha1, self.alpha1).sample(sample_shape=torch.Size([1, h1])))
        self.b2 = torch.nn.Parameter(Uniform(-self.alpha2, self.alpha2).sample(sample_shape=torch.Size([1, k])))
        
        self.params = [self.w0, self.w1, self.w2, self.b0, self.b1, self.b2] 

        for param in self.params:
          param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        b0 = self.b0.repeat(x.shape[0],1)
        x = torch.matmul(x, self.w0.T) + b0
        x = torch.nn.functional.relu(x)
        b1 = self.b1.repeat(x.shape[0],1)
        x = torch.matmul(x, self.w1.T) + b1
        x = torch.nn.functional.relu(x)
        b2 = self.b2.repeat(x.shape[0],1)
        x = torch.matmul(x, self.w2.T) + b2

        return x

def train(model: Module, optimizer: Adam, train_loader: DataLoader) -> List[float]:
    epochs = 32
    loss_list = []
    accuracy_list = []
    for i in range(epochs):
        loss = 0
        acc = 0
        for images, labels in train_loader:
            x, y = images, labels
            y_predictions = model.forward(x) 
            loss_curr = cross_entropy(y_predictions, y)

            optimizer.zero_grad()
            loss_curr.backward()
            optimizer.step()

            predictions = torch.argmax(y_predictions, 1)
            acc += torch.sum(predictions == y)/len(predictions)
            loss += loss_curr.item()

        acc = acc / len(train_loader)
        print("Current Accuracy: ", acc)

        if acc > 0.99:
            break
        loss_list.append(loss/len(train_loader))
        accuracy_list.append(acc)

    return loss_list

def main():
    (x, y), (x_test, y_test) = load_dataset("mnist")
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()
    
    # Part 1 F1
    model = F1(h = 64, d = 784, k = 10)
    optimizer = Adam(model.params, lr = 5e-3)
    train_loader = DataLoader(TensorDataset(x,y), batch_size = 64, shuffle=True)

    loss_list = train(model, optimizer, train_loader)
    y_hat = model(x_test)
    test_predictions = torch.argmax(y_hat,1)
    accuracy = torch.sum(test_predictions == y_test)/len(test_predictions)
    print("F1 Final Accuracy: ", accuracy)
    test_loss = cross_entropy(y_hat, y_test).item()
    print("\n F2 Test Loss: ", test_loss)
    plt.plot(np.arange(len(loss_list)), loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    all_params = sum(param.numel() for param in model.parameters())
    print("\n Total Parameters in the Model: ", all_params)

    # Part 2 F2
    model = F2(h0 = 32, h1 = 32, d = 784, k = 10)
    optimizer = Adam(model.params, lr = 5e-3)
    train_loader = DataLoader(TensorDataset(x,y), batch_size = 64, shuffle=True)
    loss_list = train(model, optimizer, train_loader)
    y_hat = model(x_test)
    test_predictions = torch.argmax(y_hat,1)
    accuracy = torch.sum(test_predictions == y_test)/len(test_predictions)
    print("F2 Final Accuracy: ", accuracy)
    test_loss = cross_entropy(y_hat, y_test).item()
    print("\n F2 Test Loss: ", test_loss)
    plt.plot(np.arange(len(loss_list)), loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    all_params = sum(param.numel() for param in model.parameters())
    print("\n Total Parameters in the Model: ", all_params)

if __name__ == "__main__":
    main()