#!/usr/bin/env python
# coding: utf-8

# In[4]:


import torch
from torch import nn
import numpy as np

from typing import Tuple, Union, List, Callable
from torch.optim import SGD
import torchvision
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

# use M1 macbook pro silicon gpu

print(f"PyTorch version: {torch.__version__}")

# Check PyTorch has access to MPS (Metal Performance Shader, Apple's GPU architecture)
print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
print(f"Is MPS available? {torch.backends.mps.is_available()}")

# Set the device      
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")




# In[5]:


train_dataset = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=torchvision.transforms.ToTensor())
test_dataset = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=torchvision.transforms.ToTensor())


# In[6]:


batch_size = 128

train_dataset, val_dataset = random_split(train_dataset, [int(0.9 * len(train_dataset)), int( 0.1 * len(train_dataset))])

# Create separate dataloaders for the train, test, and validation set
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=True
)


# In[7]:


imgs, labels = next(iter(train_loader))
print(f"A single batch of images has shape: {imgs.size()}")
example_image, example_label = imgs[0], labels[0]
c, w, h = example_image.size()
print(f"A single RGB image has {c} channels, width {w}, and height {h}.")

# This is one way to flatten our images
batch_flat_view = imgs.view(-1, c * w * h)
print(f"Size of a batch of images flattened with view: {batch_flat_view.size()}")

# This is another equivalent way
batch_flat_flatten = imgs.flatten(1)
print(f"Size of a batch of images flattened with flatten: {batch_flat_flatten.size()}")

# The new dimension is just the product of the ones we flattened
d = example_image.flatten().size()[0]
print(c * w * h == d)

# View the image
t =  torchvision.transforms.ToPILImage()
plt.imshow(t(example_image))

# These are what the class labels in CIFAR-10 represent. For more information,
# visit https://www.cs.toronto.edu/~kriz/cifar.html
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog",
           "horse", "ship", "truck"]
print(f"This image is labeled as class {classes[example_label]}")


# In[8]:


# logistic regression from example
def linear_model() -> nn.Module:
    """Instantiate a linear model and send it to device."""
    model =  nn.Sequential(
            nn.Flatten(),
            nn.Linear(d, 10)
         )
    return model.to(DEVICE)

# Define linear model from hw 
def linear_1hidden(m) -> nn.Module:
    
    model = nn.Sequential(nn.Flatten(), nn.Linear(32*32*3, m), nn.ReLU(), nn.Linear(m, 10))

    return model.to(DEVICE) 

# Define CNN model
def CNN(m, k, N) -> nn.Module:

    model = nn.Sequential(nn.Conv2d(3, m, k), nn.ReLU(), nn.MaxPool2d(N), nn.Flatten(),
                        nn.Linear(m * (int((33 - k )/ N)) ** 2, 10))
    
    # x_out = W2(maxpool(relu(conv2d(x, W1)+b1))) + b2 where W2 \in \mathbb 10 * M(\roundup(33-k/N))**2 
  
    return model.to(DEVICE) 



# In[9]:


def train(
    model: nn.Module, optimizer: SGD,
    train_loader: DataLoader, val_loader: DataLoader,
    epochs: int = 20
)-> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Trains a model for the specified number of epochs using the loaders.

    Returns:
    Lists of training loss, training accuracy, validation loss, validation accuracy for each epoch.
    """

    loss = nn.CrossEntropyLoss()
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    for e in tqdm(range(epochs)):
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        # Main training loop; iterate over train_loader. The loop
        # terminates when the train loader finishes iterating, which is one epoch.
        for (x_batch, labels) in train_loader:
            x_batch, labels = x_batch.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            labels_pred = model(x_batch)
            batch_loss = loss(labels_pred, labels)
            train_loss = train_loss + batch_loss.item()

            labels_pred_max = torch.argmax(labels_pred, 1)
            batch_acc = torch.sum(labels_pred_max == labels)
            train_acc = train_acc + batch_acc.item()

            batch_loss.backward()
            optimizer.step()
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_acc / (batch_size * len(train_loader)))

        # Validation loop; use .no_grad() context manager to save memory.
        model.eval()
        val_loss = 0.0
        val_acc = 0.0

        with torch.no_grad():
            for (v_batch, labels) in val_loader:
                v_batch, labels = v_batch.to(DEVICE), labels.to(DEVICE)
                labels_pred = model(v_batch)
                v_batch_loss = loss(labels_pred, labels)
                val_loss = val_loss + v_batch_loss.item()

                v_pred_max = torch.argmax(labels_pred, 1)
                batch_acc = torch.sum(v_pred_max == labels)
                val_acc = val_acc + batch_acc.item()
            val_losses.append(val_loss / len(val_loader))
            val_accuracies.append(val_acc / (batch_size * len(val_loader)))

    return train_losses, train_accuracies, val_losses, val_accuracies


# In[11]:


def parameter_search(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_fn:Callable[[], nn.Module]
) -> float:
    """
    Parameter search for our linear model using SGD.

    Args:
    train_loader: the train dataloader.
    val_loader: the validation dataloader.
    model_fn: a function that, when called, returns a torch.nn.Module.

    Returns:
    The learning rate with the least validation loss.
    NOTE: you may need to modify this function to search over and return
     other parameters beyond learning rate.
    """
    num_iter = 10
    best_loss = torch.tensor(np.inf)
    best_lr = 0.0
    
    best_M = 1.0
    best_N = 1.0
    best_k = 1.0
    
    Ms = np.arange(300, 500, 10) 
    Ns = np.arange(4,12,2) 
    ks = np.arange(3,8,1)
    
    lrs = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1] # cnn model
    
    #lrs = torch.linspace(10 ** (-3), 10 ** (-1), 10) # linear model

    #lrs = torch.linspace(10 ** (-6), 10 ** (-1), num_iter) # example 

    for lr in lrs:
        for M in Ms:
            for k in ks:
                for N in Ns:
                    print(f"trying learning rate {lr}")
                    print(f"trying m value {M}")
                    print(f"trying N value {N}")
                    print(f"trying k value {k}")

                    model = model_fn(M, k, N)
                    optim = SGD(model.parameters(), lr)

                    train_loss, train_acc, val_loss, val_acc = train(
                        model,
                        optim,
                        train_loader,
                        val_loader,
                        epochs=3
                        )

                    if min(val_loss) < best_loss:
                        best_loss = min(val_loss)
                        best_lr = lr
                        best_M = M
                        best_N = N
                        best_k = k
                        print("Val Accuracy: ", val_acc)
                    
            
    return best_lr, best_M, best_N, best_k


# In[12]:


best_lr, best_M, best_N, best_k = parameter_search(train_loader, val_loader, CNN)


# In[68]:


num_epoch = 50
#model = linear_model()
#model = linear_1hidden(best_M)
best_lr = 0.05; best_M = 480; best_k = 5; best_N = 4; 
model = CNN(best_M, best_k, best_N)
optimizer = SGD(model.parameters(), best_lr)
print(best_lr, best_M, best_k, best_N)


# In[69]:


# We are only using 20 epochs for this example. You may have to use more.
train_loss, train_accuracy, val_loss, val_accuracy = train(
    model, optimizer, train_loader, val_loader, num_epoch
)


# In[70]:


epochs = range(1, num_epoch + 1)
plt.plot(epochs, train_accuracy, label="Train Accuracy")
plt.plot(epochs, val_accuracy, label="Validation Accuracy")
plt.axhline(y=0.65, linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("CNN Model Accuracy for CIFAR-10 vs Epoch")
plt.show()


# In[71]:


def evaluate(
    model: nn.Module, loader: DataLoader
) -> Tuple[float, float]:
    """Computes test loss and accuracy of model on loader."""
    loss = nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    with torch.no_grad():
        for (batch, labels) in loader:
            batch, labels = batch.to(DEVICE), labels.to(DEVICE)
            y_batch_pred = model(batch)
            batch_loss = loss(y_batch_pred, labels)
            test_loss = test_loss + batch_loss.item()

            pred_max = torch.argmax(y_batch_pred, 1)
            batch_acc = torch.sum(pred_max == labels)
            test_acc = test_acc + batch_acc.item()
        test_loss = test_loss / len(loader)
        test_acc = test_acc / (batch_size * len(loader))
        return test_loss, test_acc


# In[72]:


test_loss, test_acc = evaluate(model, test_loader)
print(f"CNN Model Test Accuracy: {test_acc}")


# In[52]:


lrs = torch.linspace(10 ** (-3), 10 ** (-1), 10)
print(lrs)


# In[ ]:




