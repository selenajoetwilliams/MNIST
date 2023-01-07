# this class will create the neural network
# link to pytorch tutorial: https://pytorch.org/tutorials/beginner/basics/intro.html

import os
import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3

# question: can someone walk me through what each line this block of code means? 
for i in range(1, cols * rows + 1): # for each item in the 2D array
    sample_idx = torch.randint(len(training_data), size=(1,)).item() # sample a random index
    img, label = training_data[sample_idx] # set the image and label to = the image and label of the data at the randomly sampled index
    figure.add_subplot(rows, cols, i) # add a subplot?
    plt.title(labels_map[label]) # title the plot?
    plt.axis("off") # axis formatting
    plt.imshow(img.squeeze(),cmap="gray") # image formatting
plt.show() # show it!

###########################################
# TRANSFORMS 

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

###########################################
# BUILD MODEL

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device") # note: I am still using cpu 

class NeuralNetwork(nn.Module):

    # initializing the model
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            # these are the different model layers
            nn.Linear(28*28, 512), # image dimensions, # of model dimentions?
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    # defining the forward pass
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)



# question: these tensors are blank right? 
# question: what about loading the acutal data?
# question: this just creates 2 blank 2-D arrays, right? one 1x2 array and another 3x4 array?
# data = [[1,2],[3,4]]
# x_data = torch.tensor(data)
