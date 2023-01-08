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

X = torch.rand(1, 28, 28, device=device)
logits = model(X) 
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

# Model Layers

# sampling a mini batch to see what happens as it goes through the model
input_image = torch.rand(3, 28, 28)
print(input_image.size()) 

# flattening the dimensions from 28x28 2D array to a 1D array of 784 pixels
flatten = nn.Flatten() 
flat_image = flatten(input_image)
print(flat_image.size())

# linear layer
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())

# relu layer 
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}") 
# question: is the hidden layer meant to improve the assumptions, as seen in smaller abs vals of the values in each index of the tensor?

# sequential layer
# TODO: look up seq_modules in pytorch documentation
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20,10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)

# softmax layer 
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)





# question: these tensors are blank right? 
# question: what about loading the acutal data?
# question: this just creates 2 blank 2-D arrays, right? one 1x2 array and another 3x4 array?
# data = [[1,2],[3,4]]
# x_data = torch.tensor(data)
