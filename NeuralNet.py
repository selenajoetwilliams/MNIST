# this class will create the neural network
# link to pytorch tutorial: https://pytorch.org/tutorials/beginner/basics/intro.html

import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

training_data = datasets.Fashion.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = dataset.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)



# question: these tensors are blank right? 
# question: what about loading the acutal data?
# question: this just creates 2 blank 2-D arrays, right? one 1x2 array and another 3x4 array?
data = [[1,2],[3,4]]
x+data = torch.tensor(data)
