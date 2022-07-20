
# Homecoming (eYRC-2018): Task 1A
# Build a Fully Connected 2-Layer Neural Network to Classify Digits

from nnet import model
import random

# We will use torchvision's transforms and datasets
import torch
from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler

# The output of torchvision datasets are PILImage images of range [0, 1]. 
# So we are transforming them to Tensors and normalize them with mean = 0.1307 and std = 0.3081 which are the mean and std deviation of the MNIST dataset.
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=2)

test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False, num_workers=2)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Layer size
N_in = 28 * 28 # Input size
N_h1 = 256 # Hidden Layer 1 size
N_h2 = 256 # Hidden Layer 2 size
N_out = 10 # Output size
# Learning rate
lr = 0.001

# initializing model
net = model.FullyConnected(N_in, N_h1, N_h2, N_out, device=device)

N_epoch = 5 
batch_size = 4

# Training and Validation Loop
for n in range(N_epoch):
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        creloss, accuracy, outputs = net.train(inputs, labels)
    net.accuracy_reset()
    for i, data in enumerate(test_loader, 0):
        inputs, labels = data
        creloss, accuracy, outputs = net.eval(inputs, labels)
    net.accuracy_reset()

# making predictions on test examples
sampler = SubsetRandomSampler(list(range(i*batch_size, (i+1)*batch_size)))
test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, sampler=sampler, num_workers=2)

print("\nPredictions on a test batch: ")
for i, data in enumerate(test_loader, 0):
    inputs, labels = data
    score, idx = net.predict(inputs)
    print("actual digits = ", labels, "\tpredicted digits = ", idx)
print("\n")