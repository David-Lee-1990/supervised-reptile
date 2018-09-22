import numpy as np
import torch
from torch import nn, autograd as ag
import matplotlib.pyplot as plt
from copy import deepcopy
from time import time
from termcolor import colored

seed = 0
plot = True
innerstepsize = 0.02 # stepsize in inner SGD
innerepochs = 1 # number of epochs of each inner SGD
outerstepsize0 = 0.1 # stepsize of outer optimization, i.e., meta-optimization
niterations = 100 # number of outer updates; each iteration we sample one task and update on it

rng = np.random.RandomState(seed)
torch.manual_seed(seed)

# Define task distribution
x_all = np.linspace(-5, 5, 50)[:,None] # All of the x points

def gen_task():
    "Generate classification problem"
    phase = rng.uniform(low=0, high=2*np.pi)
    ampl = rng.uniform(0.1, 5)
    f_randomsine = lambda x : np.sin(x + phase) * ampl
    return f_randomsine

# Define model. Reptile paper uses ReLU, but Tanh gives slightly better results
model = nn.Sequential(
    nn.Linear(1, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 1),
)

def totorch(x):
    return ag.Variable(torch.Tensor(x))

def train_on_batch(x, y):

    x = totorch(x)
    y = totorch(y)
    model.zero_grad()
    ypred = model(x)
    loss = (ypred - y).pow(2).mean()
    loss.backward()

    for param in model.parameters():
        param.data -= innerstepsize * param.grad.data

def predict(x):
    x = totorch(x)
    return model(x).data.numpy()

# Choose a fixed task and minibatch for visualization
f = gen_task()
xtrain = x_all[rng.choice(len(x_all), size=10)]
ytrain = f(xtrain)
plt.plot(x_all, f(x_all), label="true", color=(0, 1, 0))
plt.plot(xtrain, f(xtrain), "x", label="train", color="k")
plt.ylim(-4, 4)


# Reptile training loop
for iteration in range(niterations):
    train_on_batch(xtrain, ytrain)
    # Periodically plot the results on a particular task and minibatch
    if iteration == 0 or (iteration+1) % 50 == 0:
        # plt.cla()
        plt.plot(x_all, predict(x_all), label="pred after %d iters"%iteration,
                 color=(iteration/niterations,0,1-iteration/niterations))
        # plt.pause(0.01)
plt.legend(loc="upper left")
plt.show()