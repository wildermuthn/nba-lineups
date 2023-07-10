import torch.nn as nn
import torch.nn.functional as F
import torch

def train(model):
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    print("Training model...")

    pred = torch.randn(4, 10)
    target = torch.randn(4, 10)
    loss = loss_fn(pred, target)
    print('Total loss for this batch: {}'.format(loss.item()))