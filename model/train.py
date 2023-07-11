import torch.nn as nn
import torch.nn.functional as F
import torch
import wandb

def train_loop(dataloader, model, loss_fn, optimizer, epoch):
    size = len(dataloader.dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        step = batch * len(X) + len(X) * len(dataloader) * epoch
        X = X.to(device)
        y = y.float().to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=.01)
        optimizer.step()
        optimizer.zero_grad()
        wandb.log({"train_loss": loss.item(),
                   "step": step
                   })

        if step % 10000 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
