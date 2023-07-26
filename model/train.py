import numpy as np
import torch
import wandb
import optuna

def train_loop(dataloader, model, loss_fn, optimizer, epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    last_step = None
    avg_loss = 0
    for batch, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.float().to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        if model.gradient_clipping:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=.1)
        optimizer.step()
        optimizer.zero_grad()
        step = epoch * len(dataloader.dataset) + batch * len(x)
        last_step = step
        loss_item = loss.item()
        avg_loss += loss_item
        if np.isnan(loss_item):
            wandb.finish()
            raise optuna.exceptions.TrialPruned()
        wandb.log({"train_loss": loss.item(),
                   "step": step,
                   "epoch": epoch,
                   })
    avg_loss /= len(dataloader)
    return last_step, avg_loss
