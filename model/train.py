import torch
from torch.cuda.amp import autocast, GradScaler
import wandb

def train_loop(dataloader, model, loss_fn, optimizer, epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = GradScaler()
    model.train()
    last_step = None
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.float().to(device)

        # Wrap the forward pass in the autocast context manager for mixed precision
        with autocast():
            pred = model(X)
            loss = loss_fn(pred, y)

        # Scales the loss, and calls backward() to create scaled gradients
        scaler.scale(loss).backward()

        if model.gradient_clipping:
            # You also need to modify the gradient clipping to work with the scaled gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=.01)

        # Unscales gradients and calls or skips optimizer.step()
        scaler.step(optimizer)

        # Updates the scale for next iteration
        scaler.update()

        optimizer.zero_grad()
        step = epoch * len(dataloader.dataset) + batch * len(X)
        last_step = step
        wandb.log({"train_loss": loss.item(),
                   "step": step
                   })
    return last_step
