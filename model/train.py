import torch
import wandb


def train_loop(dataloader, model, loss_fn, optimizer, epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    last_step = None
    for batch, (xX, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.float().to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        if model.gradient_clipping:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=.01)
        optimizer.step()
        optimizer.zero_grad()
        step = epoch * len(dataloader.dataset) + batch * len(X)
        last_step = step
        wandb.log({"train_loss": loss.item(),
                   "step": step
                   })
    return last_step
