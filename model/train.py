import torch
import wandb

def train_loop(dataloader, model, loss_fn, optimizer, epoch):
    size = len(dataloader.dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        n_samples = (batch+1 * len(X)) + (len(X) * len(dataloader) * epoch)
        X = X.to(device)
        y = y.float().to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        if model.gradient_clipping:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=.01)
        optimizer.step()
        optimizer.zero_grad()
        wandb.log({"train_loss": loss.item(),
                   "n_samples": n_samples
                   })
