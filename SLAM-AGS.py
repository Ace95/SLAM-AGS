import torch
import torchvision.transforms as T
from utils import nt_xent_loss, similarity_loss, BalancedBatchDataset
from model import Model
import os 
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import wandb
from torch.optim.lr_scheduler import LambdaLR
import math
import argparse
import random

# Argument parser for directories
parser = argparse.ArgumentParser(description="Pretrain with SLAM-AGS")
parser.add_argument("--positive_dir", type=str, required=True, help="Path to positive images")
parser.add_argument("--negative_dir", type=str, required=True, help="Path to negative images")
parser.add_argument("--wr", type=str, required=True, help="WR of the training set")
args = parser.parse_args()

checkpoint_path = "./checkpoints"
os.makedirs(checkpoint_path, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 150
batch_size = 256
learning_rate = 0.001
warmup_epochs = 10

mean = (0.6009, 0.5513, 0.7394)
std = (0.1982, 0.2448, 0.1076)
normalize = T.Normalize(mean=mean, std=std)

# Data preprocessing
basic_transform = T.Compose([
    T.CenterCrop(224),
    T.ToTensor(),
    normalize
])

neg_transform = T.Compose([
    T.RandomCrop(224),
    T.RandomHorizontalFlip(),
    T.RandomApply([T.ColorJitter(0.1, 0.1, 0.1, 0.1)], p=0.5),
    T.RandomApply([T.Lambda(lambda x: T.functional.rotate(x, 90))], p=0.5),
    T.ToTensor(),
    normalize
])

simclr_transform = T.Compose([
    T.RandomCrop(224),
    T.RandomHorizontalFlip(),
    T.RandomApply([T.ColorJitter(0.1, 0.1, 0.1, 0.1)], p=0.8),
    T.RandomApply([T.Lambda(lambda x: T.functional.rotate(x, 90))], p=0.8),
    T.ToTensor(),
    normalize
])

dataset = BalancedBatchDataset(
    positive_dir= args.positive_dir, 
    negative_dir= args.negative_dir,
    transform_pos=simclr_transform,
    transform_neg=basic_transform,
    batch_size = batch_size
)

loader = DataLoader(dataset, batch_size=None, num_workers=4)

# Load ResNet18 with projection head
model = Model()
model = model.to(device)

wandb.init(
    project="SLAM-AGS",    
    name=f"SLAM-AGS_{args.wr}",   
    config={
        "epochs": num_epochs,
        "batch_size": loader.batch_size,
        "model": "resnet18"
    }
)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
#optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

scheduler = LambdaLR(
    optimizer,
    lr_lambda=lambda epoch: (
        (epoch + 1) / warmup_epochs if epoch < warmup_epochs
        else 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs)))
    )
)

def train(model, loader, optimizer, epochs):
    model.train()
    loss_history = []
    eps = 1e-10

    def _gather_grads_flat(params):
        """Return a single 1D tensor which is the concatenation of grads for the provided params.
           If a param has no grad, use zeros of the correct shape."""
        flats = []
        for p in params:
            if p.grad is None:
                flats.append(torch.zeros_like(p).view(-1))
            else:
                flats.append(p.grad.detach().clone().view(-1))
        return torch.cat(flats, dim=0)

    def _set_grads_from_flat(params, flat):
        grads = []
        idx = 0
        for p in params:
            numel = p.numel()
            chunk = flat[idx: idx + numel].view_as(p).contiguous()
            # set p.grad (must be a tensor with same device/dtype)
            if p.grad is None:
                p.grad = chunk.clone()
            else:
                p.grad.copy_(chunk)
            grads.append(chunk.clone())
            idx += numel
        return grads

    params = [p for p in model.parameters() if p.requires_grad]

    for epoch in range(epochs):
        total_loss = 0.0
        total_loss_simclr = 0.0
        total_loss_sim = 0.0
        num_batches = 0

        for pos_i, pos_j, neg_i, neg_j in loader:
            pos_i, pos_j, neg_i, neg_j = pos_i.to(device), pos_j.to(device), neg_i.to(device), neg_j.to(device)

            z_pos_i = model(pos_i)
            z_pos_j = model(pos_j)
            z_neg_i = model(neg_i)
            z_neg_j = model(neg_j)
            z_neg = torch.stack([z_neg_i, z_neg_j], dim=1)

            loss_simclr = nt_xent_loss(z_pos_i, z_pos_j)
            loss_sim = similarity_loss(z_neg)

            # Compute per-task gradients
            optimizer.zero_grad()
            loss_simclr.backward(retain_graph=True)
            g1_flat = _gather_grads_flat(params)
            optimizer.zero_grad()
            loss_sim.backward(retain_graph=False)
            g2_flat = _gather_grads_flat(params)

            
            # Apply Adaptive PCGrad
            grads = [g1_flat, g2_flat]
            g_pc = [grads[0].clone(), grads[1].clone()]
            task_order = list(range(len(grads)))
            random.shuffle(task_order)

            # Compute pre-PCGrad total gradient (for rescaling)
            g_sum = grads[0] + grads[1]
            pre_norm = g_sum.norm()

            # Perform pairwise projection
            for i in task_order:
                for j in task_order:
                    if i == j:
                        continue
                    gi = g_pc[i]
                    gj = grads[j]  # using original gj
                    gj_norm2 = (gj * gj).sum()
                    if gj_norm2.item() == 0:
                        continue
                    dot = (gi * gj).sum()
                    if dot < 0:
                        g_pc[i] = gi - (dot / (gj_norm2 + eps)) * gj

            # Combine and rescale projected gradient
            g_total = g_pc[0] + g_pc[1]
            post_norm = g_total.norm()
            if post_norm.item() > 0:
                scale = pre_norm / (post_norm + eps)
                g_total = g_total * scale

            # Put the combined grad back into model.params 
            optimizer.zero_grad()
            _set_grads_from_flat(params, g_total)

            optimizer.step()

            total_loss += (loss_simclr.item() + loss_sim.item())
            total_loss_simclr += loss_simclr.item()
            total_loss_sim += loss_sim.item()
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        avg_loss_simclr = total_loss_simclr / max(1, num_batches)
        avg_loss_sim = total_loss_sim / max(1, num_batches)

        loss_history.append(avg_loss)
        print(f"Epoch {epoch + 1}, Avg Total Loss: {avg_loss:.4f}, Avg SimCLR Loss: {avg_loss_simclr:.4f}, Avg Similarity Loss: {avg_loss_sim:.4f}")
        wandb.log({
            "avg_total_loss": avg_loss,
            "avg_simclr_loss": avg_loss_simclr,
            "avg_similarity_loss": avg_loss_sim,
            "lr": scheduler.get_last_lr()[0]
        })
        scheduler.step()

    torch.save(model.state_dict(), 'weaksupcon_'+args.wr+'.pth')
    print('Model saved to weaksupcon_'+args.wr+'.pth')
    
    return loss_history

loss_history = train(model, loader, optimizer, epochs=num_epochs)
wandb.finish()