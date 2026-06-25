"""
Assignment 5 (Bonus) - Image Generation Using Diffusion Models
Name: Afzaal
Roll No: MSDS25062

This script trains a denoising diffusion model on a subset of the animal
dataset and generates new images from pure noise.

Usage:
    python MSDS25062_05.py --dataset_path /path/to/animal_data --epochs 10

This is the same implementation that was originally developed and run on
Google Colab, restructured here to run from the command line.
"""

import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Command line arguments
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Diffusion Model - Image Generation (Animal Dataset)')
    parser.add_argument('--dataset_path', type=str, required=True,
                         help='Path to the animal_data folder containing class subfolders')
    parser.add_argument('--classes', type=str, nargs='+',
                         default=['Bear', 'Cat', 'Dog', 'Lion', 'Tiger'],
                         help='Animal classes to use for training (default: 5 classes)')
    parser.add_argument('--num_images_per_class', type=int, default=20,
                         help='Number of images to use per class (default: 20)')
    parser.add_argument('--img_size', type=int, default=64, help='Image size (default: 64)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size (default: 8)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate (default: 0.0001)')
    parser.add_argument('--num_steps', type=int, default=1000, help='Diffusion timesteps T (default: 1000)')
    parser.add_argument('--output_dir', type=str, default='outputs',
                         help='Directory to save plots and generated samples')
    parser.add_argument('--save_dir', type=str, default='saved_models',
                         help='Directory to save trained model checkpoints')
    return parser.parse_args()


args = parse_args()

os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(args.save_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# ---------------------------------------------------------------------------
# Step 1: Data Loader
# ---------------------------------------------------------------------------
class AnimalDataset(Dataset):
    def __init__(self, root_dir, classes, num_images_per_class=20, img_size=64):
        self.root_dir = root_dir
        self.img_size = img_size
        self.images = []

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        for animal_class in classes:
            class_dir = os.path.join(root_dir, animal_class)
            if not os.path.exists(class_dir):
                continue

            image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            selected_images = image_files[:num_images_per_class]

            for img_file in selected_images:
                self.images.append(os.path.join(class_dir, img_file))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            return image
        except Exception:
            return torch.randn(3, self.img_size, self.img_size)


dataset = AnimalDataset(
    args.dataset_path,
    args.classes,
    num_images_per_class=args.num_images_per_class,
    img_size=args.img_size
)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

print(f'Dataset size: {len(dataset)}')
print(f'Number of batches: {len(dataloader)}')


# ---------------------------------------------------------------------------
# Step 2: Forward diffusion process (noise schedule + forward step)
# ---------------------------------------------------------------------------
class DiffusionSchedule:
    def __init__(self, num_steps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_steps = num_steps
        self.betas = torch.linspace(beta_start, beta_end, num_steps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)


def forward_diffusion_step(x0, t, alphas_cumprod_t, sqrt_one_minus_alphas_cumprod_t):
    """Adds Gaussian noise to a clean image x0 at timestep t (closed-form q(x_t | x_0))."""
    noise = torch.randn_like(x0)
    sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod_t[t]).view(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t[t].view(-1, 1, 1, 1)

    x_t = sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise
    return x_t, noise


diffusion_schedule = DiffusionSchedule(num_steps=args.num_steps)
diffusion_schedule.betas = diffusion_schedule.betas.to(device)
diffusion_schedule.alphas = diffusion_schedule.alphas.to(device)
diffusion_schedule.alphas_cumprod = diffusion_schedule.alphas_cumprod.to(device)
diffusion_schedule.sqrt_alphas_cumprod = diffusion_schedule.sqrt_alphas_cumprod.to(device)
diffusion_schedule.sqrt_one_minus_alphas_cumprod = diffusion_schedule.sqrt_one_minus_alphas_cumprod.to(device)


# ---------------------------------------------------------------------------
# Step 3: Denoising model (U-Net style encoder-decoder with skip connections)
# ---------------------------------------------------------------------------
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -(np.log(10000) / (half_dim - 1)))
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class DenoisingUNet(nn.Module):
    def __init__(self, img_channels=3, base_channels=64):
        super().__init__()
        self.time_embed = TimeEmbedding(128)

        self.initial_conv = nn.Conv2d(img_channels, base_channels, 3, padding=1)

        self.down1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)

        self.middle = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels * 8, base_channels * 4, 3, padding=1),
            nn.ReLU()
        )

        self.up2 = nn.Sequential(
            nn.Conv2d(base_channels * 4 + base_channels * 4, base_channels * 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1),
            nn.ReLU()
        )
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.up1 = nn.Sequential(
            nn.Conv2d(base_channels * 2 + base_channels * 2, base_channels * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            nn.ReLU()
        )
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.final_conv = nn.Conv2d(base_channels, img_channels, 3, padding=1)

    def forward(self, x, t):
        skip1 = self.down1(self.initial_conv(x))
        x = self.pool1(skip1)

        skip2 = self.down2(x)
        x = self.pool2(skip2)

        x = self.middle(x)

        x = self.upsample2(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.up2(x)

        x = self.upsample1(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.up1(x)

        x = self.final_conv(x)
        return x


model = DenoisingUNet(img_channels=3, base_channels=64).to(device)
print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')


# ---------------------------------------------------------------------------
# Step 4: Custom loss function (MSE between predicted noise and true noise)
# ---------------------------------------------------------------------------
def custom_diffusion_loss(model, x0, schedule, device):
    batch_size = x0.shape[0]
    t = torch.randint(0, schedule.num_steps, (batch_size,), device=device)

    x_t, noise = forward_diffusion_step(
        x0, t,
        schedule.alphas_cumprod,
        schedule.sqrt_one_minus_alphas_cumprod
    )

    predicted_noise = model(x_t, t)

    loss = nn.functional.mse_loss(predicted_noise, noise)
    return loss


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
optimizer = optim.Adam(model.parameters(), lr=args.lr)
losses = []

for epoch in range(args.epochs):
    epoch_loss = 0.0
    pbar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{args.epochs}')

    for batch in pbar:
        x0 = batch.to(device)

        loss = custom_diffusion_loss(model, x0, diffusion_schedule, device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    avg_loss = epoch_loss / len(dataloader)
    losses.append(avg_loss)
    print(f'Epoch {epoch + 1}, Avg Loss: {avg_loss:.6f}')

plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.savefig(os.path.join(args.output_dir, 'training_loss.png'), dpi=150, bbox_inches='tight')
plt.close()

torch.save(model.state_dict(), os.path.join(args.save_dir, 'diffusion_model.pth'))
print('Model saved!')


# ---------------------------------------------------------------------------
# Step 5 & 6: Sampling function (reverse process) + visible results
# ---------------------------------------------------------------------------
@torch.no_grad()
def sample(model, schedule, num_samples=4, img_size=64):
    model.eval()
    x_t = torch.randn(num_samples, 3, img_size, img_size, device=device)

    for t in tqdm(reversed(range(schedule.num_steps)), total=schedule.num_steps):
        t_tensor = torch.full((num_samples,), t, dtype=torch.long, device=device)

        predicted_noise = model(x_t, t_tensor)

        alpha_t = schedule.alphas[t].view(1, 1, 1, 1)
        alpha_cumprod_t = schedule.alphas_cumprod[t].view(1, 1, 1, 1)
        variance = schedule.betas[t].view(1, 1, 1, 1)

        x_t = (1.0 / torch.sqrt(alpha_t)) * (x_t - (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_cumprod_t) * predicted_noise)

        if t > 0:
            x_t += torch.sqrt(variance) * torch.randn_like(x_t)

    return x_t


def denormalize(x):
    return (x + 1) / 2


samples = sample(model, diffusion_schedule, num_samples=4, img_size=args.img_size)

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    img = denormalize(samples[i].cpu().detach())
    img = img.permute(1, 2, 0).numpy()
    ax.imshow(np.clip(img, 0, 1))
    ax.set_title(f'Generated Sample {i + 1}')
    ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, 'generated_samples.png'), dpi=150, bbox_inches='tight')
plt.close()


# ---------------------------------------------------------------------------
# Visualize the forward noising process (Figure 1 style: image -> noise)
# ---------------------------------------------------------------------------
test_img = next(iter(dataloader))
test_img = test_img[:1].to(device)

steps_to_visualize = [0, 100, 250, 500, 750, args.num_steps - 1]
fig, axes = plt.subplots(1, 6, figsize=(15, 3))

for idx, step in enumerate(steps_to_visualize):
    t = torch.tensor([step], device=device)
    x_t, _ = forward_diffusion_step(
        test_img, t,
        diffusion_schedule.alphas_cumprod,
        diffusion_schedule.sqrt_one_minus_alphas_cumprod
    )

    img = denormalize(x_t[0].cpu().detach())
    img = img.permute(1, 2, 0).numpy()

    axes[idx].imshow(np.clip(img, 0, 1))
    axes[idx].set_title(f'Step {step}')
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, 'noise_addition_steps.png'), dpi=150, bbox_inches='tight')
plt.close()


# ---------------------------------------------------------------------------
# Generate a grid of diverse samples
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 4, figsize=(12, 6))

for i in range(8):
    generated = sample(model, diffusion_schedule, num_samples=1, img_size=args.img_size)
    img = denormalize(generated[0].cpu())
    img = img.permute(1, 2, 0).numpy()

    ax = axes[i // 4, i % 4]
    ax.imshow(np.clip(img, 0, 1))
    ax.set_title(f'Sample {i + 1}')
    ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, 'generated_samples_grid.png'), dpi=150, bbox_inches='tight')
plt.close()

print('Generated 8 diverse samples!')


# ---------------------------------------------------------------------------
# Save final model + checkpoint (for use in test_single_sample.ipynb)
# ---------------------------------------------------------------------------
torch.save(model.state_dict(), os.path.join(args.save_dir, 'diffusion_model_final.pth'))
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': args.epochs,
    'losses': losses
}, os.path.join(args.save_dir, 'diffusion_checkpoint.tar'))

print('Model saved successfully!')
print(f"Model size: {os.path.getsize(os.path.join(args.save_dir, 'diffusion_model_final.pth')) / 1e6:.2f} MB")
print(f'All outputs saved to: {args.output_dir}')
print(f'All model checkpoints saved to: {args.save_dir}')
