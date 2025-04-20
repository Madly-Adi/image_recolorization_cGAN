
import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from PIL import Image

# ---------------- Dataset ----------------

class ColorizationDataset(Dataset):
    def __init__(self, dataset, img_size=(128, 128)):
        self.dataset = dataset
        self.img_size = img_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        img = img.resize(self.img_size)
        img = np.array(img.convert("RGB"))
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab = lab.astype(np.float32) / 255.0

        L = lab[:, :, 0:1]
        ab = lab[:, :, 1:]

        L = torch.from_numpy(L).permute(2, 0, 1)
        ab = torch.from_numpy(ab).permute(2, 0, 1)

        return L, ab

# ---------------- Models ----------------

class UNetGenerator(nn.Module):
    def __init__(self, input_nc=1, output_nc=2):
        super().__init__()
        def down(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 4, 2, 1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, True)
            )

        def up(in_c, out_c):
            return nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, 4, 2, 1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(True)
            )

        self.down1 = down(input_nc, 64)
        self.down2 = down(64, 128)
        self.down3 = down(128, 256)
        self.down4 = down(256, 512)

        self.up1 = up(512, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = nn.ConvTranspose2d(128, output_nc, 4, 2, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        u1 = self.up1(d4)
        u2 = self.up2(torch.cat([u1, d3], 1))
        u3 = self.up3(torch.cat([u2, d2], 1))
        u4 = self.up4(torch.cat([u3, d1], 1))

        return self.tanh(u4)

class PatchDiscriminator(nn.Module):
    def __init__(self, input_nc=3):
        super().__init__()
        def block(in_c, out_c, norm=True):
            layers = [nn.Conv2d(in_c, out_c, 4, 2, 1)]
            if norm:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            block(input_nc, 64, norm=False),
            block(64, 128),
            block(128, 256),
            nn.Conv2d(256, 1, 4, 1, 1)
        )

    def forward(self, x):
        return self.model(x)

# ---------------- Training ----------------

def lab_to_rgb(L, ab):
    L = L.cpu().numpy()[0, 0]
    ab = ab.cpu().detach().numpy()[0].transpose(1, 2, 0)
    lab = np.concatenate((L[..., np.newaxis], ab), axis=2)
    lab = (lab * 255.0).astype(np.uint8)
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return rgb

def main():
    os.makedirs("outputs/images", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw_dataset = datasets.CIFAR10(root='./data', train=True, download=True)
    dataset = ColorizationDataset(raw_dataset)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    G = UNetGenerator().to(device)
    D = PatchDiscriminator().to(device)

    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.L1Loss()

    optimizer_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    for epoch in range(2):
        for i, (L, ab) in enumerate(train_loader):
            L, ab = L.to(device), ab.to(device)
            fake_ab = G(L)
            real_input = torch.cat([L, ab], 1)
            fake_input = torch.cat([L, fake_ab], 1)

            optimizer_D.zero_grad()
            pred_real = D(real_input)
            pred_fake = D(fake_input.detach())
            valid = torch.ones_like(pred_real)
            fake = torch.zeros_like(pred_fake)
            loss_D = criterion_GAN(pred_real, valid) + criterion_GAN(pred_fake, fake)
            loss_D.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()
            pred_fake = D(fake_input)
            loss_GAN = criterion_GAN(pred_fake, valid)
            loss_L1 = criterion_L1(fake_ab, ab)
            loss_G = loss_GAN + 100 * loss_L1
            loss_G.backward()
            optimizer_G.step()

            if i % 100 == 0:
                print(f"Epoch {epoch}, Batch {i}, Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")

    torch.save(G.state_dict(), "generator.pth")

    # Visualize a few outputs
    G.eval()
    L_sample, ab_sample = next(iter(train_loader))
    L_sample = L_sample.to(device)
    ab_sample = ab_sample.to(device)
    with torch.no_grad():
        fake_ab = G(L_sample)

    for i in range(5):
        pred_rgb = lab_to_rgb(L_sample[i:i+1], fake_ab[i:i+1])
        Image.fromarray(pred_rgb).save(f"outputs/images/generated_{i}.png")

if __name__ == "__main__":
    main()
