
import torch
import cv2
import numpy as np
from PIL import Image
from model import UNetGenerator
import os

def load_image(path, size=(128, 128)):
    img = Image.open(path).convert("L").resize(size)
    img_np = np.array(img).astype(np.float32) / 255.0
    L = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)
    return L

def lab_to_rgb(L, ab):
    L = L.cpu().numpy()[0, 0]
    ab = ab.cpu().detach().numpy()[0].transpose(1, 2, 0)
    lab = np.concatenate((L[..., np.newaxis], ab), axis=2)
    lab = (lab * 255.0).astype(np.uint8)
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return rgb

def infer_image(gray_img_path, output_path="custom_output.png", model_path="generator.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = UNetGenerator().to(device)
    G.load_state_dict(torch.load(model_path, map_location=device))
    G.eval()

    L = load_image(gray_img_path).to(device)

    with torch.no_grad():
        ab = G(L)

    rgb = lab_to_rgb(L, ab)
    Image.fromarray(rgb).save(output_path)
    print(f"Colorized image saved to {output_path}")

if __name__ == "__main__":
    infer_image("my_gray.png")
