import os
from pathlib import Path
import torch
from torchvision import transforms
import cv2
import numpy as np
from tqdm import tqdm
import pyiqa
from scipy.interpolate import interp1d
from transformers import AutoTokenizer, CLIPModel

from network.conv_node import NODE
from misc import *

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
# CLODE model
model = NODE(device, (3, 400, 600), 32, augment_dim=0, time_dependent=True, adjoint=True)
model.eval()
model.to(device)
model.load_state_dict(torch.load("pth/universal.pth", map_location=device, weights_only=True), strict=False)

# - saving trajectory
trajectory = []
ode_func = model.odefunc.forward
def get_trajectory(t, x):
    t_val = t.item() if isinstance(t, torch.Tensor) else float(t)
    t_denoised = torch.clamp(x[:, :3, :, :] - model.odefunc.denoise(x[:, :3, :, :]), 0, 1)
    t_denoised = t_denoised.detach().cpu()
    t_img = x[:, :3, :, :].detach().cpu()    
    trajectory.append((t_val, t_img, t_denoised))
    
    return ode_func(t, x)

model.odefunc.forward = get_trajectory

# CLIP model
CLIP_PROMPTS = [
    "brightness", "noisiness", "quality", "colorfullness",
    "contrast", "complex", "warm", "sharp",
    "natural",
]

clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch16")
clip.eval()

with torch.no_grad():
    text_tokens = tokenizer(CLIP_PROMPTS, padding=True, return_tensors="pt")
    text_features = clip.get_text_features(**text_tokens).cpu().numpy()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                        std=[0.26862954, 0.26130258, 0.27577711])
])

def cosine_similarity(vec1, vec2):
    sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return sim.item()

# Image dataset
def load_image(filepath):
    img = cv2.imread(str(filepath))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (np.asarray(img)/255.0)
    img = torch.from_numpy(img).float()
    img = img.permute(2, 0, 1).unsqueeze(0)
    
    return img.to(device)

IMAGE_PATH = Path("/home/soom/data/LOL/our485")

filenames = sorted(os.listdir(IMAGE_PATH / 'low'))
data = np.zeros((len(filenames), 1000, 18))       # t val, lum mean, noise loss, CLIP scores(9), psnr, ssim, non-reference metrics(4)

# Non-reference metrics
niqe_metric = pyiqa.create_metric('niqe', device=device)
brisque_metric = pyiqa.create_metric('brisque', device=device)
pi_metric = pyiqa.create_metric('pi', device=device)
entropy_metric = pyiqa.create_metric('entropy', device=device)

"""
Main:
    1. Load low and high images
    2. Get the trajectory of the model
    3. Calculate the noise loss (from before-denoising images), luminance, and CLIP scores (from denoised images)
    4. Calculate full reference metrics (psnr, ssim) and non-reference metrics (niqe, brisque, pi, entropy)
    5. Interpolate the scores about 1000 points: t values [2, 5, 1000]
    6. Save the t values and scores

"""
x_dense = np.linspace(2, 5, 1000)

for i in tqdm(range(len(filenames))):
    lq_img = load_image(IMAGE_PATH / 'low' / filenames[i])
    gt_img = load_image(IMAGE_PATH / 'high' / filenames[i])
    trajectory = []

    T = 5
    with torch.no_grad():
        integration_time = torch.tensor([0, T]).float().to(device)
        pred = model(lq_img, integration_time, inference=True)['output']
    trajectory.sort(key=lambda x: x[0])
    t_vals, imgs, denoised_imgs = zip(*trajectory)
    t_vals, imgs, denoised_imgs = np.array(t_vals), np.array(imgs), np.array(denoised_imgs)
    
    # Remove duplicates
    _, unique_indices = np.unique(t_vals, return_index=True)
    t_vals, imgs, denoised_imgs = t_vals[unique_indices], imgs[unique_indices], denoised_imgs[unique_indices]
    
    datum = np.zeros((len(t_vals), 17))

    lum = denoised_imgs.squeeze(1).max(axis=1)
    lum_mean = lum.mean(axis=(1,2))
    datum[:, 0] = lum_mean

    for _t, pred in enumerate(imgs):
        pred = np.clip(pred, 0, 1)
        pred = torch.from_numpy(pred).float()
        with torch.no_grad():
            noise_loss = model.odefunc.loss_func(pred.to(device))
            noise_loss = noise_loss.item()
        datum[_t, 1] = noise_loss
    
    for _t, pred in enumerate(denoised_imgs):
        pred = torch.from_numpy(pred).float()
        with torch.no_grad():
            img_224 = transform(pred[0]).unsqueeze(0)
            img_feat = clip.get_image_features(img_224).cpu().numpy()
        datum[_t, 2:11] = [cosine_similarity(img_feat, text_feat) for text_feat in text_features]

        metrics = [
            calculate_psnr(pred[0], gt_img[0]),
            calculate_ssim(pred[0], gt_img[0]),
            niqe_metric(pred).item(),
            brisque_metric(pred).item(),
            pi_metric(pred).item(),
            entropy_metric(pred).item()
        ]
        datum[_t, 11:] = metrics

    # Normalize scores
    datum = (datum - datum.min(axis=0, keepdims=True)) / (datum.max(axis=0, keepdims=True) - datum.min(axis=0, keepdims=True))

    # Interpolate scores
    interp_func = interp1d(t_vals, datum, kind='cubic', axis=0)
    y_dense = interp_func(x_dense)
    data[i, :, 0] = x_dense
    data[i, :, 1:] = y_dense

np.save('trajectory_data_lol_univpth.npy', data)