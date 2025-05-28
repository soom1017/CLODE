import torch

import os
from pathlib import Path
from tqdm import tqdm

from losses import *
from misc import *
from network.conv_node import NODE

os.environ["CUDA_VISIBLE_DEVICES"] = '5'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# MODEL_PATH = Path(__file__).parent / "pth" / "lowlight.pth"
MODEL_PATH = Path(__file__).parent / "fsp" / "checkpoints" / "default" / "best.pth"
IMAGE_PATH = Path("/home/soom/data/LOL/eval15")
filenames = sorted(os.listdir(IMAGE_PATH / 'low'))

model = NODE(device, (3, 128, 128), 32, augment_dim=0, time_dependent=True, adjoint=True)
model.eval()
model.to(device)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True), strict=False)

T = torch.tensor([0, 3]).float().to(device)

_psnr = 0
_ssim = 0

for filename in tqdm(filenames):
    with torch.no_grad():
        lq = image_tensor(IMAGE_PATH / 'low' / filename).unsqueeze(0).to(device)
        gt = image_tensor(IMAGE_PATH / 'high' / filename).unsqueeze(0).to(device)
        pred = model(lq, T, inference=True)['output']
        
        _psnr += calculate_psnr(pred, gt)
        _ssim += calculate_ssim(pred, gt)
    
_psnr /= len(filenames)
_ssim /= len(filenames)
print(f"PSNR: {_psnr:.4f}, SSIM: {_ssim:.4f}")