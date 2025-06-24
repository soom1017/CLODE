import os
import sys
from pathlib import Path
from tqdm import tqdm
import torch
import pyiqa

from losses import *
from misc import *
from network.conv_node import NODE

MODEL_NAME, DATA_TYPE, DEVICE = sys.argv[1], sys.argv[2], sys.argv[3]
MODEL_PATH = Path(__file__).parent / "fsp" / "checkpoints" / MODEL_NAME / "best_psnr.pth"

dirs = { "LOL": "LOL/eval15",  "LSRW": "LSRW/eval", "SICE": "SICE/eval"}
IMAGE_PATH = Path("/home/soom/data") / dirs[DATA_TYPE]

os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

filenames = sorted(os.listdir(IMAGE_PATH / 'low'))
data = np.zeros((len(filenames), 21, 7))    # t_val, psnr, ssim, clipiqa (RN50 / ViTL14), pi, entropy

model = NODE(device, (3, 128, 128), 32, augment_dim=0, time_dependent=True, adjoint=True)
model.eval()
model.to(device)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True), strict=False)

t0_vals = np.linspace(-1, 1, 21)

iqa_metric = pyiqa.create_metric('clipiqa', device=device)
iqa2_metric = pyiqa.create_metric('clipiqa+_vitL14_512', device=device)
pi_metric = pyiqa.create_metric('pi', device=device)
entropy_metric = pyiqa.create_metric('entropy', device=device)

_psnr = 0
_ssim = 0

_psnr2 = 0
_ssim2 = 0

with torch.no_grad():
    for i, filename in enumerate(tqdm(filenames)):
        lq = image_tensor(IMAGE_PATH / 'low' / filename).unsqueeze(0).to(device)
        gt = image_tensor(IMAGE_PATH / 'high' / filename).unsqueeze(0).to(device)
        
        lq_iqa = iqa_metric(lq).item()
        lq_iqa2 = iqa2_metric(lq).item()
        lq_pi = pi_metric(lq).item()
        lq_entropy = entropy_metric(lq).item()
        
        for j, T in enumerate(tqdm(t0_vals, leave=False)):
            integration_time = torch.tensor([T, 3]).float().to(device)
            pred = model(lq, integration_time, inference=True)['output']
            
            data[i, j, 0] = T
            data[i, j, 1] = calculate_psnr(pred[0], gt[0])
            data[i, j, 2] = calculate_ssim(pred[0], gt[0])
            data[i, j, 3] = iqa_metric(pred).item() - lq_iqa  # CLIP-IQA RN50
            data[i, j, 4] = iqa2_metric(pred).item() - lq_iqa2  # # CLIP-IQA ViT-L/14
            data[i, j, 5] = pi_metric(pred).item() - lq_pi  # PI
            data[i, j, 5] *= -1
            data[i, j, 6] = entropy_metric(pred).item() - lq_entropy # Entropy
        
        # CLIP-IQA + PI + Entropy 기준으로 
        # [CONSIDER]: scale the scores to [0, 1] range?
        best_idx = np.argmax(data[i, :, 3] + data[i, :, 5] + data[i, :, 6])
        _psnr += data[i, best_idx, 1]
        _ssim += data[i, best_idx, 2]
        
        best_idx2 = np.argmax(data[i, :, 4] + data[i, :, 5] + data[i, :, 6])
        _psnr2 += data[i, best_idx2, 1]
        _ssim2 += data[i, best_idx2, 2]
    
# Save the results
SAVE_PATH = Path(__file__).parent / "fsp" / "data" / MODEL_NAME / "t0"
SAVE_PATH.mkdir(parents=True, exist_ok=True)

np.save(SAVE_PATH / f"results_{DATA_TYPE}.npy", data)
print(f"Results saved to {SAVE_PATH / f'results_{DATA_TYPE}.npy'}")

# Show results
print(f"Average PSNR (CLIP-IQA + PI + Entropy): {_psnr / len(filenames):.4f}")
print(f"Average SSIM (CLIP-IQA + PI + Entropy): {_ssim / len(filenames):.4f}")

print(f"Average PSNR (CLIP-IQA ViT-L/14 + PI + Entropy): {_psnr2 / len(filenames):.4f}")
print(f"Average SSIM (CLIP-IQA ViT-L/14 + PI + Entropy): {_ssim2 / len(filenames):.4f}")