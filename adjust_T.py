import os
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from scipy.interpolate import interp1d
import pyiqa

from network.conv_node import NODE
from misc import *
from losses import *

MODEL_NAME, DATA_TYPE, DEVICE = sys.argv[1], sys.argv[2], sys.argv[3]
MODEL_PATH = Path(__file__).parent / "fsp" / "checkpoints" / MODEL_NAME / "best_psnr.pth"

dirs = { "LOL": "LOL/eval15",  "LSRW": "LSRW/eval", "SICE": "SICE/eval"}
IMAGE_PATH = Path("/home/soom/data") / dirs[DATA_TYPE]

os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
# CLODE model
model = NODE(device, (3, 128, 128), 32, augment_dim=0, time_dependent=True, adjoint=True)
model.eval()
model.to(device)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True), strict=False)

# - saving trajectory
trajectory = []
ode_func = model.odefunc.forward
def get_trajectory(t, x):
    t_val = t.item() if isinstance(t, torch.Tensor) else float(t)
    t_denoised = torch.clamp(x[:, :3, :, :] - model.odefunc.denoise(x[:, :3, :, :]), 0, 1)
    t_denoised = t_denoised.detach().cpu()
    trajectory.append((t_val, t_denoised))
    
    return ode_func(t, x)

model.odefunc.forward = get_trajectory

# Image dataset
def load_image(filepath):
    img = cv2.imread(str(filepath))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (np.asarray(img)/255.0)
    img = torch.from_numpy(img).float()
    img = img.permute(2, 0, 1).unsqueeze(0)
    
    return img.to(device)

filenames = sorted(os.listdir(IMAGE_PATH / 'low'))
data = np.zeros((len(filenames), 1000, 7))  # t val, psnr, ssim, clipiqa (RN50 / ViTL14), pi, entropy

"""
Main:
    PSNR trajectory 뽑고 interpolate해서 저장

"""
x_dense = np.linspace(2, 5, 1000)

iqa_metric = pyiqa.create_metric('clipiqa', device=device)
iqa2_metric = pyiqa.create_metric('clipiqa+_vitL14_512', device=device)
pi_metric = pyiqa.create_metric('pi', device=device)
entropy_metric = pyiqa.create_metric('entropy', device=device)

for i in tqdm(range(len(filenames))):
    lq_img = load_image(IMAGE_PATH / 'low' / filenames[i])
    gt_img = load_image(IMAGE_PATH / 'high' / filenames[i])
    
    lq_iqa = iqa_metric(lq_img).item()
    lq_iqa2 = iqa2_metric(lq_img).item()
    lq_pi = pi_metric(lq_img).item()
    lq_entropy = entropy_metric(lq_img).item()
    
    # Trajectory
    trajectory = []

    T = 5
    with torch.no_grad():
        integration_time = torch.tensor([0, T]).float().to(device)
        pred = model(lq_img, integration_time, inference=True)['output']
    trajectory.sort(key=lambda x: x[0])
    t_vals, denoised_imgs = zip(*trajectory)
    t_vals, denoised_imgs = np.array(t_vals), np.array(denoised_imgs)
    
    ## Remove duplicates
    _, unique_indices = np.unique(t_vals, return_index=True)
    t_vals, denoised_imgs = t_vals[unique_indices], denoised_imgs[unique_indices]
    
    datum = np.zeros((len(t_vals), 6))
    
    # Scores
    for _t, pred in enumerate(denoised_imgs):
        pred = torch.from_numpy(pred).float()

        metrics = [
            calculate_psnr(pred[0], gt_img[0]),
            calculate_ssim(pred[0], gt_img[0]),
            iqa_metric(pred).item() - lq_iqa,  # CLIP-IQA RN50
            iqa2_metric(pred).item() - lq_iqa2,  # CLIP-IQA ViT-L/14
            lq_pi - pi_metric(pred).item(),  # PI
            lq_entropy - entropy_metric(pred).item()  # Entropy
        ]
        datum[_t, :] = metrics

    # Interpolate scores
    interp_func = interp1d(t_vals, datum, kind='cubic', axis=0)
    y_dense = interp_func(x_dense)
    data[i, :, 0] = x_dense
    data[i, :, 1:] = y_dense

SAVE_PATH = Path(__file__).parent / "fsp" / "data" / MODEL_NAME / "T"
SAVE_PATH.mkdir(parents=True, exist_ok=True)

np.save(SAVE_PATH / f"results_{DATA_TYPE}.npy", data)
print(f"Results saved to {SAVE_PATH / f'results_{DATA_TYPE}.npy'}")

# Show results
model.odefunc.forward = ode_func  # Restore original forward function

def show_results(best_idx, desc):
    _psnr = 0
    _ssim = 0
    for i in tqdm(range(len(filenames))):
        lq_img = load_image(IMAGE_PATH / 'low' / filenames[i])
        gt_img = load_image(IMAGE_PATH / 'high' / filenames[i])
        
        T = data[i, best_idx[i], 0]
        with torch.no_grad():
            integration_time = torch.tensor([0, T]).float().to(device)
            pred = model(lq_img, integration_time, inference=True)['output']
        
        _psnr += calculate_psnr(pred[0], gt_img[0])
        _ssim += calculate_ssim(pred[0], gt_img[0])
    print(f"Average PSNR (CLIP-IQA {desc} + PI + Entropy): {_psnr / len(filenames):.4f}")
    print(f"Average SSIM (CLIP-IQA {desc} + PI + Entropy): {_ssim / len(filenames):.4f}")
    
idx = np.argmax(data[:, :, 3] + data[:, :, 5] + data[:, :, 6], axis=1)
show_results(idx, desc="RN50")

idx2 = np.argmax(data[:, :, 4] + data[:, :, 5] + data[:, :, 6], axis=1)
show_results(idx2, desc="ViT-L/14")