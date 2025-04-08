from omegaconf import OmegaConf
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import torch
from model import *
from network.conv_node import NODE
from data.clip_dataset import TrainDataset
from data.eval_dataset import EvalDataset
from utils import plot_regression_results
from misc import calculate_psnr, calculate_ssim

RUN_NAME = sys.argv[1]
args = OmegaConf.load(Path(__file__).parent / 'model' / RUN_NAME / 'config.yaml')
args.merge_with_cli()

models = {
    'v1': Regressor_v1,
    'v2': Regressor_v2,
}

CLODE_model_path = Path(__file__).parent / '..' / 'pth'
model_path = Path(__file__).parent / 'model' / RUN_NAME
model_file = next(model_path.glob('*.pth'))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = NODE(device, (3, 400, 600), 32, augment_dim=0, time_dependent=True, adjoint=True)
model.eval()
model.to(device)
model.load_state_dict(torch.load(CLODE_model_path / 'lowlight.pth', weights_only=True), strict=False)

regressor = models[args.model]().to(device)
regressor.load_state_dict(torch.load(model_file, weights_only=True, map_location=device))
regressor.eval()


# Plot regression performance on [train, val] dataset
train_data = 'LOL' if args.data in ['LOL', 'LSRW'] else 'SICE'
train_set = TrainDataset(train_data, split='train', val_size=0.2, device=device)
val_set = TrainDataset(train_data, split='val', val_size=0.2, device=device)

with torch.no_grad():
    y_pred = regressor(train_set.X).cpu().numpy()
    y_true = train_set.y.cpu().numpy()
    
    plot_regression_results(
        y_true, 
        y_pred, 
        'Regression Results (for Train set): True vs Predicted T', 
        model_path / 'train_results.png'
    )

    y_pred = regressor(val_set.X).cpu().numpy()
    y_true = val_set.y.cpu().numpy()
    
    plot_regression_results(
        y_true, 
        y_pred, 
        'Regression Results (for Test set): True vs Predicted T', 
        model_path / 'test_results.png'
    )
    
# Caculate PSNR for test dataset
test_dataset = EvalDataset(args.data, device)
psnrs = []
ssims = []

for datum in tqdm(test_dataset):
    x, lq, gt = datum
    
    with torch.no_grad():
        # (Ours) Regression for T
        pred_T = regressor(x).item()
        # CLODE
        integration_time = torch.tensor([0, pred_T]).float().cuda()
        pred = model(lq, integration_time, inference=True)['output'][0]
        
        _psnr = calculate_psnr(pred, gt)
        _ssim = calculate_ssim(pred, gt)
    psnrs.append(_psnr)
    ssims.append(_ssim)

print(f"PSNR: {np.mean(psnrs):.4f} dB")
print(f"SSIM: {np.mean(ssims):.4f}")