import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel
from sklearn.model_selection import train_test_split
from misc import image_tensor

datasets = {
    'LOL': '/home/soom/data/LOL/our485',
}

def extract_extra_features(img_tensor):
    """
    img_tensor: [1, 3, H, W] - RGB float tensor (0~1 range)
    returns: [5] torch.tensor
    """
    # Convert to grayscale [1, 1, H, W]
    gray = TF.rgb_to_grayscale(img_tensor).squeeze(0)  # [1, H, W]
    gray_flat = gray.flatten()

    # 1. Mean luminance
    mean_luminance = gray_flat.mean().item()

    # 2. Local contrast (std dev of grayscale)
    local_contrast = gray_flat.std().item()

    # 3. Sharpness (Laplacian)
    laplacian_kernel = torch.tensor([[[[-1, -1, -1],
                                       [-1,  8, -1],
                                       [-1, -1, -1]]]], dtype=torch.float32).to(img_tensor.device)
    laplacian = F.conv2d(gray.unsqueeze(0), laplacian_kernel, padding=1)
    sharpness = laplacian.abs().mean().item()

    # 4. Histogram spread (brightness range)
    hist = torch.histc(gray_flat, bins=64, min=0, max=1)
    hist_norm = hist / hist.sum()
    spread = (hist_norm > 0.01).sum().item() / 64.0  # 비율로 정규화

    # 5. Noise estimate (high freq energy)
    sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32).to(img_tensor.device)
    edges = F.conv2d(gray.unsqueeze(0), sobel_x, padding=1)
    noise = edges.abs().mean().item()

    return torch.tensor([
        mean_luminance,
        local_contrast,
        sharpness,
        spread,
        noise,
    ], dtype=torch.float32)


class TrainDataset(Dataset):
    def __init__(self, dataset, split, val_size, device):
        super().__init__()
        
        # Load images
        try:
            self.data_path = Path(datasets[dataset])
            self.npy_path = Path(__file__).parent / dataset
        except KeyError:
            raise ValueError(f"Dataset not supported. Choose from: {list(datasets.keys())}.")
        
        self.image_labels = sorted(os.listdir(self.data_path / 'low'))
        self.device = device
        
        # Load input
        input_feature_path = self.npy_path / 'train_clip_features_v3.npy'
        X = self._process_clip_and_generate_input(input_feature_path)
            
        y = np.load(self.npy_path / f'train_best_t_v2.npy')[:, 0]
        y = torch.tensor(y, dtype=torch.float32).to(device)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=val_size, random_state=42
        )
        if split == 'train':
            self.X, self.y = X_train, y_train
            self.len = len(y_train)
        elif split == 'val':
            self.X, self.y = X_test, y_test
            self.len = len(y_test)
        
    def _process_clip_and_generate_input(self, save_path):
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        
        with torch.no_grad():
            model.eval()
            
            # [num_images, 768 + 5]
            image_features = []
            for label in self.image_labels:            
                lq_224 = image_tensor(self.data_path / 'low' / label, size=(224, 224))
                image_feat = model.get_image_features(lq_224)
                extra_feat = extract_extra_features(lq_224).to(self.device)
                image_features.append(torch.cat([image_feat.squeeze(0), extra_feat]))    
            x = torch.stack(image_features).to(self.device)
            
            np.save(save_path, x.cpu().numpy())
        
        return x
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
def get_dataloader(dataset, batch_size=16, val_size=0.2, device='cuda:0'):
    train_set = TrainDataset(dataset, split='train', val_size=val_size, device=device)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
    val_set = TrainDataset(dataset, split='val', val_size=val_size, device=device)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader