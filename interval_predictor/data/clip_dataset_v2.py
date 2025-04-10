import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, CLIPModel
from sklearn.model_selection import train_test_split
from misc import image_tensor

datasets = {
    'LOL': '/home/soom/data/LOL/our485',
}

CLIP_PROMPTS = [
    # brightness, noisiness, quality, colorfullness, contrast, complexity, warm
    "bright photo", "dark photo", 
    "good photo", "bad photo",
    "clean photo", "noisy photo", 
    "colorful photo", "dull photo",
    "high contrast photo", "low contrast photo",
    "complex photo", "simple photo",
    "warm photo", "cold photo",
]

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
        input_feature_path = self.npy_path / 'train_clip_features_v2.npy'
        if not input_feature_path.exists():
            X = self._process_clip_and_generate_input(input_feature_path)
        else:
            X = np.load(input_feature_path)
            X = torch.tensor(X, dtype=torch.float32).to(device)
            
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
        tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        
        with torch.no_grad():
            model.eval()
            
            # [14, 768]
            text_tokens = tokenizer(CLIP_PROMPTS, padding=True, return_tensors="pt")
            text_features = model.get_text_features(**text_tokens.to(self.device))
            
            # [num_images, 768]
            image_features = []
            for label in self.image_labels:            
                lq_224 = image_tensor(self.data_path / 'low' / label, size=(224, 224))
                image_feat = model.get_image_features(lq_224)
                image_features.append(image_feat.squeeze(0))    
            image_features = torch.stack(image_features).to(self.device)
            
            # [num_images, 14, 768] + [num_images, 1, 768] = [num_images, 15, 768]
            text_features = text_features.unsqueeze(0).repeat(image_features.shape[0], 1, 1)
            image_features = image_features.unsqueeze(1)
            
            x = torch.cat([text_features, image_features], dim=1)
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