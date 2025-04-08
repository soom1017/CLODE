import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split

class TrainDataset(Dataset):
    def __init__(self, dataset, split, val_size, device):
        super().__init__()
        
        self.data_path = Path(__file__).parent / dataset
        
        # Load input
        features = np.load(self.data_path / f'train_clip_features.npy')
        scores = np.load(self.data_path / f'train_clip_scores.npy')        
        X = np.hstack([features, scores])
        
        # Load output
        y = np.load(self.data_path / f'train_best_t_v2.npy')[:, 0]
        
        X = torch.tensor(X, dtype=torch.float32).to(device)
        y = torch.tensor(y, dtype=torch.float32).to(device)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=val_size, random_state=42
        )
        
        if split == 'train':
            self.X = X_train
            self.y = y_train
        elif split == 'val':
            self.X = X_test
            self.y = y_test
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        
        return x, y
    
def get_dataloader(dataset, batch_size=16, val_size=0.2, device='cuda:0'):
    train_set = TrainDataset(dataset, split='train', val_size=val_size, device=device)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
    val_set = TrainDataset(dataset, split='val', val_size=val_size, device=device)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader