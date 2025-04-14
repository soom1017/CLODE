import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers import CLIPModel
from misc import image_tensor
from .clip_dataset_v3 import extract_extra_features

datasets = {
    'LOL': '/home/soom/data/LOL/eval15',
    'LSRW': '/home/soom/data/LSRW/Eval'
}

class EvalDataset(Dataset):
    def __init__(self, dataset='LOL', device='cuda:0'):
        super().__init__()
        
        # Load images
        try:
            self.data_path = Path(datasets[dataset])
        except KeyError:
            raise ValueError(f"Dataset not supported. Choose from: {list(datasets.keys())}.")
        
        self.image_labels = sorted(os.listdir(self.data_path / 'low'))
        self.device = device
        
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        self.model.eval()

    def _process_clip_and_generate_input(self, idx):
        with torch.no_grad():
            # [1, 768]
            lq_224 = image_tensor(self.data_path / 'low' / self.image_labels[idx], size=(224, 224))
            image_feat = self.model.get_image_features(lq_224)
            extra_feat = extract_extra_features(lq_224).to(self.device)
            
            # [768 + 5]
            x = torch.cat([image_feat.squeeze(0), extra_feat])
        
        return x.unsqueeze(0)
    
    def __len__(self):
        return len(self.image_labels)
    
    def __getitem__(self, idx):
        lq = image_tensor(self.data_path / 'low' / self.image_labels[idx])
        gt = image_tensor(self.data_path / 'high' / self.image_labels[idx])
        
        x = self._process_clip_and_generate_input(idx)
        
        return x, lq, gt