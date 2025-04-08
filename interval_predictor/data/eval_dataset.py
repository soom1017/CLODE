import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchmetrics.multimodal import CLIPImageQualityAssessment

from misc import image_tensor

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
        
        # Load CLIP IQA model
        self.prompts = ('brightness', 'noisiness', 'quality')
        self.clip_metric = CLIPImageQualityAssessment(
            model_name_or_path="openai/clip-vit-large-patch14",
            prompts=self.prompts
        ).to(device)
    
        self.clip_vision_encoder = self.clip_metric.model.vision_model.to(device)
        self.clip_vision_encoder.eval()
    
        self.clip_visual_projection = self.clip_metric.model.visual_projection.to(device)
        self.clip_visual_projection.eval()

    def _generate_input(self, img):    
        """
        Generate input (CLIP feature + scores) for evaluation.
        """
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        
        with torch.no_grad():
            scores = self.clip_metric(img)
            scores_t = torch.tensor([
                scores[self.prompts[0]].item(),
                scores[self.prompts[1]].item(),
                scores[self.prompts[2]].item()
            ], device=self.device).unsqueeze(0)  
            
            feature = self.clip_vision_encoder(img) 
            feature_t = self.clip_visual_projection(feature[1]) 
            
            x = torch.cat([feature_t, scores_t], dim=1)

        return x
    
    def __len__(self):
        return len(self.image_labels)
    
    def __getitem__(self, idx):
        lq = image_tensor(self.data_path / 'low' / self.image_labels[idx])
        gt = image_tensor(self.data_path / 'high' / self.image_labels[idx])
        
        lq_224 = image_tensor(self.data_path / 'low' / self.image_labels[idx], size=(224, 224))
        x = self._generate_input(lq_224)
        
        return x, lq, gt