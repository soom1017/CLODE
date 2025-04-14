import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, CLIPModel
from misc import image_tensor
import torchvision.transforms as transforms
from PIL import Image

datasets = {
    'LOL': '/home/soom/data/LOL/eval15',
    'LSRW': '/home/soom/data/LSRW/Eval'
}

CLIP_PROMPTS = [
    "brightness", "natural", "colorfullness",
]

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
        
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch16")
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])
        ])

    def _process_clip_and_generate_input(self, idx):
        with torch.no_grad():
            # [14, 768]
            text_tokens = self.tokenizer(CLIP_PROMPTS, padding=True, return_tensors="pt")
            text_features = self.model.get_text_features(**text_tokens.to(self.device))
            
            # [1, 768]
            image = Image.open(self.data_path / 'low' / self.image_labels[idx]).convert('RGB')
            lq_224 = self.transform(image).unsqueeze(0).to(self.device)
            image_feature = self.model.get_image_features(lq_224)
            
            # [14, 768] + [1, 768] = [15, 768]
            x = torch.cat([text_features, image_feature], dim=0)
        
        return x.unsqueeze(0)
    
    def __len__(self):
        return len(self.image_labels)
    
    def __getitem__(self, idx):
        lq = image_tensor(self.data_path / 'low' / self.image_labels[idx])
        gt = image_tensor(self.data_path / 'high' / self.image_labels[idx])
        
        x = self._process_clip_and_generate_input(idx)
        
        return x, lq, gt