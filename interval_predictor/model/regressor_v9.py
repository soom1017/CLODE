import torch
import torch.nn as nn

class Regressor(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=64, num_prompts=3):
        super(Regressor, self).__init__()
        self.num_prompts = num_prompts
        self.vision_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )

        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        # [B, 4, 64]
        x = self.vision_mlp(x)
        
        image_emb = x[:, -1, :].unsqueeze(dim=1).permute(1, 0, 2)    # [1, B, 64]
        prompt_emb = x[:, :-1, :].permute(1, 0, 2)                   # [3, B, 64]

        # [B, 64]
        attn_output, _ = self.cross_attention(image_emb, prompt_emb, prompt_emb)
        attn_output = attn_output.squeeze(0)
        
        # [B,]
        T_pred = self.fc(attn_output).squeeze(-1)
        return T_pred