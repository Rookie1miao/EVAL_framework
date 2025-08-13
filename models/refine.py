import torch.nn as nn
import torch.nn.functional as F
import torch

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return identity + out

class CAGLocal(nn.Module):
    def __init__(self, channels, k=5, scale=5):
        super().__init__()
        self.k = k
        self.scale = scale
        self.q_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.k_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.v_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, feat_q, feat_kv):
        B, C, H, W = feat_q.shape
        _, _, Hh, Wh = feat_kv.shape
        assert Hh == H * self.scale and Wh == W * self.scale, \
            f"High-res feature size must be {self.scale}× low-res, " \
            f"got {(Hh, Wh)} vs {(H, W)}"

        q = self.q_proj(feat_q)         
        k = self.k_proj(feat_kv)           
        v = self.v_proj(feat_kv)             

        unfold = nn.Unfold(kernel_size=self.k,
                           stride=self.scale)     
        k = unfold(k)                                
        v = unfold(v)                           
        k = k.view(B, C, self.k * self.k, H * W)        
        v = v.view(B, C, self.k * self.k, H * W)    

        q = q.view(B, C, -1).permute(0, 2, 1)      
        attn = (q.unsqueeze(2) *                 
                k.permute(0, 3, 2, 1)).sum(-1)        
        attn = self.softmax(attn / (C ** 0.5))

        out = (attn.unsqueeze(-1) *              
               v.permute(0, 3, 2, 1)).sum(2)       
        out = out.permute(0, 2, 1).view(B, C, H, W)     

        return feat_q + out                       

class FeatureExtractor(nn.Module):
    def __init__(self, in_ch, base_ch=32, num_blocks=8):
        super().__init__()
        layers = [nn.Conv2d(in_ch, base_ch, 3, 1, 1), nn.ReLU(inplace=True)]
        for _ in range(num_blocks):
            layers.append(ResidualBlock(base_ch))
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)

class DualFeatureRefiner(nn.Module):
    def __init__(self, base_ch=32, num_blocks=8, scale=5, k=5, **kwargs):
        super().__init__()
        self.scale = scale
        self.low_branch   = FeatureExtractor(in_ch=1, base_ch=base_ch, num_blocks=num_blocks)
        self.high_branch  = FeatureExtractor(in_ch=1, base_ch=base_ch, num_blocks=num_blocks//2)
        self.cag          = CAGLocal(base_ch, k=k, scale=scale)
        self.fusion_conv  = nn.Conv2d(base_ch*2, base_ch, 1)
        self.fusion_blocks= nn.Sequential(*[ResidualBlock(base_ch) for _ in range(num_blocks)])
        self.head         = nn.Conv2d(base_ch, 1, 3, 1, 1)

    def forward(self,inputs, instances):

        feat_low  = self.low_branch(inputs)     

        feat_high = self.high_branch(instances)       

        feat_low  = self.cag(feat_low, feat_high)      

        fused = torch.cat([feat_low, 
                           F.avg_pool2d(feat_high, kernel_size=self.scale)], dim=1)  
        fused = self.fusion_conv(fused)
        fused = self.fusion_blocks(fused)
        delta = self.head(fused)

        out = inputs + delta
        return out
