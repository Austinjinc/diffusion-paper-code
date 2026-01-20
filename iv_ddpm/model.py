import torch
import torch.nn as nn
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim, num_groups=2):
        super().__init__()
        self.projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, 15),
            nn.SiLU(),
            nn.Linear(15,out_channels * 4)
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(num_groups, out_channels)
        self.act1 = nn.SiLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.act2 = nn.SiLU()

    def forward(self, x, comb_emb):

        scale_shift = self.projection(comb_emb)
        scale1, shift1, scale2, shift2 = scale_shift.chunk(4, dim=1)

        h = self.conv1(x)
        h = self.norm1(h)
        h = h * scale1.unsqueeze(-1).unsqueeze(-1) + shift1.unsqueeze(-1).unsqueeze(-1) # Broadcasted addition
        h = self.act1(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = h * scale2.unsqueeze(-1).unsqueeze(-1) + shift2.unsqueeze(-1).unsqueeze(-1) # Broadcasted addition
        h = self.act2(h)
        return h

class ConditionalUnet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Time Embedding
        self.time_mlp = SinusoidalPositionEmbeddings(cfg.emb_dim // 2)

        # Scalar Conditioning MLP
        self.scalar_mlp = nn.Sequential(
            nn.Linear(cfg.n_scalars, 10),
            nn.SiLU(),
            nn.Linear(10, 10),
            nn.SiLU(),
            nn.Linear(10, cfg.emb_dim // 2) # Output scalar embedding
            
        )
        self.enc1 = DoubleConv(cfg.c_in, cfg.enc_channels, cfg.emb_dim, num_groups=2)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=3)
        self.bottle = DoubleConv(cfg.enc_channels, cfg.bottle_channels, cfg.emb_dim, num_groups=2)
        self.up = nn.ConvTranspose2d(cfg.bottle_channels, cfg.enc_channels, kernel_size=3, stride=3)
        self.dec1 = DoubleConv(cfg.enc_channels * 2, cfg.enc_channels, cfg.emb_dim, num_groups=2)
        self.out_conv = nn.Conv2d(cfg.enc_channels, cfg.c_out, kernel_size=1)

    def forward(self, x_t, time_t, scalars):
        t_emb = self.time_mlp(time_t)
        s_emb = self.scalar_mlp(scalars)
        comb_emb = torch.cat([t_emb, s_emb], dim=-1)  # Combine time and scalar embeddings
        skip_conn = self.enc1(x_t, comb_emb)
        
        h = self.pool(skip_conn)
        h = self.bottle(h, comb_emb)
        h = self.up(h)
        h = torch.cat([h, skip_conn], dim=1)
        h = self.dec1(h, comb_emb)

        return self.out_conv(h)