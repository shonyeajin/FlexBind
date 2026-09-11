import torch
import torch.nn as nn
from .layers import ComplexCWT, ScaleAttention, ComplexInteractionBlock, MambaPython

class IDRModel(nn.Module):
    def __init__(self, d, D_att=128, D_w=128, scales=[1, 3, 4, 5, 8, 16, 32]):
        super().__init__()
        self.d_plm = d
        self.D_w = D_w
        self.D_att = D_att

        self.heads = nn.ModuleDict({
            "disorder": nn.Linear(d, 1),
            "protein": nn.Linear(d, 1),
            "rna": nn.Linear(d, 1),
            "dna": nn.Linear(d, 1),
        })

        self.cwt = ComplexCWT(in_dim=d, scales=scales, wavelet_dim=D_w)
        self.real_att = ScaleAttention(d, D_att, D_w)
        self.imag_att = ScaleAttention(d, D_att, D_w)
        self.cib = ComplexInteractionBlock(D_att)
        self.out_proj = nn.Linear(D_att, d)

        self.mamba_blocks = nn.ModuleList([
            MambaPython(d_model=d, d_state=16, expand=2, conv_kernel=3)
            for _ in range(3)
        ])

    def forward(self, emb, mask=None):
        real, imag = self.cwt(emb)
        ctx_R = self.real_att(emb, real)
        ctx_I = self.imag_att(emb, imag)

        fused = self.cib(ctx_R, ctx_I)
        E_fused = emb + self.out_proj(fused)

        H = E_fused
        for blk in self.mamba_blocks:
            H = blk(H)

        out = {k: self.heads[k](H).squeeze(-1) for k in self.heads}
        return out
