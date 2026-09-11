import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ComplexCWT(nn.Module):
    def __init__(self, in_dim, scales, wavelet_dim, kernel_size=5):
        super().__init__()
        self.scales = scales
        self.D_w = wavelet_dim
        self.kernel_size = kernel_size

        self.dim_reduce = nn.Conv1d(in_dim, wavelet_dim, kernel_size=1)

        self.kernels_real = nn.Parameter(
            torch.randn(len(scales), wavelet_dim, wavelet_dim, kernel_size)
        )
        self.kernels_imag = nn.Parameter(
            torch.randn(len(scales), wavelet_dim, wavelet_dim, kernel_size)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.dim_reduce(x)

        real_out = []
        imag_out = []

        for i, s in enumerate(self.scales):
            ker_r = self.kernels_real[i]
            ker_i = self.kernels_imag[i]

            r = F.conv1d(x, ker_r, padding=(self.kernel_size // 2) * s, dilation=s)
            m = F.conv1d(x, ker_i, padding=(self.kernel_size // 2) * s, dilation=s)

            real_out.append(r)
            imag_out.append(m)

        real_out = torch.stack(real_out, dim=2)
        imag_out = torch.stack(imag_out, dim=2)

        return real_out, imag_out

class ScaleAttention(nn.Module):
    def __init__(self, D_plm, D_att, D_w):
        super().__init__()
        self.W_Q = nn.Linear(D_plm, D_att)
        self.W_K = nn.Linear(D_w, D_att)
        self.W_V = nn.Linear(D_w, D_att)

    def forward(self, E, W_red):
        B, D_w, S, L = W_red.shape

        Q = self.W_Q(E)
        W_red_perm = W_red.permute(0, 3, 2, 1)
        W_red_flat = W_red_perm.reshape(B * L, S, D_w)

        K = self.W_K(W_red_flat)
        V = self.W_V(W_red_flat)

        Q_flat = Q.reshape(B * L, 1, -1)
        att = torch.softmax((Q_flat @ K.transpose(1, 2)) / math.sqrt(K.shape[-1]), dim=-1)
        ctx = att @ V

        ctx = ctx.reshape(B, L, -1)
        return ctx

class ComplexInteractionBlock(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.a = nn.Parameter(torch.ones(D))
        self.b = nn.Parameter(torch.zeros(D))
        self.fuse = nn.Linear(2 * D, D)

    def forward(self, R, I):
        R_ = self.a * R - self.b * I
        I_ = self.b * R + self.a * I
        out = torch.cat([R_, I_], dim=-1)
        return self.fuse(out)

class TinyMamba(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(D, 4 * D),
            nn.GELU(),
            nn.Linear(4 * D, D)
        )
        self.norm = nn.LayerNorm(D)

    def forward(self, x):
        return self.norm(x + self.ff(x))

class MambaLite(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2, conv_kernel=4):
        super().__init__()

        hidden = d_model * expand

        self.gate_proj = nn.Linear(d_model, hidden)
        self.x_proj = nn.Linear(d_model, hidden)

        self.dwconv = nn.Conv1d(
            hidden, hidden, kernel_size=conv_kernel,
            groups=hidden, padding=conv_kernel // 2
        )

        self.ss_out = nn.Linear(hidden, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        gate = torch.sigmoid(self.gate_proj(x))
        u = self.x_proj(x)

        h = u.transpose(1, 2)
        h = self.dwconv(h)
        h = h.transpose(1, 2)

        h = h * gate
        out = self.ss_out(h)

        return self.norm(x + out)

class MambaPython(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2, conv_kernel=3):
        super().__init__()

        self.d_model = d_model
        hidden = d_model * expand
        self.hidden = hidden
        self.d_state = d_state

        self.x_proj = nn.Linear(d_model, hidden)
        self.g_proj = nn.Linear(d_model, hidden)

        self.dwconv = nn.Conv1d(
            hidden, hidden,
            kernel_size=conv_kernel,
            padding=conv_kernel // 2,
            groups=hidden
        )

        self.A_raw = nn.Parameter(torch.randn(hidden, d_state))
        self.B = nn.Parameter(torch.randn(hidden, d_state))
        self.C = nn.Parameter(torch.randn(hidden, d_state))

        self.dt_proj = nn.Linear(d_model, hidden)
        self.out_proj = nn.Linear(hidden, d_model)
        self.norm = nn.LayerNorm(d_model)

    def discretize(self, A, B, dt):
        dt = dt.unsqueeze(-1)
        A_dt = A * dt
        A_d = torch.exp(A_dt)

        A_safe = A + 1e-6
        B_d = ((A_d - 1.0) / A_safe) * B

        return A_d, B_d

    def selective_scan(self, A_d, B_d, C, u):
        B_, L, H = u.shape
        S = A_d.shape[1]

        x = torch.zeros(B_, H, S, device=u.device)
        outputs = []

        for t in range(L):
            u_t = u[:, t, :].unsqueeze(-1)
            x = A_d.unsqueeze(0) * x + B_d.unsqueeze(0) * u_t

            y_t = torch.sum(x * C.unsqueeze(0), dim=-1)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)

    def forward(self, x):
        B, L, D = x.shape
        u = self.x_proj(x)
        g = torch.sigmoid(self.g_proj(x))

        h = u.transpose(1, 2)
        h = self.dwconv(h)
        h = h.transpose(1, 2)

        u = u * g + h * (1 - g)

        dt_full = F.softplus(self.dt_proj(x))
        dt = dt_full.mean(dim=(0, 1))
        dt = torch.clamp(dt, 0.001, 2.0)

        A = -F.softplus(self.A_raw)
        A_d, B_d = self.discretize(A, self.B, dt)

        y = self.selective_scan(A_d, B_d, self.C, u)
        out = self.out_proj(y)

        return self.norm(x + out)
