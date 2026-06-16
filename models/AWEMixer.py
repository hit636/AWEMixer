import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt 

# =====================================================================================
# 1. 基础模块：RevIN (保持不变)
# =====================================================================================
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode: str):
        if mode == 'norm':
            self.mean = torch.mean(x, dim=1, keepdim=True).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            x = (x - self.mean) / self.stdev
            if self.affine: x = x * self.affine_weight + self.affine_bias
        elif mode == 'denorm':
            if self.affine: x = (x - self.affine_bias) / (self.affine_weight + 1e-10)
            x = x * self.stdev + self.mean
        return x

# =====================================================================================
# 2. Multi-Scale Temporal Representation (多尺度时域表示)
# 严格对齐 Eq 5-7: k_s = 2s+1, same padding (Reflection), 不改变序列长度
# =====================================================================================
class MultiScaleTemporalRepresentation(nn.Module):
    def __init__(self, num_scales, d_model):
        super().__init__()
        self.embedders = nn.ModuleList()
        # s 属于 1, ..., S
        for s in range(1, num_scales + 1):
            kernel_size = 2 * s + 1
            padding = kernel_size // 2  # Eq 5: p_s = d_s * (k_s - 1) / 2
            
            self.embedders.append(nn.Sequential(
                nn.ReflectionPad1d(padding), # 使用反射填充减少边界伪影
                nn.Conv1d(1, 16, kernel_size=kernel_size, stride=1),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1), # 自适应平均池化得到全局 Anchor
                nn.Flatten(),
                nn.Linear(16, d_model)
            ))

    def forward(self, x_ci):
        x_in = x_ci.unsqueeze(1) 
        # 返回形状 (B*C, S, D)
        return torch.stack([emb(x_in) for emb in self.embedders], dim=1) 

# =====================================================================================
# 3. Multi-Band Wavelet Representation (多频段小波表示)
# 严格对齐 Eq 9-11: 使用 Undecimated 小波滤波 (空洞卷积实现)，不插值
# =====================================================================================
class MultiBandWaveletRepresentation(nn.Module):
    def __init__(self, wavelet_name='db4', level=3):
        super().__init__()
        self.level = level
        wavelet = pywt.Wavelet(wavelet_name)
        
        # 提取低通和高通滤波器
        h0 = np.array(wavelet.dec_lo[::-1], dtype=np.float32) / np.sqrt(2)
        h1 = np.array(wavelet.dec_hi[::-1], dtype=np.float32) / np.sqrt(2)
        filters = np.stack([h0, h1], axis=0)
        
        self.register_buffer('filters', torch.tensor(filters).unsqueeze(1))
        self.filter_length = len(h0)

    def forward(self, x):
        x = x.unsqueeze(1).contiguous() 
        coeffs = []
        v_j = x

        for j in range(self.level):
            dilation = 2 ** j
            pad_size = (self.filter_length - 1) * dilation
            # 使用 circular 边界处理以规避时序变长，保持 L 不变
            v_j_padded = F.pad(v_j, (pad_size, 0), mode='circular') 

            res = F.conv1d(v_j_padded, self.filters, dilation=dilation)
            v_j, w_j = res[:, 0:1, :], res[:, 1:2, :]
            coeffs.append(w_j.squeeze(1)) 

        coeffs.append(v_j.squeeze(1)) 
        # 返回列表: [V_J, W_J, W_{J-1}, ..., W_1]
        return coeffs[::-1] 

# =====================================================================================
# 4. Frequency Router (频段路由器)
# 严格对齐 Eq 15-27: 严格的FFT频段划分，步长为1的局部突发能量窗口，以及跨子带特征归一化
# =====================================================================================
class FrequencyRouter(nn.Module):
    def __init__(self, num_bands):
        super().__init__()
        self.num_bands = num_bands
        self.router_mlp = nn.Sequential(
            nn.Linear(4 * num_bands, 64),
            nn.GELU(),
            nn.Linear(64, num_bands)
        )

    def forward(self, x_ci, w_coeffs_stack):
        B_C, Nb, L = w_coeffs_stack.shape

        # -------------------------------------------------------------
        # Descriptor 1: FFT Spectral Statistics (Eq 15-17)
        # -------------------------------------------------------------
        amp = torch.abs(torch.fft.rfft(x_ci, dim=-1)) # Shape: (B*C, K_fft + 1)
        K_fft = L // 2
        
        e_fft = torch.zeros(B_C, Nb, device=x_ci.device)
        total_amp_energy = torch.sum(amp ** 2, dim=-1) + 1e-8
        
        for j in range(1, Nb + 1):
            # 严格依据 Eq 16 进行频点粗略分配
            start_k = int((j - 1) * (K_fft + 1) / Nb)
            end_k = int(j * (K_fft + 1) / Nb)
            if j == Nb: end_k = K_fft + 1
            
            region_amp = amp[:, start_k:end_k]
            e_fft[:, j-1] = torch.sum(region_amp ** 2, dim=-1) / total_amp_energy

        # -------------------------------------------------------------
        # Descriptor 2: Global Wavelet Energy (Eq 18)
        # -------------------------------------------------------------
        e_wavelet = torch.mean(w_coeffs_stack ** 2, dim=-1) # (B*C, Nb)

        # -------------------------------------------------------------
        # Descriptor 3: Local Burst Energy (Eq 19-20)
        # -------------------------------------------------------------
        L_loc = 16 if L >= 16 else L
        # 滑动窗口 (stride=1) 计算局部能量
        # 相当于用大小为 L_loc 的 AvgPool 计算序列均方值
        q_j = F.avg_pool1d(w_coeffs_stack ** 2, kernel_size=L_loc, stride=1) 
        # 取最大的局部能量值 (max over valid windows)
        e_local, _ = q_j.max(dim=-1) # (B*C, Nb)

        # -------------------------------------------------------------
        # Descriptor 4: Band-wise Entropy (Eq 21-22)
        # -------------------------------------------------------------
        energy_t = w_coeffs_stack ** 2
        sum_energy = torch.sum(energy_t, dim=-1, keepdim=True) + 1e-8
        p = energy_t / sum_energy 
        # Eq 22: e_ent = - (1 / log(L)) * \sum( p_j * log(p_j + eps) )
        e_entropy = - (1.0 / np.log(L)) * torch.sum(p * torch.log(p + 1e-8), dim=-1) # (B*C, Nb)

        # -------------------------------------------------------------
        # Eq 24: Z-score Normalization Across Subbands (非常关键)
        # -------------------------------------------------------------
        def normalize_across_bands(e):
            mu = e.mean(dim=1, keepdim=True)
            sigma = e.std(dim=1, keepdim=True, unbiased=False) + 1e-8
            return (e - mu) / sigma

        e_fft_norm = normalize_across_bands(e_fft)
        e_wavelet_norm = normalize_across_bands(e_wavelet)
        e_local_norm = normalize_across_bands(e_local)
        e_entropy_norm = normalize_across_bands(e_entropy)

        # -------------------------------------------------------------
        # Eq 25-26: Concatenate and Route
        # -------------------------------------------------------------
        router_input = torch.cat([e_fft_norm, e_wavelet_norm, e_local_norm, e_entropy_norm], dim=-1) 
        alpha = F.softmax(self.router_mlp(router_input), dim=-1) 
        
        return alpha.unsqueeze(-1) # (B*C, N_b, 1)

# =====================================================================================
# 5. Coherent Gated Fusion Block (相干门控融合块)
# 严格对齐 Eq 28-30
# =====================================================================================
class LightweightSingleHeadAttention(nn.Module):
    """用于表示特征间交互的极轻量级单头交叉注意力机制"""
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.scale = d_model ** -0.5
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        Q, K, V = self.q_proj(q), self.k_proj(k), self.v_proj(v)
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) * self.scale
        attn_raw = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_raw)
        context = torch.matmul(attn_weights, V)
        return self.out_proj(context), attn_raw

class CoherentGatedFusionBlock(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.cross_attn = LightweightSingleHeadAttention(d_model, dropout=dropout)
        self.gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid() # 严格对齐 Eq 29
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, Z_t, H_w_weighted):
        # 交叉注意力获取频率上下文 Cf
        C_f, attn_weights = self.cross_attn(q=Z_t, k=H_w_weighted, v=H_w_weighted)
        
        # 兼容性门控 G 
        g = self.gate(torch.cat([Z_t, C_f], dim=-1))
        
        # 残差注入 Z_tf
        Z_tf = Z_t + g * C_f
        return self.norm(Z_tf), g, attn_weights

# =====================================================================================
# 6. Cross-Scale Mixer Block (跨尺度混合器)
# 对齐 Eq 31
# =====================================================================================
class CrossScaleMixerBlock(nn.Module):
    def __init__(self, num_scales, d_model, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mixer = nn.Sequential(
            nn.Linear(num_scales, num_scales * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(num_scales * 2, num_scales)
        )

    def forward(self, x):
        residual = x
        x = self.norm(x).transpose(1, 2)
        x = self.mixer(x).transpose(1, 2)
        return residual + x

# =====================================================================================
# 7. AWEMixer 完整主模型 (完全对齐 Figure 2)
# =====================================================================================
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = getattr(configs, 'd_model', 128)
        self.num_scales = getattr(configs, 'num_scales', 3)
        self.wavelet_level = getattr(configs, 'wavelet_level', 3)
        self.num_bands = self.wavelet_level + 1
        self.num_fusion_layers = getattr(configs, 'num_encoder_layers', 1) 
        
        self.revin_layer = RevIN(configs.enc_in) if getattr(configs, 'revin', True) else None
        
        # (a) Temporal Branch
        self.temporal_representation = MultiScaleTemporalRepresentation(self.num_scales, self.d_model)
        
        # (b) Wavelet Branch
        self.wavelet_representation = MultiBandWaveletRepresentation(
            wavelet_name=getattr(configs, 'wavelet', 'db4'), 
            level=self.wavelet_level
        )
        self.band_embedders = nn.ModuleList([
            nn.Linear(self.seq_len, self.d_model) for _ in range(self.num_bands)
        ])
        
        # (c) Router
        self.frequency_router = FrequencyRouter(self.num_bands)
        
        # (d) Fusion Block
        self.gating_backbone = nn.ModuleList([
            CoherentGatedFusionBlock(self.d_model, getattr(configs, 'dropout', 0.1))
            for _ in range(self.num_fusion_layers)
        ])
        
        # (e) Cross-scale & Pred
        self.cross_scale_mixer = CrossScaleMixerBlock(self.num_scales, self.d_model)
        
        # Eq 33 预测头
        self.prediction_head = nn.Linear(self.d_model, self.pred_len)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None, return_aux=False):
        B, L_in, C = x_enc.shape
        if self.revin_layer: x_enc = self.revin_layer(x_enc, 'norm')
        x_ci = x_enc.permute(0, 2, 1).reshape(B * C, L_in)
        
        # 1. 提取时间锚点特征 -> Eq 5-8
        Z_t = self.temporal_representation(x_ci) 
        
        # 2. 提取并映射小波子带特征 -> Eq 9-14
        wav_coeffs = self.wavelet_representation(x_ci)   
        w_coeffs_stack = torch.stack(wav_coeffs, dim=1) 
        H_w = torch.stack([self.band_embedders[i](wav_coeffs[i]) for i in range(self.num_bands)], dim=1) 
        
        # 3. 自适应频率加权 -> Eq 26-27
        alpha = self.frequency_router(x_ci, w_coeffs_stack) 
        H_w_weighted = H_w * alpha
        
        # 4. 相干门控注入 -> Eq 28-30
        Z_tf = Z_t
        gates, attns = [], []
        for block in self.gating_backbone:
            Z_tf, g, attn = block(Z_tf, H_w_weighted)
            if return_aux:
                gates.append(g), attns.append(attn)
            
        # 5. 跨尺度混合与均值池化 -> Eq 31-32
        Z_mix = self.cross_scale_mixer(Z_tf) 
        Z_final = Z_mix.mean(dim=1) 
        
        # 6. 预测头与反归一化 -> Eq 33
        Y_hat = self.prediction_head(Z_final)
        
        Y_hat = Y_hat.reshape(B, C, self.pred_len).permute(0, 2, 1)
        if self.revin_layer: Y_hat = self.revin_layer(Y_hat, 'denorm')
        
        if return_aux:
            return Y_hat, {"alpha": alpha, "gates": gates, "attentions": attns, "wavelet_coeffs": w_coeffs_stack}
            
        return Y_hat
