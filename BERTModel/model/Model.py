import torch
import math
import torch.nn.functional as F 
import torch.nn as nn
from BERTModel.configs.config import default_config

#=======================================================================================================
#                                                                                     Model Architecture
#=======================================================================================================

class ResidualBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size)
        )
        
    def forward(self, x):
        return x + self.layers(x)
    


class DNABERTClassifier(nn.Module):
    def __init__(self, base_model, hidden_size=768, config=default_config):
        super().__init__()
        
        self.base = base_model
        self.config = config


        # === Attention ===
        self.token_attention = TokenAttention(hidden_size, num_heads=config.model.num_attention_heads, dropout=0.1, config=config)

        # === SNP Temperature ===
        self.token_snp_temperature = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.token_conf_temperature = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.cons_temperature = nn.Parameter(torch.tensor(1.0), requires_grad=True)

        # === Epigenetic Temperature ===
        self.gerp_temperature = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.rna_temperature = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.dnase_temperature = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.h3k27ac_temperature = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.h3k4me3_temperature = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.h3k36me3_temperature = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.h3k9me3_temperature = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.h3k27me3_temperature = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.h2afz_temperature = nn.Parameter(torch.tensor(1.0), requires_grad=True)

        # Add temperature constraints
        self.min_temperature = 0.001  # Prevent division by near-zero
        self.max_temperature = 1000.0  # Prevent extreme scaling


        # === SNP-specific processing ===
        self.snp_backbone = nn.Sequential(
            ResidualBlock(hidden_size),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.2),
            ResidualBlock(hidden_size),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # === UPGRADED: Multi-Stage SNP Heads with Wider Intermediate ===
        self.snp_heads = nn.ModuleDict({
            'snp_detection': nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),  # Wider intermediate
                nn.LayerNorm(hidden_size // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 2, 2)  # Final classification
            ),
            'snp_confidence': nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.LayerNorm(hidden_size // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 2, 1),
                nn.Sigmoid()
            ),
            'snp_consequence': nn.Sequential(
                ResidualBlock(hidden_size),
                nn.Dropout(0.2),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.LayerNorm(hidden_size // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 2, hidden_size // 4),
                nn.GELU(),
                nn.Linear(hidden_size // 4, 1),
                nn.Sigmoid()
            )
        })


        # === Epigenetic features backbone ===
        self.epigenetic_backbone = nn.Sequential(
            ResidualBlock(hidden_size),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.2),
            ResidualBlock(hidden_size),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
        )


        # === Feature-Specific Epigenetic Heads ===
        # Group related features for better learning
        common_epi_head = nn.Sequential(
            ResidualBlock(hidden_size),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        
        self.epigenetic_heads = nn.ModuleDict({
            # Accessibility features (more complex)
            'EncodetotalRNA_sum': nn.Sequential(
                common_epi_head,
                nn.Linear(hidden_size // 2, hidden_size // 4),
                nn.GELU(),
                nn.Linear(hidden_size // 4, 1),
                nn.Sigmoid()
            ),
            'EncodeDNase_sum': nn.Sequential(
                common_epi_head,
                nn.Linear(hidden_size // 2, hidden_size // 4),
                nn.GELU(),
                nn.Linear(hidden_size // 4, 1),
                nn.Sigmoid()
            ),
            
            # Histone marks (shared pathway)
            **{mark: nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 3),  # Wider for histone complexity
                nn.LayerNorm(hidden_size // 3),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 3, 1),
                nn.Sigmoid()
            ) for mark in ['EncodeH3K27ac_sum', 'EncodeH3K4me3_sum', 'EncodeH3K36me3_sum', 
                          'EncodeH3K9me3_sum', 'EncodeH3K27me3_sum', 'EncodeH2AFZ_sum']},
            
            # Conservation (specialized)
            'gerp': nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.LayerNorm(hidden_size // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 2, hidden_size // 4),
                nn.GELU(),
                nn.Linear(hidden_size // 4, 1),
                nn.Sigmoid()
            ),
        })

    def get_safe_temperature(self, temp_param):
        """Clamp temperature to safe range"""
        return torch.clamp(temp_param, self.min_temperature, self.max_temperature)

    def forward(self, input_ids, attention_mask):

        # === Base model ===
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs[0]  # [batch, seq_length, hidden_size]
        

        # === Token-level attention ===
        token_attention = self.token_attention(hidden_states, attention_mask)
        snp_features = self.snp_backbone(token_attention)
        epigenetic_features = self.epigenetic_backbone(token_attention)

        # === Token level logits ===
        token_conf = self.snp_heads['snp_confidence'](snp_features) / self.get_safe_temperature(self.token_conf_temperature)
        token_snp = self.snp_heads['snp_detection'](snp_features) / self.get_safe_temperature(self.token_snp_temperature)
        token_cons = self.snp_heads['snp_consequence'](snp_features) / self.get_safe_temperature(self.cons_temperature)
        
        # === Epigenetic features (FIXED: Use safe temperatures) ===
        token_gerp = self.epigenetic_heads['gerp'](epigenetic_features) / self.get_safe_temperature(self.gerp_temperature)
        token_rna = self.epigenetic_heads['EncodetotalRNA_sum'](epigenetic_features) / self.get_safe_temperature(self.rna_temperature)
        token_dnase = self.epigenetic_heads['EncodeDNase_sum'](epigenetic_features) / self.get_safe_temperature(self.dnase_temperature)
        token_h3k27ac = self.epigenetic_heads['EncodeH3K27ac_sum'](epigenetic_features) / self.get_safe_temperature(self.h3k27ac_temperature)
        token_h3k4me3 = self.epigenetic_heads['EncodeH3K4me3_sum'](epigenetic_features) / self.get_safe_temperature(self.h3k4me3_temperature)
        token_h3k36me3 = self.epigenetic_heads['EncodeH3K36me3_sum'](epigenetic_features) / self.get_safe_temperature(self.h3k36me3_temperature)
        token_h3k9me3 = self.epigenetic_heads['EncodeH3K9me3_sum'](epigenetic_features) / self.get_safe_temperature(self.h3k9me3_temperature)
        token_h3k27me3 = self.epigenetic_heads['EncodeH3K27me3_sum'](epigenetic_features) / self.get_safe_temperature(self.h3k27me3_temperature)
        token_h2afz = self.epigenetic_heads['EncodeH2AFZ_sum'](epigenetic_features) / self.get_safe_temperature(self.h2afz_temperature)


        return {
            'token_snp': token_snp,
            'token_conf': token_conf,
            'token_cons': token_cons,
            'token_gerp': token_gerp,
            'token_rna': token_rna,
            'token_dnase': token_dnase,
            'token_h3k27ac': token_h3k27ac,
            'token_h3k4me3': token_h3k4me3,
            'token_h3k36me3': token_h3k36me3,
            'token_h3k9me3': token_h3k9me3,
            'token_h3k27me3': token_h3k27me3,
            'token_h2afz': token_h2afz,
        }
    

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000, config=default_config):
        super().__init__()
        # Fix device handling - don't specify device in initialization
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        self.max_seq_len_cached = config.training.max_token_length
        # Initialize cache tensors without device specification
        self._create_cache(self.max_seq_len_cached)
    
    def _create_cache(self, seq_len):
        t = torch.arange(seq_len).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
    
    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.size(1)  # Get sequence length from input tensor
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )
    
    def _set_cos_sin_cache(self, seq_len, device):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)


class TokenAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=16, dropout=0.1, config=default_config):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.config = config
        
        # === Biologically-Motivated Multi-Scale Windows ===
        self.motif_window = 16      # Transcription factor binding sites
        self.regulatory_window = 64  # Enhancer/promoter regions
        self.domain_window = 256     # Chromatin domains
        
        # === Memory-Efficient Shared Projections ===
        self.qkv_proj = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # === Rotary Embeddings (Position-Critical for SNPs) ===
        self.rotary_emb = RotaryEmbedding(self.head_dim, base=10000, config=config)

        # === Enhanced Learnable Scaling ===
        # Multi-scale attention weights (learnable)
        self.scale_weights = nn.Parameter(torch.tensor([0.4, 0.35, 0.25]), requires_grad=True)
        
        # SNP modulation strengths (learnable instead of fixed)
        self.snp_modulation_weights = nn.Parameter(torch.tensor([0.5, 0.3, 0.2]), requires_grad=True)
        
        # Temperature for scale weights (prevents over-sharpening)
        self.scale_temperature = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        
        # === SNP-Aware Attention Gating ===
        self.snp_attention_gate = nn.Sequential(
            ResidualBlock(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, num_heads),
            nn.Sigmoid()
        )
        
        # === Feedforward ===
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
        # === Layer Norms ===
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # === Dropout ===
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        """
        x: [B, T, H] - Hidden states
        mask: [B, T] - Attention mask
        Returns: (output, global_tokens) for compatibility
        """
        B, T, H = x.shape
        
        # === Input Validation ===
        if T > self.config.training.max_token_length:
            raise ValueError(f"Sequence length {T} exceeds max_token_length {self.config.training.max_token_length}")
        
        # === Residual Connection 1: Multi-Scale Attention ===
        normed_x = self.norm1(x)
        attn_out = self.multi_scale_attention(normed_x, mask)
        x = x + self.dropout(attn_out)
        
        # === Residual Connection 2: Feedforward ===
        x = x + self.dropout(self.ffn(self.norm2(x)))
        
        return x
    
    def multi_scale_attention(self, x, mask):
        """Memory-efficient multi-scale attention for nucleotide sequences"""
        B, T, H = x.shape
        
        # === Single QKV Projection (Memory Efficient) ===
        qkv = self.qkv_proj(x).view(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, T, D]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # === Apply Rotary Embeddings (Critical for SNP Position Sensitivity) ===
        cos, sin = self.rotary_emb(q, seq_len=T)
        q, k = self.apply_rotary_pos_emb(q, k, cos, sin)
        
        # === SNP-Aware Attention Modulation ===
        snp_gates = self.snp_attention_gate(x)  # [B, T, num_heads]
        snp_gates = snp_gates.transpose(1, 2).unsqueeze(-1)  # [B, H, T, 1]
        
        # === Multi-Scale Attention Computation ===
        attention_outputs = []
        
        # 1. Motif-Level Attention
        motif_out = self.windowed_attention(q, k, v, mask, self.motif_window)
        motif_out = motif_out * (1 + snp_gates * self.snp_modulation_weights[0])
        attention_outputs.append(motif_out)
        
        # 2. Regulatory-Level Attention
        reg_out = self.windowed_attention(q, k, v, mask, self.regulatory_window)
        reg_out = reg_out * (1 + snp_gates * self.snp_modulation_weights[1])
        attention_outputs.append(reg_out)
        
        # 3. Domain-Level Attention
        domain_out = self.landmark_attention(q, k, v, mask)
        domain_out = domain_out * (1 + snp_gates * self.snp_modulation_weights[2])
        attention_outputs.append(domain_out)
        
        # === Temperature-Scaled Learnable Fusion ===
        scale_weights = F.softmax(self.scale_weights / self.scale_temperature, dim=0)
        fused_output = sum(w * out for w, out in zip(scale_weights, attention_outputs))
        
        # === Output Projection ===
        output = self.out_proj(fused_output.view(B, T, H))
        return output


    def windowed_attention(self, q, k, v, mask, window_size):
        """O(n) efficient sliding window attention"""
        B, H, T, D = q.shape
        
        # Ensure odd window size
        if window_size % 2 == 0:
            window_size += 1
        pad_size = window_size // 2
        
        # Pad k, v tensors
        k_pad = F.pad(k, (0, 0, pad_size, pad_size))  # [B, H, T+2*pad, D]
        v_pad = F.pad(v, (0, 0, pad_size, pad_size))  # [B, H, T+2*pad, D]
        
        # Create sliding windows using unfold (vectorized)
        k_windows = k_pad.unfold(2, window_size, 1)  # [B, H, T, D, W]
        v_windows = v_pad.unfold(2, window_size, 1)  # [B, H, T, D, W]
        
        # Reshape for batch attention computation
        k_windows = k_windows.permute(0, 1, 2, 4, 3).contiguous()  # [B, H, T, W, D]
        v_windows = v_windows.permute(0, 1, 2, 4, 3).contiguous()  # [B, H, T, W, D]
        
        # Compute attention for all windows at once
        q_expanded = q.unsqueeze(3)  # [B, H, T, 1, D]
        
        # Attention scores: [B, H, T, 1, D] × [B, H, T, D, W] -> [B, H, T, W]
        scores = torch.matmul(q_expanded, k_windows.transpose(-2, -1)).squeeze(3) / math.sqrt(D)
        
        # Apply mask if provided
        if mask is not None:
            mask_pad = F.pad(mask.float(), (pad_size, pad_size), value=0.0)
            mask_windows = mask_pad.unfold(1, window_size, 1)  # [B, T, W]
            mask_windows = mask_windows.unsqueeze(1).expand(B, H, T, window_size)
            scores = scores.masked_fill(mask_windows == 0, float('-inf'))
        
        # Compute attention weights and output
        attn_weights = F.softmax(torch.clamp(scores, -10, 10), dim=-1)  # [B, H, T, W]
        output = torch.matmul(attn_weights.unsqueeze(3), v_windows).squeeze(3)  # [B, H, T, D]
        
        return output
    
    def landmark_attention(self, q, k, v, mask):
        """Landmark-based attention for very long sequences"""
        B, H, T, D = q.shape
        
        # Select landmarks (every 4th position + start/end)
        landmark_stride = max(1, T // 128)
        landmark_indices = torch.arange(0, T, landmark_stride, device=q.device)
        
        # Add start and end positions as landmarks
        landmark_indices = torch.cat([
            torch.tensor([0], device=q.device),
            landmark_indices[1:-1] if len(landmark_indices) > 2 else landmark_indices[1:],
            torch.tensor([T-1], device=q.device) if T > 1 else torch.tensor([], device=q.device)
        ])
        landmark_indices = torch.unique(landmark_indices)
        
        k_landmarks = k[:, :, landmark_indices, :]  # [B, H, L, D]
        v_landmarks = v[:, :, landmark_indices, :]  # [B, H, L, D]
        
        # Attention to landmarks only
        scores = torch.matmul(q, k_landmarks.transpose(-2, -1)) / math.sqrt(D)
        
        if mask is not None:
            landmark_mask = mask[:, landmark_indices].unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~landmark_mask.bool(), float('-inf'))
        
        attn_weights = F.softmax(torch.clamp(scores, -10, 10), dim=-1)
        return torch.matmul(attn_weights, v_landmarks)
    
    def apply_rotary_pos_emb(self, q, k, cos, sin):
        """Apply rotary position embeddings"""
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed
    
    def rotate_half(self, x):
        """Rotates half the hidden dims of the input"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)



