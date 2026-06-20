from functools import partial
import math
import torch
import torch.nn as nn
import torchinfo
from timm.models.layers import trunc_normal_, DropPath
from timm.models.vision_transformer import Block
from util.pos_embed import get_2d_sincos_pos_embed
from spikingjelly.clock_driven import layer
import encoder
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Global timestep for Multispike quantization
# Keep this identical to the original code for fair comparison.
# ---------------------------------------------------------------------------
T = 4


# ===========================================================================
# Log-PE helper (paper-faithful)
# ===========================================================================

def _build_log_pe_2d(H: int, W: int) -> torch.Tensor:
    """
    2D Log-PE on a grid. Distance between tokens (y1,x1) and (y2,x2) is L1.
        R_{i,j} = ceil(log2((H+W-2)) - log2(|Δy| + |Δx| + 1)), clamped at >= 0
    Returns FloatTensor of shape (H*W, H*W).
    """
    if H * W <= 1:
        return torch.zeros(H * W, H * W, dtype=torch.float)

    ys, xs = torch.meshgrid(
        torch.arange(H, dtype=torch.float),
        torch.arange(W, dtype=torch.float),
        indexing="ij",
    )
    coords_y = ys.flatten().unsqueeze(0)   # (1, N)
    coords_x = xs.flatten().unsqueeze(0)   # (1, N)

    dy = (coords_y - coords_y.T).abs()     # (N, N)
    dx = (coords_x - coords_x.T).abs()     # (N, N)
    dist = dy + dx                          # L1

    max_d = float(H + W - 2)
    rpe = torch.ceil(
        torch.log2(torch.tensor(max_d, dtype=torch.float)) - torch.log2(dist + 1.0)
    )
    return rpe.clamp(min=0.0)


# ===========================================================================
# Spiking-neuron primitives (unchanged)
# ===========================================================================

class multispike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, lens=T):
        ctx.save_for_backward(input)
        ctx.lens = lens
        return torch.floor(torch.clamp(input, 0, lens) + 0.5)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp1 = 0 < input
        temp2 = input < ctx.lens
        return grad_input * temp1.float() * temp2.float(), None


class Multispike(nn.Module):
    def __init__(self, spike=multispike, norm=T):
        super().__init__()
        self.lens = norm
        self.spike = spike
        self.norm = norm

    def forward(self, inputs):
        return self.spike.apply(inputs) / self.norm


# ===========================================================================
# CNN building blocks (unchanged)
# ===========================================================================

def MS_conv_unit(in_channels, out_channels, kernel_size=1, padding=0, groups=1):
    return nn.Sequential(
        layer.SeqToANNContainer(
            encoder.SparseConv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=groups,
                bias=True,
            ),
            encoder.SparseBatchNorm2d(out_channels),
        )
    )


class MS_ConvBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        self.neuron1 = Multispike()
        self.conv1 = MS_conv_unit(dim, dim * mlp_ratio, 3, 1)
        self.neuron2 = Multispike()
        self.conv2 = MS_conv_unit(dim * mlp_ratio, dim, 3, 1)

    def forward(self, x, mask=None):
        short_cut = x
        x = self.neuron1(x)
        x = self.conv1(x)
        x = self.neuron2(x)
        x = self.conv2(x)
        return x + short_cut


class MS_MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0, layer=0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = Multispike()

        self.fc2_conv = nn.Conv1d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = Multispike()

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T_, B, C, N = x.shape

        x = self.fc1_lif(x)
        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T_, B, self.c_hidden, N).contiguous()

        x = self.fc2_lif(x)
        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T_, B, C, N).contiguous()

        return x


class RepConv(nn.Module):
    def __init__(self, in_channel, out_channel, bias=False):
        super().__init__()
        mid = int(in_channel * 1.5)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channel, mid, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(mid),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(mid, out_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(out_channel),
        )

    def forward(self, x):
        return self.conv2(self.conv1(x))


class RepConv2(nn.Module):
    def __init__(self, in_channel, out_channel, bias=False):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channel, in_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(in_channel),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(out_channel),
        )

    def forward(self, x):
        return self.conv2(self.conv1(x))


# ===========================================================================
# Original linearized attention (kept for baseline)
# ===========================================================================

class MS_Attention_Conv_qkv_id(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125
        self.sr_ratio = sr_ratio

        self.head_lif = Multispike()

        self.q_conv = nn.Sequential(RepConv(dim, dim), nn.BatchNorm1d(dim))
        self.k_conv = nn.Sequential(RepConv(dim, dim), nn.BatchNorm1d(dim))
        self.v_conv = nn.Sequential(RepConv(dim, dim * sr_ratio), nn.BatchNorm1d(dim * sr_ratio))

        self.q_lif = Multispike()
        self.k_lif = Multispike()
        self.v_lif = Multispike()

        self.attn_lif = Multispike()
        self.proj_conv = nn.Sequential(RepConv(sr_ratio * dim, dim), nn.BatchNorm1d(dim))

    def forward(self, x):
        T_, B, C, N = x.shape

        x = self.head_lif(x)
        x_for_qkv = x.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qkv).reshape(T_, B, C, N)
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.transpose(-1, -2).reshape(T_, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)

        k_conv_out = self.k_conv(x_for_qkv).reshape(T_, B, C, N)
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.transpose(-1, -2).reshape(T_, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)

        v_conv_out = self.v_conv(x_for_qkv).reshape(T_, B, self.sr_ratio * C, N)
        v_conv_out = self.v_lif(v_conv_out)
        v = v_conv_out.transpose(-1, -2).reshape(T_, B, N, self.num_heads, self.sr_ratio * C // self.num_heads).permute(0, 1, 3, 2, 4)

        x = k.transpose(-2, -1) @ v
        x = (q @ x) * self.scale
        x = x.transpose(3, 4).reshape(T_, B, self.sr_ratio * C, N)
        x = self.attn_lif(x)

        x = self.proj_conv(x.flatten(0, 1)).reshape(T_, B, C, N)
        return x


# ===========================================================================
# Quadratic attention + Log-PE only
# ===========================================================================
class MS_Attention_Quadratic_LogPE(nn.Module):
    """
    Fair ablation:
      - replace linearized K^T@V with quadratic Q@K^T
      - add 2D Log-PE bias (L1 distance on patch grid) to the full NxN attention map
      - learnable scalar gate on the PE bias (init=1.0)
      - keep everything else as close as possible to original
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        sr_ratio: int = 1,
        num_patches: int = 196,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.sr_ratio = sr_ratio

        # Keep fixed for fair comparison with original
        self.scale = 0.125

        self.head_lif = Multispike()
        self.q_conv = nn.Sequential(RepConv(dim, dim), nn.BatchNorm1d(dim))
        self.k_conv = nn.Sequential(RepConv(dim, dim), nn.BatchNorm1d(dim))
        self.v_conv = nn.Sequential(RepConv(dim, dim * sr_ratio), nn.BatchNorm1d(dim * sr_ratio))

        self.q_lif = Multispike()
        self.k_lif = Multispike()
        self.v_lif = Multispike()

        self.attn_lif = Multispike()
        self.proj_conv = nn.Sequential(RepConv(sr_ratio * dim, dim), nn.BatchNorm1d(dim))

        # 2D Log-PE: derive grid side from num_patches
        side = int(round(math.sqrt(num_patches)))
        assert side * side == num_patches, (
            f"num_patches={num_patches} must be a perfect square for 2D Log-PE"
        )
        log_pe = _build_log_pe_2d(side, side)
        self.register_buffer("log_pe", log_pe.unsqueeze(0).unsqueeze(0).unsqueeze(0))  # (1,1,1,N,N)

        # Learnable gate on the PE bias (init=1.0 so default behavior matches fixed bias)
        #self.pe_scale = nn.Parameter(torch.ones(1))
        self.pe_scale = nn.Parameter(torch.zeros(1))
        # self.pe_scale = nn.Parameter(torch.full((1,), 0.1)) try if unstable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T_, B, C, N = x.shape
        H = self.num_heads
        head_dim = C // H
        v_hdim = self.sr_ratio * C // H

        x = self.head_lif(x)
        x_flat = x.flatten(0, 1)

        q = self.q_lif(self.q_conv(x_flat).reshape(T_, B, C, N))
        k = self.k_lif(self.k_conv(x_flat).reshape(T_, B, C, N))
        v = self.v_lif(self.v_conv(x_flat).reshape(T_, B, self.sr_ratio * C, N))

        q = q.transpose(-1, -2).reshape(T_, B, N, H, head_dim).permute(0, 1, 3, 2, 4)
        k = k.transpose(-1, -2).reshape(T_, B, N, H, head_dim).permute(0, 1, 3, 2, 4)
        v = v.transpose(-1, -2).reshape(T_, B, N, H, v_hdim).permute(0, 1, 3, 2, 4)

        attn = torch.matmul(q, k.transpose(-2, -1))                  # (T,B,H,N,N)
        pe_bias = F.softplus(self.pe_scale) * self.log_pe
        attn = (attn + pe_bias) * self.scale

        x_out = torch.matmul(attn, v)                                # (T,B,H,N,v_hdim)
        x_out = x_out.transpose(3, 4).reshape(T_, B, self.sr_ratio * C, N)
        x_out = self.attn_lif(x_out)
        x_out = self.proj_conv(x_out.flatten(0, 1)).reshape(T_, B, C, N)
        return x_out

# ===========================================================================
# Downsampling (unchanged)
# ===========================================================================

class MS_DownSampling(nn.Module):
    def __init__(self, in_channels=2, embed_dims=256, kernel_size=3, stride=2, padding=1, first_layer=True):
        super().__init__()
        self.encode_conv = encoder.SparseConv2d(
            in_channels,
            embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.encode_bn = encoder.SparseBatchNorm2d(embed_dims)
        self.first_layer = first_layer
        if not first_layer:
            self.encode_spike = Multispike()

    def forward(self, x):
        T_, B, _, _, _ = x.shape

        if hasattr(self, "encode_spike"):
            x = self.encode_spike(x)

        x = self.encode_conv(x.flatten(0, 1))
        _, _, H, W = x.shape
        x = self.encode_bn(x).reshape(T_, B, -1, H, W)
        return x


# ===========================================================================
# Transformer block
# ===========================================================================

class MS_Block(nn.Module):
    def __init__(
        self,
        dim,
        choice,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
        init_values=1e-6,
        finetune=False,
        attn_type: str = "linearized",
        num_patches: int = 196,
    ):
        super().__init__()
        self.model = choice

        if self.model == "base":
            self.rep_conv = RepConv2(dim, dim)

        self.lif = Multispike()

        if attn_type == "quadratic_logpe":
            self.attn = MS_Attention_Quadratic_LogPE(
                dim=dim,
                num_heads=num_heads,
                sr_ratio=sr_ratio,
                num_patches=num_patches,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        else:
            self.attn = MS_Attention_Conv_qkv_id(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                sr_ratio=sr_ratio,
            )

        self.finetune = finetune
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.mlp = MS_MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

        if self.finetune:
            self.layer_scale1 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
            self.layer_scale2 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        T_, B, C, N = x.shape

        if self.model == "base":
            x = x + self.rep_conv(self.lif(x).flatten(0, 1)).reshape(T_, B, C, N)

        if self.finetune:
            x = x + self.drop_path(self.attn(x) * self.layer_scale1.unsqueeze(0).unsqueeze(0).unsqueeze(-1))
            x = x + self.drop_path(self.mlp(x) * self.layer_scale2.unsqueeze(0).unsqueeze(0).unsqueeze(-1))
        else:
            x = x + self.attn(x)
            x = x + self.mlp(x)

        return x


# ===========================================================================
# DeepMIM + SpikMAE encoder-decoder with quadratic + Log-PE
# ===========================================================================

class SpikmaeDeepMIM(nn.Module):
    def __init__(
        self,
        T: int = 1,
        choice=None,
        img_size_h: int = 224,
        img_size_w: int = 224,
        patch_size: int = 16,
        embed_dim=(128, 256, 512),
        num_heads: int = 8,
        mlp_ratios: int = 4,
        in_channels: int = 3,
        qk_scale=None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        num_classes: int = 1000,
        qkv_bias: bool = False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths: int = 12,
        sr_ratios: int = 1,
        decoder_embed_dim: int = 256,
        decoder_depth: int = 4,
        decoder_num_heads: int = 4,
        mlp_ratio: float = 4.0,
        norm_pix_loss: bool = False,
        nb_classes: int = 1000,
        attn_type: str = "quadratic_logpe",
    ):
        super().__init__()

        # Fairness: keep the original temporal behavior
        self.T = 1
        self.depths = depths
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.num_patches = (img_size_h // patch_size) * (img_size_w // patch_size)
        self.downsample_ratio = patch_size
        self.aux_decoder_depth = 2 # 2 half of decoder_depth

        # DeepMIM supervision points here assume a 12-block encoder
        assert depths == 12, "This fair DeepMIM setup assumes depths=12."

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]

        # ---------------- CNN stages ----------------
        self.downsample1_1 = MS_DownSampling(
            in_channels=in_channels,
            embed_dims=embed_dim[0] // 2,
            kernel_size=7,
            stride=2,
            padding=3,
            first_layer=True,
        )
        self.ConvBlock1_1 = nn.ModuleList([MS_ConvBlock(dim=embed_dim[0] // 2, mlp_ratio=mlp_ratios)])

        self.downsample1_2 = MS_DownSampling(
            in_channels=embed_dim[0] // 2,
            embed_dims=embed_dim[0],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
        )
        self.ConvBlock1_2 = nn.ModuleList([MS_ConvBlock(dim=embed_dim[0], mlp_ratio=mlp_ratios)])

        self.downsample2 = MS_DownSampling(
            in_channels=embed_dim[0],
            embed_dims=embed_dim[1],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
        )
        self.ConvBlock2_1 = nn.ModuleList([MS_ConvBlock(dim=embed_dim[1], mlp_ratio=mlp_ratios)])
        self.ConvBlock2_2 = nn.ModuleList([MS_ConvBlock(dim=embed_dim[1], mlp_ratio=mlp_ratios)])

        self.downsample3 = MS_DownSampling(
            in_channels=embed_dim[1],
            embed_dims=embed_dim[2],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
        )

        # ---------------- Transformer stages ----------------
        self.block3 = nn.ModuleList([
            MS_Block(
                dim=embed_dim[2],
                choice=choice,
                num_heads=num_heads,
                mlp_ratio=mlp_ratios,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[j],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios,
                finetune=False,
                attn_type=attn_type,
                num_patches=self.num_patches,
            )
            for j in range(depths)
        ])

        self.norm = nn.BatchNorm1d(embed_dim[-1])
        self.aux_norm6 = nn.BatchNorm1d(embed_dim[-1])
        self.aux_norm8 = nn.BatchNorm1d(embed_dim[-1])
        self.aux_norm10 = nn.BatchNorm1d(embed_dim[-1])

        # ---------------- Primary decoder ----------------
        self.decoder_embed = nn.Linear(embed_dim[-1], decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, decoder_embed_dim), requires_grad=False
        )
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=False, norm_layer=norm_layer)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_channels, bias=True)

        # ---------------- Auxiliary decoders ----------------
        # Same architecture/depth as primary for fair comparison,
        # but independent parameters as in DeepMIM.
        for suffix in ("6", "8", "10"):
            setattr(self, f"aux_decoder_embed{suffix}", nn.Linear(embed_dim[-1], decoder_embed_dim, bias=True))
            setattr(self, f"aux_mask_token{suffix}", nn.Parameter(torch.zeros(1, 1, decoder_embed_dim)))
            setattr(
                self,
                f"aux_decoder_pos_embed{suffix}",
                nn.Parameter(torch.zeros(1, self.num_patches, decoder_embed_dim), requires_grad=False),
            )
            setattr(
                self,
                f"aux_decoder_blocks{suffix}",
                nn.ModuleList([
                    Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=False, norm_layer=norm_layer)
                    for _ in range(self.aux_decoder_depth)
                ]),
            )
            setattr(self, f"aux_decoder_norm{suffix}", norm_layer(decoder_embed_dim))
            setattr(
                self,
                f"aux_decoder_pred{suffix}",
                nn.Linear(decoder_embed_dim, patch_size ** 2 * in_channels, bias=True),
            )

        self.initialize_weights()

    # ----------------------------------------------------------------------
    def initialize_weights(self):
        n_side = int(self.num_patches ** 0.5)

        # Decoder positional embeddings only (same style as original code)
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], n_side, cls_token=False
        )
        decoder_pos_embed = torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)

        self.decoder_pos_embed.data.copy_(decoder_pos_embed)
        for suffix in ("6", "8", "10"):
            getattr(self, f"aux_decoder_pos_embed{suffix}").data.copy_(decoder_pos_embed)

        nn.init.normal_(self.mask_token, std=0.02)
        for suffix in ("6", "8", "10"):
            nn.init.normal_(getattr(self, f"aux_mask_token{suffix}"), std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # ----------------------------------------------------------------------
    def random_masking(self, x, mask_ratio):
        """
        Pixel-level masking, same style as original code.
        """
        T_dim, N, _, _, _ = x.shape  # N = batch size here
        L = self.num_patches
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        active = torch.ones([N, L], device=x.device)
        active[:, len_keep:] = 0
        active = torch.gather(active, dim=1, index=ids_restore)

        return ids_keep, active, ids_restore

    # ----------------------------------------------------------------------
    def forward_encoder(self, x, mask_ratio=0.5):
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)

        ids_keep, active, ids_restore = self.random_masking(x, mask_ratio)
        B, N = active.shape
        side = int(N ** 0.5)
        active_b1ff = active.reshape(B, 1, side, side)

        encoder._cur_active = active_b1ff

        active_hw = active_b1ff.repeat_interleave(self.downsample_ratio, 2).repeat_interleave(self.downsample_ratio, 3)
        active_hw = active_hw.unsqueeze(0)
        x = x * active_hw

        x = self.downsample1_1(x)
        for blk in self.ConvBlock1_1:
            x = blk(x)

        x = self.downsample1_2(x)
        for blk in self.ConvBlock1_2:
            x = blk(x)

        x = self.downsample2(x)
        for blk in self.ConvBlock2_1:
            x = blk(x)
        for blk in self.ConvBlock2_2:
            x = blk(x)

        x = self.downsample3(x)
        x = x.flatten(3)  # (T, B, C, N)
        # No encoder pos_embed here for fair comparison with original code

        feat6 = feat8 = feat10 = None

        # Explicit 12-block DeepMIM supervision points
        idx6, idx8, idx10 = 5, 7, 9

        for i, blk in enumerate(self.block3):
            x = blk(x)
            if i == idx6:
                feat6 = x
            elif i == idx8:
                feat8 = x
            elif i == idx10:
                feat10 = x

        feat12 = x

        # Same temporal aggregation style as original: x.mean(0)
        feat6 = self.aux_norm6(feat6.mean(0)).transpose(-1, -2).contiguous()
        feat8 = self.aux_norm8(feat8.mean(0)).transpose(-1, -2).contiguous()
        feat10 = self.aux_norm10(feat10.mean(0)).transpose(-1, -2).contiguous()
        latent = self.norm(feat12.mean(0)).transpose(-1, -2).contiguous()

        return latent, feat6, feat8, feat10, active, ids_restore, active_hw

    # ----------------------------------------------------------------------
    def _decoder_forward(self, x, ids_restore, embed_layer, mask_token, pos_embed, blocks, norm_layer, pred_layer):
        """
        Keep decoder logic aligned with the original code style.
        """
        x = embed_layer(x)  # (B, N, C)

        mask_tokens = mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x_ = torch.gather(
            x_,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]),
        )
        x = x_ + pos_embed

        for blk in blocks:
            x = blk(x)

        x = norm_layer(x)
        x = pred_layer(x)
        return x

    def forward_decoder(self, latent, ids_restore):
        return self._decoder_forward(
            latent,
            ids_restore,
            self.decoder_embed,
            self.mask_token,
            self.decoder_pos_embed,
            self.decoder_blocks,
            self.decoder_norm,
            self.decoder_pred,
        )

    def forward_aux_decoder(self, feat, ids_restore, suffix):
        return self._decoder_forward(
            feat,
            ids_restore,
            getattr(self, f"aux_decoder_embed{suffix}"),
            getattr(self, f"aux_mask_token{suffix}"),
            getattr(self, f"aux_decoder_pos_embed{suffix}"),
            getattr(self, f"aux_decoder_blocks{suffix}"),
            getattr(self, f"aux_decoder_norm{suffix}"),
            getattr(self, f"aux_decoder_pred{suffix}"),
        )

    # ----------------------------------------------------------------------
    def patchify(self, imgs):
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], self.in_channels, h, p, w, p)
        x = torch.einsum("nchpwq->nhwpqc", x)
        return x.reshape(imgs.shape[0], h * w, p ** 2 * self.in_channels)

    def unpatchify(self, x):
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(x.shape[0], h, w, p, p, self.in_channels)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], self.in_channels, h * p, h * p)

    # ----------------------------------------------------------------------
    def forward_loss(self, imgs, pred, mask):
        inp = self.patchify(imgs)
        mean = inp.mean(dim=-1, keepdim=True)
        var = (inp.var(dim=-1, keepdim=True) + 1e-6) ** 0.5
        inp = (inp - mean) / var

        l2 = ((pred - inp) ** 2).mean(dim=2)
        non_active = mask.logical_not().int().view(mask.shape[0], -1)
        loss = l2.mul_(non_active).sum() / (non_active.sum() + 1e-8)
        return loss, mean, var

    # ----------------------------------------------------------------------
    def forward(self, imgs, mask_ratio=0.5, vis=False):
        latent, feat6, feat8, feat10, active, ids_restore, active_hw = self.forward_encoder(imgs, mask_ratio)
        self.aux_weights = {"6": 0.1, "8": 0.3, "10": 0.5}

        rec6 = self.forward_aux_decoder(feat6, ids_restore, "6")
        rec8 = self.forward_aux_decoder(feat8, ids_restore, "8")
        rec10 = self.forward_aux_decoder(feat10, ids_restore, "10")
        rec = self.forward_decoder(latent, ids_restore)

        loss6, _, _ = self.forward_loss(imgs, rec6, active)
        loss8, _, _ = self.forward_loss(imgs, rec8, active)
        loss10, _, _ = self.forward_loss(imgs, rec10, active)
        recon, mean, var = self.forward_loss(imgs, rec, active)

        #total_loss = recon + loss6 + loss8 + loss10
        total_loss = (
                recon
                + self.aux_weights["10"] * loss10
                + self.aux_weights["8"]  * loss8
                + self.aux_weights["6"]  * loss6
            )

        if vis:
            masked_bchw = imgs * active_hw.flatten(0, 1)
            rec_bchw = self.unpatchify(rec * var + mean)
            rec_or_inp = torch.where(active_hw.flatten(0, 1).bool(), imgs, rec_bchw)
            return imgs, masked_bchw, rec_or_inp

        return total_loss, recon


# ===========================================================================
# Factory functions
# ===========================================================================

def spikmae_deepmim_12_512(**kwargs):
    return SpikmaeDeepMIM(
        T=1,
        choice="base",
        img_size_h=224,
        img_size_w=224,
        patch_size=16,
        embed_dim=[128, 256, 512],
        num_heads=8,
        mlp_ratios=4,
        in_channels=3,
        num_classes=1000,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=12,
        sr_ratios=1,
        decoder_embed_dim=256,
        decoder_depth=4,
        decoder_num_heads=4,
        attn_type="quadratic_logpe",
        **kwargs,
    )


def spikmae_deepmim_12_768(**kwargs):
    return SpikmaeDeepMIM(
        T=1,
        choice="large",
        img_size_h=224,
        img_size_w=224,
        patch_size=16,
        embed_dim=[192, 384, 768],
        num_heads=8,
        mlp_ratios=4,
        in_channels=3,
        num_classes=1000,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=12,
        sr_ratios=1,
        decoder_embed_dim=256,
        decoder_depth=4,
        decoder_num_heads=4,
        attn_type="quadratic_logpe",
        **kwargs,
    )


def spikmae_deepmim_12_768_no_rpe(**kwargs):
    return SpikmaeDeepMIM(
        T=1,
        choice="large",
        img_size_h=224,
        img_size_w=224,
        patch_size=16,
        embed_dim=[192, 384, 768],
        num_heads=8,
        mlp_ratios=4,
        in_channels=3,
        num_classes=1000,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=12,
        sr_ratios=1,
        decoder_embed_dim=256,
        decoder_depth=4,
        decoder_num_heads=4,
        attn_type="linearized",
        **kwargs,
    )


# ===========================================================================
# Quick sanity check
# ===========================================================================

if __name__ == "__main__":
    lpe = _build_log_pe_2d(14, 14)
    assert lpe.shape == (196, 196)
    # self-distance = ceil(log2(26) - log2(1)) = 5
    assert lpe[0, 0].item() == 5.0, f"2D Log-PE diagonal should be 5, got {lpe[0, 0]}"
    # token 0=(0,0) and token 14=(1,0) -> L1 dist 1, same as token 1=(0,1)
    assert lpe[0, 14].item() == lpe[0, 1].item(), "vertical and horizontal neighbors should match"
    # far corner (0,0) -> (13,13), L1=26 -> ceil(log2(26)-log2(27))=0
    assert lpe[0, 195].item() == 0.0, f"far corner should be 0, got {lpe[0, 195]}"
    assert (lpe == lpe.T).all(), "Log-PE must be symmetric"
    print(f"2D Log-PE (14x14): max={lpe.max():.0f}, min={lpe.min():.0f} ✓")

    model = spikmae_deepmim_12_768()
    x = torch.randn(2, 3, 224, 224)
    loss = model(x, mask_ratio=0.50)
    print(f"DeepMIM loss (forward pass): {loss.item():.4f} ✓")
