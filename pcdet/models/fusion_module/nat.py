import torch
import torch.nn as nn

from timm.models.layers import DropPath
#from natten.nattentorch import LegacyNeighborhoodAttention, LegacyNeighborhoodCrossAttention
from natten import NeighborhoodAttention, NeighborhoodCrossAttention

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
class Attention_Cross(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, q):
        # kv = self.to_kv(x).chunk(2, dim = -1)
        # k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), kv)
        B, H, W, C = x.shape
        N = H * W
        kv = self.to_kv(x).reshape(B, H, W, 2, self.heads, self.dim_head).permute(3, 0, 4, 1, 2, 5)
        k, v = kv[0], kv[1]
        q = q.reshape(B, H, W, self.heads, self.dim_head).permute(0, 3, 1, 2, 4)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        #out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # qkv = self.to_qkv(x).chunk(3, dim = -1)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        B, H, W, C = x.shape
        N = H * W
        qkv = self.to_qkv(x).reshape(B, H, W, 3, self.heads, self.dim_head).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        #out = rearrange(out, 'b h n d -> b n (h d)')
        out = out.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        return self.to_out(out)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class NATLayer(nn.Module):
    def __init__(self, dim, num_heads, kernel_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, layer_scale=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = NeighborhoodAttention(
            dim, kernel_size=kernel_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        #### global attention
        # self.attn = Attention(dim, heads = num_heads, dim_head =(dim//num_heads), dropout=attn_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        if not self.layer_scale:
            shortcut = x
            x = self.norm1(x)
            x = self.attn(x)
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + self.drop_path(self.gamma1 * x)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x


class NATCrossLayer(nn.Module):
    def __init__(self, dim, num_heads, kernel_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, layer_scale=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = NeighborhoodCrossAttention(
            dim, kernel_size=kernel_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        ##### global attention
        # self.attn = Attention_Cross(dim, heads = num_heads, dim_head =(dim//num_heads), dropout=attn_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)

    def forward(self, x, q_extra):
        if not self.layer_scale:
            shortcut = x
            x = self.norm1(x)
            x = self.attn(x, q_extra)
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x, q_extra)
        x = shortcut + self.drop_path(self.gamma1 * x)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x


class ConvDownsampler(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.reduction = nn.Conv2d(dim, 2 * dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        x = self.reduction(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = self.norm(x)
        return x


class NATBlock(nn.Module):
    def __init__(self, dim, depth, depth_cross, num_heads, kernel_size, downsample=True,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, layer_scale=None):
        super().__init__()
        self.dim = dim
        self.depth = depth

        self.blocks = nn.ModuleList([
            NATLayer(dim=dim,
                     num_heads=num_heads, kernel_size=kernel_size,
                     mlp_ratio=mlp_ratio,
                     qkv_bias=qkv_bias, qk_scale=qk_scale,
                     drop=drop, attn_drop=attn_drop,
                     drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                     norm_layer=norm_layer,
                     layer_scale=layer_scale)
            for i in range(depth)])

        self.cross_blocks = nn.ModuleList([
            NATCrossLayer(dim=dim,
                          num_heads=num_heads, kernel_size=kernel_size,
                          mlp_ratio=mlp_ratio,
                          qkv_bias=qkv_bias, qk_scale=qk_scale,
                          drop=drop, attn_drop=attn_drop,
                          drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                          norm_layer=norm_layer,
                          layer_scale=layer_scale)
            for i in range(depth_cross)])

        self.downsample = None if not downsample else ConvDownsampler(dim=dim * 2, norm_layer=norm_layer)

    def forward(self, x, x_multi):
        x_cross = x.clone()
        for blk in self.blocks:
            x = blk(x)

        for cross_blk in self.cross_blocks:
            x_cross = cross_blk(x_multi, x_cross)

        x = torch.cat([x, x_cross], dim=-1)

        if self.downsample is None:
            return x
        return self.downsample(x)


if __name__ == '__main__':
    nat_block = NATBlock(dim=64, depth=3, depth_cross=1, num_heads=8, kernel_size=7, downsample=False).cuda()
    x_in = torch.randn([1, 256, 256, 64]).cuda()
    x_multi = torch.randn([1, 256, 256, 64]).cuda()
    x_out = nat_block(x_in, x_multi)
    print(x_out.shape)


