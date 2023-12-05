import torch
import torch.nn as nn

from timm.models.layers import DropPath
# from natten.nattentorch import LegacyNeighborhoodAttention
from natten import NeighborhoodAttention, NeighborhoodCrossAttention

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

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)

    def forward(self, x, x_mask=None):
        if not self.layer_scale:
            shortcut = x
            x = self.norm1(x)
            x = self.attn(x)
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            if x_mask is not None:
                x_out = x * x_mask
                # x_out = torch.zeros_like(x).cuda()
                # x_out[:, x_mask[:, 0], x_mask[:, 1], :] = x[:, x_mask[:, 0], x_mask[:, 1], :]
            else:
                x_out = x
            return x_out
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + self.drop_path(self.gamma1 * x)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        if x_mask is not None:
            x_out = x * x_mask
            # x_out = torch.zeros_like(x).cuda()
            # x_out[:, x_mask[:, 0], x_mask[:, 1], :] = x[:, x_mask[:, 0], x_mask[:, 1], :]
        else:
            x_out = x
        return x_out


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

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)

    def forward(self, x, q_extra, x_mask=None):
        if not self.layer_scale:
            shortcut = x
            x = self.norm1(x)
            x = self.attn(x, q_extra)
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            if x_mask is not None:
                x_out = x * x_mask
                # x_out = torch.zeros_like(x).cuda()
                # x_out[:, x_mask[:, 0], x_mask[:, 1], :] = x[:, x_mask[:, 0], x_mask[:, 1], :]
            else:
                x_out = x
            return x_out
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x, q_extra)
        x = shortcut + self.drop_path(self.gamma1 * x)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        if x_mask is not None:
            x_out = x * x_mask
            # x_out = torch.zeros_like(x).cuda()
            # x_out[:, x_mask[:, 0], x_mask[:, 1], :] = x[:, x_mask[:, 0], x_mask[:, 1], :]
        else:
            x_out = x
        return x_out


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

    def forward(self, x, x_multi, x_mask=None):
        x_cross = x.clone()
        # torch.set_printoptions(profile="full")
        # print(x_mask)
        # exit()
        for blk in self.blocks:
            x = blk(x, x_mask)

        for cross_blk in self.cross_blocks:
            x_cross = cross_blk(x_multi, x_cross, x_mask)

        x = torch.cat([x, x_cross], dim=-1)

        if self.downsample is None:
            return x
        return self.downsample(x)





if __name__ == '__main__':
    nat_block = NATBlock(dim=64, depth=3, depth_cross=1, num_heads=8, kernel_size=7, downsample=False).cuda()
    x_in = torch.randn([1, 256, 256, 64]).cuda()
    x_multi = torch.randn([1, 256, 256, 64]).cuda()
    print(x_in.shape)
    print('#####')
    x_out = nat_block(x_in, x_multi)
    print(x_out.shape)


    import random

    coords_x = torch.Tensor(random.sample(range(0, 256), 100)).long()
    coords_y = torch.Tensor(random.sample(range(0, 256), 100)).long()

    coords = torch.zeros([10, 2]).long()  # batch_dict['voxel_coords']

    coords[:, 0] = coords_x
    coords[:, 1] = coords_y

    x_out_with_coords = nat_block(x_in, x_multi, coords)
    print(x_out_with_coords.shape)



