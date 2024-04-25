import math
from functools import partial
import numpy as np
import timm

import torch
from torch import nn
import torch.nn.functional as F

from Models import pvt_v2, metaformer
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
from utils.FourierUpsampling import frescat

class RB(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.):
        super().__init__()

        self.in_ch = in_channels
        self.out_ch = out_channels

        self.in_layers = nn.Sequential(
            # nn.GroupNorm(32, in_channels),

            # nn.SiLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(in_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            # nn.GELU(),
            # nn.ReLU(),
        )

        self.middle_layers = nn.Sequential(
            # nn.GroupNorm(32, in_channels),

            # nn.SiLU(),
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, padding=1),
            # nn.BatchNorm2d(in_channels * 2, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            # nn.GELU(),
            # nn.ReLU(),
        )

        self.out_layers = nn.Sequential(
            # nn.GroupNorm(32, out_channels),

            # nn.SiLU(),
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            # nn.GELU(),
            # nn.ReLU(),
        )

        self.gelu = nn.GELU()
        self.BN = nn.BatchNorm2d(in_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.dropout = nn.Dropout(dropout_rate)

        if out_channels == in_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # self.skip = nn.Identity()

    def forward(self, x):
        # stage-conv
        # if self.in_ch / 32 == 1:
        #     h = self.in_layers(x)
        #     h = self.out_layers(h)
        #     h = self.out_layers(h)
        # elif self.in_ch / 32 == 2:
        #     h = self.in_layers(x)
        #     h = self.out_layers(h)
        # else:
        #     h = self.in_layers(x)
        h = self.in_layers(x)
        h = self.BN(h)
        h = self.middle_layers(h)
        h = self.gelu(h)
        h = self.out_layers(h)
        h = self.dropout(h)

        return h + self.skip(x)


class HighMixer(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1, padding=1,
                 **kwargs, ):
        super().__init__()
        self.cnn_in = cnn_in = dim // 2
        # print(cnn_in)
        self.pool_in = pool_in = dim - cnn_in

        self.cnn_dim = cnn_dim = cnn_in * 2
        self.pool_dim = pool_dim = pool_in * 2

        self.conv1 = nn.Conv2d(cnn_in, cnn_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.proj1 = nn.Conv2d(cnn_dim, cnn_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=False,
                               groups=cnn_dim)
        self.mid_gelu1 = nn.GELU()

        self.Maxpool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)
        self.proj2 = nn.Conv2d(pool_in, pool_dim, kernel_size=1, stride=1, padding=0)
        self.mid_gelu2 = nn.GELU()


    def forward(self, x):
        # B, C H, W
        # print(x.shape)
        cx = x[:, :self.cnn_in, :, :].contiguous()
        cx = self.conv1(cx)
        cx = self.proj1(cx)
        cx = self.mid_gelu1(cx)

        px = x[:, self.cnn_in:, :, :].contiguous()
        px = self.Maxpool(px)
        px = self.proj2(px)
        px = self.mid_gelu2(px)

        hx = torch.cat((cx, px), dim=1)
        return hx


class LowMixer(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., pool_size=2,
                 **kwargs, ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.dim = dim

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.pool = nn.AvgPool2d(pool_size, stride=pool_size, padding=0,
                                 count_include_pad=False) if pool_size > 1 else nn.Identity()
        self.uppool = nn.Upsample(scale_factor=pool_size) if pool_size > 1 else nn.Identity()

    def att_fun(self, q, k, v, B, N, C):
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = (attn @ v).transpose(2, 3).reshape(B, C, N)
        return x

    def forward(self, x):
        # B, C, H, W
        B, _, _, _ = x.shape
        xa = self.pool(x)
        xa = xa.permute(0, 2, 3, 1).view(B, -1, self.dim)
        B, N, C = xa.shape
        qkv = self.qkv(xa).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        xa = self.att_fun(q, k, v, B, N, C)
        xa = xa.view(B, C, int(N ** 0.5), int(N ** 0.5))  # .permute(0, 3, 1, 2)

        xa = self.uppool(xa)
        return xa


class Mixer(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., attention_head=1, pool_size=2,
                 **kwargs, ):
        super().__init__()
        self.dim = dim
        if dim == 3:
            self.num_heads = num_heads
            self.head_dim = head_dim = dim // num_heads

            self.low_dim = low_dim = 1
            self.high_dim = high_dim = 2
        else:
            self.num_heads = num_heads
            self.head_dim = head_dim = dim // num_heads

            self.low_dim = low_dim = attention_head * head_dim
            self.high_dim = high_dim = dim - low_dim

        self.high_mixer = HighMixer(high_dim)
        self.low_mixer = LowMixer(low_dim, num_heads=attention_head, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                  pool_size=pool_size, )

        self.conv_fuse = nn.Conv2d(low_dim + high_dim * 2, low_dim + high_dim * 2, kernel_size=3, stride=1, padding=1,
                                   bias=False, groups=low_dim + high_dim * 2)
        self.proj = nn.Conv2d(low_dim + high_dim * 2, dim, kernel_size=1, stride=1, padding=0)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # B, H, W, C = x.shape
        # x = x.permute(0, 3, 1, 2)
        hx = x[:, :self.high_dim, :, :].contiguous()
        hx = self.high_mixer(hx)

        lx = x[:, self.high_dim:, :, :].contiguous()
        lx = self.low_mixer(lx)

        x = torch.cat((hx, lx), dim=1)
        x = x + self.conv_fuse(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        # x = x.permute(0, 2, 3, 1).contiguous()
        return x


class iFormer_Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, attention_head=1, pool_size=2,
                 attn=Mixer,
                 use_layer_scale=False, layer_scale_init_value=1e-5,
                 ):
        super().__init__()
        self.dim = dim

        self.RB_3 = RB(3, 3, dropout_rate=0.2)
        self.norm_3 = norm_layer(3, eps=1e-6)
        self.mlp_3 = metaformer.Mlp(dim=3, drop=drop)

        self.norm1 = norm_layer(dim, eps=1e-6)

        self.attn = attn(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                         attention_head=attention_head, pool_size=pool_size, )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = metaformer.Mlp(dim=dim, drop=drop)


        self.use_layer_scale = use_layer_scale
        if self.use_layer_scale:
            # print('use layer scale init value {}'.format(layer_scale_init_value))
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.layer_scale_2 * self.mlp(self.norm2(x)))
        else:
            if self.dim == 3:
                x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
                x = self.norm_3(x)
                x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
                x = self.RB_3(x)
            else:
                x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
                x = self.norm1(x)
                x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
                x = x + self.drop_path(self.attn(x))
                x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
                x = self.norm2(x)
                x = self.mlp(x)
                x = x + self.drop_path(x)
                x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvNextBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim_in, dim_out, drop_path=0., layer_scale_init_value=0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim_in, dim_in, kernel_size=7, padding=3, groups=dim_in)  # depthwise conv
        self.norm = LayerNorm(dim_in, eps=1e-6)
        self.pwconv1 = nn.Linear(dim_in, 4 * dim_in)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim_in, dim_in)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim_in)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.channl_conv = nn.Conv2d(dim_in, dim_out, kernel_size=1)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)

        x = self.channl_conv(x)
        return x


class RepBlock(nn.Module):
    def __init__(self, in_channels, out_channels, block_type, channelAttention_reduce=4,):
        super().__init__()
        self.C = in_channels
        self.O = out_channels
        self.BN = nn.BatchNorm2d(in_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.gelu = nn.GELU()
        self.block_type = block_type

        # assert in_channels == out_channels
        self.dconv5_5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)
        # self.conv_3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        # self.conv_5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)
        # self.conv_7 = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels)
        self.dconv1_3 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3), padding=(0, 1), groups=in_channels)
        self.dconv3_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 1), padding=(1, 0), groups=in_channels)
        self.dconv1_5 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 5), padding=(0, 2), groups=in_channels)
        self.dconv5_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(5, 1), padding=(2, 0), groups=in_channels)
        self.dconv1_7 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 7), padding=(0, 3), groups=in_channels)
        self.dconv7_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(7, 1), padding=(3, 0), groups=in_channels)
        # self.dconv1_9 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 9), padding=(0, 4), groups=in_channels)
        # self.dconv9_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(9, 1), padding=(4, 0), groups=in_channels)
        self.dconv1_11 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 11), padding=(0, 5), groups=in_channels)
        self.dconv11_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(11, 1), padding=(5, 0), groups=in_channels)
        self.dconv1_21 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 21), padding=(0, 10), groups=in_channels)
        self.dconv21_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(21, 1), padding=(10, 0), groups=in_channels)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1), padding=0)
        self.conv_LE = nn.Conv2d(in_channels * 4, in_channels, kernel_size=(1, 1), padding=0)
        self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding=0)
        self.act = nn.GELU()

    def forward(self, inputs):

        x_init = self.dconv5_5(inputs)
        # x_0 = self.gelu(x_init)
        # x_0 = self.BN(x_0)
        x_0 = x_init
        # if self.block_type == 'x':
        #     x_1 = self.conv_3(x_0)
        #     x_1 = self.BN(x_1)
        #     x_2 = self.conv_5(x_0)
        #     x_2 = self.BN(x_2)
        #     x_3 = self.conv_7(x_0)
        #     x_3 = self.BN(x_3)
        # else:
        x_1 = self.dconv1_3(x_0)
        x_1 = self.dconv3_1(x_1)
        x_1 = self.BN(x_1)
        x_2 = self.dconv1_5(x_0)
        x_2 = self.dconv5_1(x_2)
        x_2 = self.BN(x_2)
        x_3 = self.dconv1_7(x_0)
        x_3 = self.dconv7_1(x_3)
        x_3 = self.BN(x_3)
        x = x_1 + x_2 + x_3 + x_init
        spatial_att = self.conv(x)
        # spatial_att = self.BN(spatial_att)
        out = spatial_att + inputs
        out = self.conv_out(out)
        return out


class XXBlock(nn.Module):
    def __init__(self, in_channels, out_channels, type):
        super().__init__()
        self.BN = nn.BatchNorm2d(in_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.GELU(),
            RepBlock(in_channels, out_channels, type),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
        )
        self.FFNBlock = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
        )

    def forward(self, x):
        h = self.BN(x)
        h = self.block(h)
        x = x + h
        h = self.BN(x)
        h = self.FFNBlock(h)
        x = x + h
        return x


class Head(nn.Module):
    def __init__(self, in_ch, da=0.0):
        super().__init__()
        self.dropout_rata = da
        self.conv1 = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.BN = nn.BatchNorm2d(in_ch, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.dropout = nn.Dropout(da)
        self.conv2 = nn.Conv2d(in_ch, 1, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.BN(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x


class TB_attention(nn.Module):
    def __init__(self, size, da):

        super().__init__()
        self.dropout_rata = da
        self.caformer = metaformer.MetaFormer(
            depths=[3, 12, 18, 3],
            dims=[64, 128, 320, 512],
            # token_mixers=[metaformer.Mixer, metaformer.Mixer, metaformer.Mixer, metaformer.Mixer],
            token_mixers=[metaformer.SepConv, metaformer.SepConv, metaformer.Attention, metaformer.Attention],
            head_fn=metaformer.MlpHead,
        )
        checkpoint_a = torch.load(r"D:\coder\github\FCBFormer-vXX\pretrained_model\caformer_s36_384.pth")
        self.caformer.load_state_dict(checkpoint_a)

        self.norm_layer = LayerNorm
        self.act_layer = nn.GELU
        self.down_176 = nn.Sequential(
            RB(3, 32),
            RB(32, 32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        )
        self.LE = nn.ModuleList([])
        for i in range(6):

            # iformer
            self.LE.append(
                nn.Sequential(
                    # iFormer_Block(dim=[3, 64, 128, 320, 512][i], num_heads=[1, 3, 6, 12, 16][i], mlp_ratio=4, qkv_bias=True, drop=self.dropout_rata,
                    #       attn_drop=0., drop_path=0., norm_layer=self.norm_layer, act_layer=self.act_layer, attention_head=[1, 1, 3, 8, 11][i],
                    #       pool_size=1, use_layer_scale=False, layer_scale_init_value=1e-5),
                    # ConvNextBlock([3, 64, 128, 320, 512][i], [3, 64, 128, 320, 512][i],
                    #               drop_path=self.dropout_rata),
                    RepBlock(in_channels=[3, 32, 64, 128, 320, 512][i], out_channels=[3, 32, 64, 128, 320, 512][i], block_type="LE"),
                    # RepBlock(in_channels=[3, 64, 128, 320, 512][i], out_channels=[3, 64, 128, 320, 512][i], type="LE"),
                    # XXBlock(in_channels=[3, 64, 128, 320, 512][i], out_channels=[3, 64, 128, 320, 512][i], type="LE"),
                    # nn.Identity(),
                )
            )

        self.SFA = nn.ModuleList([])
        for i in range(6):
            self.SFA.append(
                nn.Sequential(
                    # ConvNextBlock([3 + 64, 64 + 64, 128 + 128, 320 + 320, 512][i], [3, 64, 64, 128, 320][i], drop_path=self.dropout_rata),
                    # nn.Dropout(self.dropout_rata),
                    RepBlock(in_channels=[3 + 32, 32 + 64, 64 + 64, 128 + 128, 320 + 320, 512][i],
                           out_channels=[3, 32, 64, 64, 128, 320][i], block_type='SFA'),
                    nn.Dropout(self.dropout_rata),
                    # frescat([64, 64, 128, 320][i])
                    # RB([64, 64, 128, 320][i], [64, 64, 128, 320][i]),
                    # nn.Upsample(size=[int(size), int(size), int(size/4), int(size/8), int(size/16)][i]),
                    nn.Upsample(scale_factor=[1, 2, 2, 2, 2, 2][i]),
                    # nn.ConvTranspose2d([64, 64, 128, 320][i], [64, 64, 128, 320][i], kernel_size=2, stride=2)
                )
            )
        self.PHs = nn.ModuleList([])
        self.upsizes = nn.ModuleList([])
        for i in range(6):
            self.PHs.append(nn.Sequential(
                Head([3, 32, 64, 128, 320, 512][i], da),
            ))
        for i in range(5):
            self.upsizes.append(nn.Sequential(
                # nn.Upsample(size=[int(size/16), int(size/8), int(size/4), int(size)][i]),
                nn.Upsample(scale_factor=[2, 2, 2, 2, 2][i]),
            ))

    def get_pyramid(self, x):
        # pyramid_a = []
        x_176 = self.down_176(x)

        pyramid_a = self.caformer(x, x_176)
        i = 0
        for p in pyramid_a:
            if i > 1:
                pyramid_a[i] = pyramid_a[i].permute(0, 3, 1, 2).contiguous()
            i = i + 1
        return pyramid_a

    def feature_forward(self, x):
        pyramid_a = self.get_pyramid(x)
        pyramid_emph_a = []
        LE_features_ph = [] # LE的预测图
        SFA_features_ph = [] # SFA的预测图
        for i, level in enumerate(pyramid_a):
            # pyramid_emph.append(self.LE[i](torch.concat((pyramid_a[i], pyramid_c[i]), dim=1)))
            # pyramid_emph.append(torch.add(pyramid_a[i], pyramid_c[i],))
            feature_i = self.LE[i](pyramid_a[i])
            pyramid_emph_a.append(feature_i)
            LE_features_ph.append(self.PHs[i](feature_i))

        # add_feature_phs = []
        # 跳链接部分的特征金字塔（add方式）
        LE_feature_ph = LE_features_ph.pop()
        for i in range(5):
            LE_feature_ph = torch.add(self.upsizes[i](LE_feature_ph), LE_features_ph.pop())
            # LE_feature_ph = torch.concat((self.upsizes[i](LE_feature_ph), LE_features_ph.pop()), dim=1)
            # add_feature_phs.append(feature_ph)

        # decoder
        l_i_a = self.SFA[-1](pyramid_emph_a[-1])
        SFA_features_ph.append(self.PHs[-2](l_i_a))
        for i in range(4, -1, -1):
            # l_a = torch.add(pyramid_emph_a[i], l_i_a)
            l_a = torch.concat((pyramid_emph_a[i], l_i_a), dim=1)
            l_a = self.SFA[i](l_a)
            if i > 2:
                feature_i = self.PHs[i-1](l_a)
            else:
                feature_i = self.PHs[i](l_a)
            SFA_features_ph.append(feature_i)
            l_i_a = l_a
        feature = l_a

        # 解码器路线的特征金字塔（add方式）
        SFA_feature_ph = SFA_features_ph[0]
        for i in range(4):
            # add方式
            SFA_feature_ph = torch.add(self.upsizes[i+1](SFA_feature_ph), SFA_features_ph[i+1])
            # concat方式
            # SFA_feature_ph = torch.concat((self.upsizes[i + 1](SFA_feature_ph), SFA_features_ph[i + 1]), dim=1)
            # feature_ph = torch.add(self.upsizes[i](feature_ph), SFA_features_ph.pop())
        SFA_feature_ph = torch.add(SFA_feature_ph, SFA_features_ph[-1])
        feature_ph = torch.concat((SFA_feature_ph, LE_feature_ph), dim=1)

        return feature, feature_ph

    def forward(self, x, augsub_type, augsub_ratio):
        if augsub_type == "masking":
            x = self.patchify(x)
            x = self.random_masking(x, augsub_ratio)
            x = self.unpatchify(x)
        feature, feature_ph = self.feature_forward(x)

        return feature, feature_ph


class XXFormer(nn.Module):
    def __init__(self, size, drop_rate=0.0):
        super().__init__()

        # self.size = size
        self.drop_rate = drop_rate
        self.TB_a = TB_attention(size=size, da=self.drop_rate)
        # self.TB_c = TB_conv(da=self.drop_rate)
        self.PH_a = nn.Sequential(
            RB(64, 64, dropout_rate=drop_rate),
            RB(64, 32, dropout_rate=drop_rate),
            nn.Conv2d(32, 1, kernel_size=1)
        )
        self.PH_c = nn.Sequential(
            RB(64, 64, dropout_rate=drop_rate),
            RB(64, 32, dropout_rate=drop_rate),
            nn.Conv2d(32, 1, kernel_size=1)
        )
        self.PH = nn.Sequential(
            RB(5, 5, dropout_rate=drop_rate),
            nn.Conv2d(5, 1, kernel_size=1)
        )
        # self.up_tosize = nn.Upsample(size=size)
        # self.up_tosize = frescat(channels=64)

    def forward(self, x, augsub_type='none', augsub_ratio=0.5):
        feature_a, feature_ph_a = self.TB_a(x, augsub_type, augsub_ratio)
        out = self.PH(torch.concat((feature_a, feature_ph_a), dim=1))

        return out
