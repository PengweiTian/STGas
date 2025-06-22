import torch.nn as nn
import torch
import warnings

from STGas.model.backbone.temporal import build_temporal_backbone
from STGas.model.backbone.temporal.temporal_shift import make_temporal_shift


class GasConv(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(GasConv, self).__init__()

        self.conv_module = nn.Sequential(
            # pw
            nn.Conv2d(input_channels, input_channels * 3, 1, 1, 0, bias=False),
            nn.BatchNorm2d(input_channels * 3),
            nn.Hardswish(inplace=True),
            # dw
            nn.Conv2d(input_channels * 3, input_channels * 3, 3, 2, 1, groups=input_channels * 3, bias=False),
            nn.BatchNorm2d(input_channels * 3),
            CCA(input_channels * 3, 3),
            nn.Hardswish(inplace=True),
            # pw-linear
            nn.Conv2d(input_channels * 3, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
        )

    def forward(self, x):
        x = self.conv_module(x)
        return x


class CCA(nn.Module):
    def __init__(self, input_channel, split_num):
        super(CCA, self).__init__()
        self.split_num = split_num
        split_channel = input_channel // split_num

        self.Conv_list = nn.ModuleList(
            nn.Conv2d(split_channel, split_channel, 3, 1, 1, 1, groups=split_channel)
            for _ in range(self.split_num)
        )

    def forward(self, x):
        split_size = [x.shape[1] // self.split_num for _ in range(self.split_num)]
        split_xs = torch.split(x, split_size, dim=1)
        x_channels = [self.Conv_list[0](split_xs[0])]

        if self.split_num > 1:
            pre_x = x_channels[0]
            for item, Conv in zip(split_xs[1:], self.Conv_list[1:]):
                item_x = Conv(item + pre_x)
                x_channels.append(item_x)
                pre_x = item_x

        return torch.cat(x_channels, dim=1)


'''
https://arxiv.org/abs/2112.05561
'''


class GAM(nn.Module):
    def __init__(self, in_channels, out_channels, rate=4):
        super().__init__()
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        inchannel_rate = int(in_channels / rate)

        self.linear1 = nn.Linear(in_channels, inchannel_rate)
        self.activation = nn.LeakyReLU(inplace=True)
        self.linear2 = nn.Linear(inchannel_rate, in_channels)

        self.conv1 = nn.Conv2d(in_channels, inchannel_rate, kernel_size=7, padding=3, padding_mode='replicate')

        self.conv2 = nn.Conv2d(inchannel_rate, out_channels, kernel_size=7, padding=3, padding_mode='replicate')

        self.norm1 = nn.BatchNorm2d(inchannel_rate)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        # B,C,H,W ==> B,H*W,C
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)

        # B,H*W,C ==> B,H,W,C
        x_att_permute = self.linear2(self.activation(self.linear1(x_permute))).view(b, h, w, c)

        # B,H,W,C ==> B,C,H,W
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)

        x = x * x_channel_att

        x_spatial_att = self.activation(self.norm1(self.conv1(x)))
        x_spatial_att = self.sigmoid(self.norm2(self.conv2(x_spatial_att)))

        out = x * x_spatial_att

        return out


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups,
                            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)

    @torch.no_grad()
    def fuse(self):
        if isinstance(self.m, Conv2d_BN):
            m = self.m.fuse()
            assert (m.groups == m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        elif isinstance(self.m, torch.nn.Conv2d):
            m = self.m
            assert (m.groups != m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        else:
            return self


class RepVGGDW(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = torch.nn.Conv2d(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed
        self.bn = torch.nn.BatchNorm2d(ed)

    def forward(self, x):
        return self.bn((self.conv(x) + self.conv1(x)) + x)

    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [1, 1, 1, 1])

        identity = torch.nn.functional.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device),
                                           [1, 1, 1, 1])

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        bn = self.bn
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = conv.weight * w[:, None, None, None]
        b = bn.bias + (conv.bias - bn.running_mean) * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        conv.weight.data.copy_(w)
        conv.bias.data.copy_(b)
        return conv


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from timm.models.layers import SqueezeExcite


class RepViTGas_Block(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, stride=2, use_se=True, use_hs=True):
        super(RepViTGas_Block, self).__init__()

        if stride == 2:
            self.token_mixer = nn.Sequential(
                Conv2d_BN(input_channel, input_channel, kernel_size, stride, (kernel_size - 1) // 2,
                          groups=input_channel),
                SqueezeExcite(input_channel, 0.25) if use_se else nn.Identity(),
                Conv2d_BN(input_channel, output_channel, ks=1, stride=1, pad=0)
            )
            self.channel_mixer = Residual(nn.Sequential(
                # pw
                Conv2d_BN(output_channel, 2 * output_channel, 1, 1, 0),
                nn.GELU() if use_hs else nn.GELU(),
                # pw-linear
                Conv2d_BN(2 * output_channel, output_channel, 1, 1, 0, bn_weight_init=0),
            ))
        else:
            self.token_mixer = RepVGGDW(input_channel)
            self.SE = SqueezeExcite(input_channel, 0.25) if use_se else nn.Identity()
            self.GAM = GAM(input_channel, input_channel) if use_se else nn.Identity()

            self.channel_mixer = Residual(nn.Sequential(
                # pw
                Conv2d_BN(input_channel, input_channel // 2, 1, 1, 0),
                nn.GELU() if use_hs else nn.GELU(),
                # pw-linear
                Conv2d_BN(input_channel // 2, output_channel, 1, 1, 0, bn_weight_init=0),
            ))

    def forward(self, x):
        x = self.token_mixer(x)
        SE_x = self.SE(x.clone())
        GAM_x = self.GAM(x.clone())
        x = self.channel_mixer(SE_x + GAM_x)
        return x


class STGas(nn.Module):

    def __init__(self, out_stages, temporal, stage_repeats, stage_out_channels, frame_seg, repViTGas_use_se,
                 activation="Hardswish"):
        super(STGas, self).__init__()

        self.out_stages = out_stages
        self.stage_repeats = stage_repeats
        self.stage_out_channels = stage_out_channels
        use_se = [[bool(x) for x in sublist] for sublist in repViTGas_use_se]

        self.gasConv_0 = nn.Identity()
        self.gasConv_1 = GasConv(self.stage_out_channels[0], self.stage_out_channels[1])

        # 空间Backbone
        self.stage_modules = []
        input_channels = self.stage_out_channels[1]
        for idx, stage in enumerate(self.out_stages):
            output_channels = self.stage_out_channels[stage]
            temp_seq = [GasConv(input_channels, output_channels)]
            temp_seq += [RepViTGas_Block(output_channels, output_channels, stride=1, use_se=use_se[idx][i])
                         for i in range(self.stage_repeats[idx])]
            self.stage_modules.append(nn.Sequential(*temp_seq))
            input_channels = output_channels
        self.stage_modules = nn.Sequential(*list([stage for stage in self.stage_modules]))

        # 时序Backbone
        self.temporal = nn.Identity()
        if temporal in ["CTDFF", "TDN"]:
            self.temporal = build_temporal_backbone(temporal, frame_seg)
            self.temporal.conv2 = GasConv(3, self.stage_out_channels[0])
        elif temporal in ["TSM"]:
            make_temporal_shift(self, "STGas", frame_seg, n_div=8, place="block")
            self.gasConv_0 = GasConv(3, self.stage_out_channels[0])

        self.init_weights()

    def init_weights(self):
        for name, child in self.named_children():
            if "temporal" in name:
                continue

            for module in child.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)

                elif isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)

                elif isinstance(module, nn.BatchNorm2d):
                    nn.init.constant_(module.weight, 1)
                    nn.init.constant_(module.bias, 0)

        print("Finish initialize STGas.")

    def forward(self, x):
        x = self.temporal(x)
        x = self.gasConv_0(x)
        x = self.gasConv_1(x)

        output = []
        for stage in self.stage_modules:
            x = stage(x)
            output.append(x)
        return tuple(output)


if __name__ == '__main__':
    bs = 4
    segment = 8
    extra_count = 2
    total_frames = segment + extra_count
    input_tensor = [torch.randn(bs, 3, 256, 256) for _ in range(total_frames)]

    temp = STGas([2, 3, 4], [2, 4, 3], [24, 48, 72, 96, 120], 8)
    res = temp(input_tensor)
