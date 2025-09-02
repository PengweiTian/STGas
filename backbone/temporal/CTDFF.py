import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange


class GasConv(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(GasConv, self).__init__()

        self.conv_module = nn.Sequential(
            # pw
            nn.Conv2d(input_channels, input_channels * 3, 1, 1, 0, bias=False),
            nn.BatchNorm2d(input_channels * 3),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(input_channels * 3, input_channels * 3, 3, 1, 1, groups=input_channels * 3, bias=False),
            nn.BatchNorm2d(input_channels * 3),
            nn.Hardswish(inplace=True),
            CCA(input_channels * 3, 3),
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


# output: 3->24
class CTDFF(nn.Module):

    def __init__(self, frame_seg):
        super(CTDFF, self).__init__()
        output_channel = 24
        self.frame_seg = frame_seg

        self.avg_diff = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(9, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.maxpool_diff = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.conv2 = nn.Sequential(
            nn.Conv2d(3, output_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.aisa = AISA(embed_size=64)
        self.feed_forward = FeedForward(64, 128)
        self.layer_norm1 = nn.LayerNorm(64)
        self.layer_norm2 = nn.LayerNorm(64)

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, output_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )

        # self.gasConv1 = nn.Sequential(
        #     nn.Conv2d(24, 24, 3, 1, 1, bias=False),
        #     nn.BatchNorm2d(24),
        #     nn.ReLU(inplace=True)
        # )
        # self.gasConv2 = nn.Sequential(
        #     nn.Conv2d(24, 24, 3, 1, 1, bias=False),
        #     nn.BatchNorm2d(24),
        #     nn.ReLU(inplace=True)
        # )
        self.gasConv1 = GasConv(24, 48)
        self.gasConv2 = GasConv(24, 48)

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, AISA):
                nn.init.normal_(module.W1, mean=0, std=0.1)
                nn.init.normal_(module.W2, mean=0, std=0.1)
        print("Finish initialize CTDFF.")

    def forward(self, x):
        t_list = []
        for i in range(self.frame_seg):
            t0, t1, t2 = x[i + 0].clone(), x[i + 1].clone(), x[i + 2].clone()

            t_diff = self.avg_diff(torch.cat([t1 - t0, t2 - t1, t2 - t0], 1))  # B,9,128,128
            t_diff = self.conv1(t_diff)  # B,24,64,64
            t_diff = self.maxpool_diff(1.0 / 1.0 * t_diff)  # B,24,32,32

            # AISA
            t_diff_flat = t_diff.flatten(2).transpose(1, 2)  # B,32*32,24
            t_diff_flat = self.layer_norm1(t_diff_flat + self.aisa(t_diff_flat))
            t_diff_flat = self.layer_norm2(t_diff_flat + self.feed_forward(t_diff_flat))

            t_diff = t_diff_flat.view(t_diff.size()[0], -1, t_diff.size()[2], t_diff.size()[3])  # B,24,32,32
            t_diff = self.conv3(t_diff)

            # 融合1
            t_key = self.conv2(t1)  # B,24,128,128
            t_key = self.maxpool(t_key)  # B,24,64,64
            temp_diff = F.interpolate(t_diff, t_key.size()[2:])
            t_key = 0.5 * t_key + 0.5 * temp_diff

            # 融合2
            t_diff = self.gasConv1(t_diff)  # B,24,32,32
            t_key = self.gasConv2(t_key)  # B,24,64,64
            temp_diff = F.interpolate(t_diff, t_key.size()[2:])
            t_key = 0.5 * t_key + 0.5 * temp_diff

            t_list.append(t_key)

        res_list = [t_list[i].unsqueeze(2) for i in range(self.frame_seg)]
        res_cat = torch.cat(res_list, dim=2)  # B C T H W
        bs, c, t, h, w = res_cat.shape
        res_cat = res_cat.view(bs * self.frame_seg, c, t // self.frame_seg, h, w).squeeze(2)  # (B*T) C H W
        return res_cat


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.net(x)


class AISA(nn.Module):
    def __init__(self, embed_size, num_heads=8):
        super(AISA, self).__init__()

        assert embed_size % num_heads == 0, "Embedding size must be divisible by number of heads"

        self.embed_size = embed_size
        self.num_heads = num_heads
        self.scale = (embed_size // num_heads) ** -0.5

        self.to_qkv = nn.Linear(embed_size, embed_size * 3, bias=False)

        self.W1 = nn.Parameter(torch.randn(num_heads, 1, 1))
        self.W2 = nn.Parameter(torch.randn(num_heads, 1, 1))

        self.fc_out = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        N, seq_length, embed_size = x.shape

        QKV = self.to_qkv(x).chunk(3, dim=-1)
        Q, K, V = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), QKV)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        important_attention = F.softmax(scores, dim=-1)
        redundant_attention = F.leaky_relu(scores)
        # redundant_attention = torch.tanh(scores)
        redundant_attention = redundant_attention / (redundant_attention.abs().sum(-1, keepdim=True) + 1e-6)

        out = torch.matmul(self.W1 * important_attention + self.W2 * redundant_attention, V)

        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(N, seq_length, self.embed_size)

        out = self.fc_out(out)
        return out


if __name__ == '__main__':
    """
        I(t-1),I(t),I(t+1),seg=k
        检测帧：[f1,f2,f3,...,fk]
        I(t-1)=[f0,f1,f2,f3,...,fk-1] 
        I(t)=[f1,f2,f3,...,fk]
        I(t+1)=[f2,f3,...,fk,fk+1]
        当有t个时间维度序列时,需要在额外加t-1个序列，即每个时间维度片段，开头和结尾各加(t-1)/2
        input: (B*T) C H W
    """
    bs = 4
    segment = 8
    extra_count = 2
    total_frames = segment + extra_count
    input_tensor = [torch.randn(bs, 3, 256, 256) for _ in range(total_frames)]

    model = CTDFF(segment)
    res = model(input_tensor)
    print(res.shape)
