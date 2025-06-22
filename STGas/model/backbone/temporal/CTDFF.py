import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange


# output: 3->24
class CTDFF(nn.Module):

    def __init__(self, frame_seg, depth=1):
        super(CTDFF, self).__init__()
        output_channel = 24
        self.frame_seg = frame_seg

        self.avg_diff = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(9, 72, 3, 2, 1, bias=False),
            nn.BatchNorm2d(72),
            nn.Hardswish(inplace=True)
        )
        self.maxpool_diff = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.conv2 = nn.Sequential(
            nn.Conv2d(3, output_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.Hardswish(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(72, AISA(embed_size=72)),
                PreNorm(72, FeedForward(72, 144))
            ]))
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(72, output_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channel)
        )

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, AISA):
                for m in module.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)

                # 初始化AISA中的可学习参数W1和W2
                nn.init.kaiming_normal_(module.W1, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.kaiming_normal_(module.W2, mode='fan_in', nonlinearity='leaky_relu')

        print("Finish initialize CTDFF.")

    def forward(self, x):
        t_list = []
        for i in range(self.frame_seg):
            t0, t1, t2 = x[i + 0].clone(), x[i + 1].clone(), x[i + 2].clone()

            t_diff = self.avg_diff(torch.cat([t1 - t0, t2 - t1, t2 - t0], 1))  # B,9,128,128
            t_diff = self.conv1(t_diff)  # B,72,64,64
            t_diff = self.maxpool_diff(1.0 / 1.0 * t_diff)  # B,72,32,32

            # AISA
            t_diff_flat = t_diff.flatten(2).transpose(1, 2)  # B,32*32,24
            for aisa, ff in self.layers:
                t_diff_flat = aisa(t_diff_flat) + t_diff_flat
                t_diff_flat = ff(t_diff_flat) + t_diff_flat

            t_diff = t_diff_flat.view(t_diff.size()[0], -1, t_diff.size()[2], t_diff.size()[3])  # B,24,32,32
            t_diff = self.conv1_1(t_diff)  # B,24,32,32

            # 融合1
            t_key = self.conv2(t1)  # B,24,128,128
            t_key = self.maxpool(t_key)  # B,24,64,64
            temp_diff = F.interpolate(t_diff, t_key.size()[2:])
            t_key = 0.5 * t_key + 0.5 * temp_diff

            t_list.append(t_key)

        res_list = [t_list[i].unsqueeze(2) for i in range(self.frame_seg)]
        res_cat = torch.cat(res_list, dim=2)  # B C T H W
        bs, c, t, h, w = res_cat.shape
        res_cat = res_cat.view(bs * self.frame_seg, c, t // self.frame_seg, h, w).squeeze(2)  # (B*T) C H W
        return res_cat


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
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
            nn.Dropout(0.2)
        )

    def forward(self, x):
        N, seq_length, embed_size = x.shape

        QKV = self.to_qkv(x).chunk(3, dim=-1)
        Q, K, V = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), QKV)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        important_attention = F.softmax(scores, dim=-1)
        redundant_attention = F.leaky_relu(scores)

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
