import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange


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

            # 融合
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
