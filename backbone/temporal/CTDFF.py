import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange


# output: 3->24
class CTDFF(nn.Module):

    def __init__(self, frame_seg, patch_size=16, image_size=256, embed_dim=768):
        super(CTDFF, self).__init__()
        output_channel = 24
        self.frame_seg = frame_seg

        self.avg_diff = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            # dw
            nn.Conv2d(9, 9, 3, 1, 1, groups=9, bias=False),
            nn.BatchNorm2d(9),
            nn.ReLU(inplace=True),
            # pw
            nn.Conv2d(9, output_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channel)
        )

        self.patch_embed = PatchEmbed(output_channel, patch_size, image_size // 2, embed_dim)
        self.aisa = AISA(embed_dim)
        self.inverse_patch = PatchEmbed(output_channel, patch_size, image_size // 2, embed_dim, inverse=True)

        self.maxpool_diff = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.conv2 = nn.Sequential(
            # dw
            nn.Conv2d(3, 3, 3, 2, 1, groups=3, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            # pw
            nn.Conv2d(3, output_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channel)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, Attention):
                nn.init.normal_(module.w, mean=0, std=0.1)
        print("Finish initialize CTDFF.")

    def forward(self, x):
        t_list = []
        for i in range(self.frame_seg):
            t0, t1, t2 = x[i + 0].clone(), x[i + 1].clone(), x[i + 2].clone()

            t_diff = self.avg_diff(torch.cat([t1 - t0, t2 - t1, t2 - t0], 1))  # B,9,128,128
            t_diff = self.conv1(t_diff)  # B,24,128,128

            # AISA
            t_diff_patch = self.patch_embed(t_diff)  # B,num_patch,768
            t_diff_patch = self.aisa(t_diff_patch)  # B,num_patch,768
            t_diff = self.inverse_patch(t_diff_patch)  # B,24,128,128

            t_diff = self.maxpool_diff(1.0 / 1.0 * t_diff)  # B,24,32,32

            # 融合
            t_key = self.conv2(t1)  # B,24,128,128
            t_key = self.maxpool(t_key)  # B,24,64,64
            temp_diff = F.interpolate(t_diff, t_key.size()[2:], mode='bilinear')
            t_key = 0.5 * t_key + 0.5 * temp_diff

            t_list.append(t_key)

        res_list = [t_list[i].unsqueeze(2) for i in range(self.frame_seg)]
        res_cat = torch.cat(res_list, dim=2)  # B C T H W
        bs, c, t, h, w = res_cat.shape
        res_cat = res_cat.view(bs * self.frame_seg, c, t // self.frame_seg, h, w).squeeze(2)  # (B*T) C H W
        return res_cat


class PatchEmbed(nn.Module):
    def __init__(self, input_channel, patch_size=16, image_size=256, embed_dim=768, inverse=False):
        super().__init__()
        self.inverse = inverse
        self.grid_size = image_size // patch_size
        num_patches = self.grid_size ** 2

        self.proj = nn.Identity()
        if not inverse:
            self.proj = nn.Conv2d(input_channel, embed_dim, kernel_size=patch_size, stride=patch_size)
        else:
            self.proj = nn.ConvTranspose2d(embed_dim, input_channel, kernel_size=patch_size, stride=patch_size)

        self.norm = nn.LayerNorm(embed_dim) if not inverse else nn.BatchNorm2d(input_channel)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, embed_dim))

    def forward(self, x):
        if self.inverse:
            x = x.transpose(1, 2).view(x.size(0), -1, self.grid_size, self.grid_size)
            x = self.proj(x)
            x = self.norm(x)
            return x

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x + self.pos_embedding
        return x


class FeedForward(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, input_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, embed_size, num_heads=8):
        super(Attention, self).__init__()

        assert embed_size % num_heads == 0, "Embedding size must be divisible by number of heads"

        self.embed_size = embed_size
        self.num_heads = num_heads
        self.scale = (embed_size // num_heads) ** -0.5

        self.to_qkv = nn.Linear(embed_size, embed_size * 3, bias=False)

        self.w = nn.Parameter(torch.ones(2))

        self.fc_out = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.Dropout(0.1)
        )
        self.norm = nn.LayerNorm(embed_size)

    def forward(self, x):
        x = self.norm(x)
        N, seq_length, embed_size = x.shape

        QKV = self.to_qkv(x).chunk(3, dim=-1)
        Q, K, V = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), QKV)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        important_attention = F.softmax(scores, dim=-1)
        redundant_attention = F.leaky_relu(scores)

        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))

        out = torch.matmul(w1 * important_attention + w2 * redundant_attention, V)

        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(N, seq_length, self.embed_size)

        out = self.fc_out(out)
        return out


class AISA(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim)
        self.ff = FeedForward(embed_dim)

    def forward(self, x):
        x = self.attn(x) + x
        x = self.ff(x) + x
        return self.norm(x)


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
