import torch
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer, TransformerEncoder
import torch.nn as nn


NUM_CLIENTS = 20

def init_linear(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None: nn.init.zeros_(m.bias)


class Partial:
    def __init__(self, module, *args, **kwargs):
        self.module = module
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args_c, **kwargs_c):
        return self.module(*args_c, *self.args, **kwargs_c, **self.kwargs)


class LayerNormChannels(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        x = x.transpose(1, -1)
        x = self.norm(x)
        x = x.transpose(-1, 1)
        return x


class Residual(nn.Module):
    def __init__(self, *layers, shortcut=None):
        super().__init__()
        self.shortcut = nn.Identity() if shortcut is None else shortcut
        self.residual = nn.Sequential(*layers)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.shortcut(x) + self.gamma * self.residual(x)


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )


def get_shortcut(in_channels, out_channels, stride):
    if (in_channels == out_channels and stride == 1):
        shortcut = nn.Identity()
    else:
        shortcut = nn.Conv2d(in_channels, out_channels, 1)

    if stride > 1:
        shortcut = nn.Sequential(nn.MaxPool2d(stride), shortcut)

    return shortcut


class SqueezeExciteBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.out_channels = channels
        channels_r = channels // reduction
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels_r, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels_r, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)
class MBConv(Residual):
    def __init__(self, in_channels, out_channels, shape, kernel_size=3, stride=1, expansion_factor=4):
        mid_channels = in_channels * expansion_factor
        super().__init__(
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            ConvBlock(in_channels, mid_channels, 1), # Pointwise
            ConvBlock(mid_channels, mid_channels, kernel_size, stride=stride, groups=mid_channels), # Depthwise
            SqueezeExciteBlock(mid_channels),
            nn.Conv2d(mid_channels, out_channels, 1), # Pointwise
            shortcut = get_shortcut(in_channels, out_channels, stride)
        )


class SelfAttention2d(nn.Module):
    def __init__(self, in_channels, out_channels, head_channels, shape, p_drop=0.):
        super().__init__()
        self.heads = out_channels // head_channels
        self.head_channels = head_channels
        self.scale = head_channels ** -0.5

        self.to_keys = nn.Conv2d(in_channels, out_channels, 1)
        self.to_queries = nn.Conv2d(in_channels, out_channels, 1)
        self.to_values = nn.Conv2d(in_channels, out_channels, 1)
        self.unifyheads = nn.Conv2d(out_channels, out_channels, 1)

        height, width = shape
        self.pos_enc = nn.Parameter(torch.randn(self.heads, (2 * height - 1) * (2 * width - 1)))
        self.register_buffer("relative_indices", self.get_indices(height, width))

        self.drop = nn.Dropout(p_drop)

    def forward(self, x):
        b, _, h, w = x.shape

        keys = self.to_keys(x).view(b, self.heads, self.head_channels, -1)
        values = self.to_values(x).view(b, self.heads, self.head_channels, -1)
        queries = self.to_queries(x).view(b, self.heads, self.head_channels, -1)

        att = keys.transpose(-2, -1) @ queries

        indices = self.relative_indices.expand(self.heads, -1)
        rel_pos_enc = self.pos_enc.gather(-1, indices)
        rel_pos_enc = rel_pos_enc.unflatten(-1, (h * w, h * w))

        att = att * self.scale + rel_pos_enc
        att = F.softmax(att, dim=-2)

        out = values @ att
        out = out.view(b, -1, h, w)
        out = self.unifyheads(out)
        out = self.drop(out)
        return out

    @staticmethod
    def get_indices(h, w):
        y = torch.arange(h, dtype=torch.long)
        x = torch.arange(w, dtype=torch.long)

        y1, x1, y2, x2 = torch.meshgrid(y, x, y, x)
        indices = (y1 - y2 + h - 1) * (2 * w - 1) + x1 - x2 + w - 1
        indices = indices.flatten()

        return indices

class FeedForward(nn.Sequential):
    def __init__(self, in_channels, out_channels, mult=4, p_drop=0.):
        hidden_channels = in_channels * mult
        super().__init__(
            nn.Conv2d(in_channels, hidden_channels, 1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, 1),
            nn.Dropout(p_drop)
        )
class TransformerBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, head_channels, shape, stride=1, p_drop=0.):
        shape = (shape[0] // stride, shape[1] // stride)
        super().__init__(
            Residual(
                LayerNormChannels(in_channels),
                nn.MaxPool2d(stride) if stride > 1 else nn.Identity(),
                SelfAttention2d(in_channels, out_channels, head_channels, shape, p_drop=p_drop),
                shortcut = get_shortcut(in_channels, out_channels, stride)
            ),
            Residual(
                LayerNormChannels(out_channels),
                FeedForward(out_channels, out_channels, p_drop=p_drop)
            )
        )


class Stem(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__(
            ConvBlock(in_channels, out_channels, 3, stride=stride),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )




class BlockStack(nn.Sequential):
    def __init__(self, num_blocks, shape, in_channels, out_channels, stride, block):
        layers = []
        for _ in range(num_blocks):
            layers.append(block(in_channels, out_channels, shape=shape, stride=stride))
            shape = (shape[0] // stride, shape[1] // stride)
            in_channels = out_channels
            stride = 1
        super().__init__(*layers)

class Classifier(nn.Module):
    def __init__(self, channels, classes,p_drop=0.1):
        super(Classifier, self).__init__()
        self.norm = LayerNormChannels(channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p_drop)
        self.self_attn = TransformerHead(channels, 512)
        self.classifier = nn.Linear(512, classes)

    def forward(self, x):
        x = self.norm(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.self_attn(x)
        x = self.classifier(x)
        return x
class Head(nn.Sequential):
    def __init__(self, channels, classes, p_drop=0.):
        super().__init__(
            LayerNormChannels(channels),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p_drop),
            nn.Linear(channels, classes),

        )

class MultiHead(nn.Module):
    def __init__(self, channels, classes, p_drop=0.):
        super().__init__()
        self.multi_head = nn.ModuleList()
        for i in range(NUM_CLIENTS):
            self.multi_head.append(Head(channels, classes, p_drop=p_drop))

    def forward(self, x):
        x= torch.stack([self.multi_head[i](x) for i in range(10)], dim=1)
        return x.transpose(-1, -2)


class TransformerHead(nn.Module):
    def __init__(self, feature_size, num_layers, num_classes, p_drop):
        super(TransformerHead, self).__init__()
        encoder_layers = TransformerEncoderLayer(d_model=feature_size, nhead=4)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, num_classes)
        self.activation = nn.ReLU()
    def forward(self, src):

        src = src.transpose(1, 2)
        #print(src.shape)
        output = self.transformer_encoder(src)
        output = output.mean(dim=1)
        output = self.decoder(output)
        return output

#Client Model
class CoAtNet(nn.Sequential):
    def __init__(self, classes, image_size, head_channels, channel_list, num_blocks, strides=None,
                 in_channels=3, trans_p_drop=0., head_p_drop=0.):
        if strides is None: strides = [2] * len(num_blocks)

        block_list = [MBConv,  # S1
                      MBConv,  # S2
                      Partial(TransformerBlock, head_channels, p_drop=trans_p_drop),  # S3
                      Partial(TransformerBlock, head_channels, p_drop=trans_p_drop)]  # S4

        layers = [Stem(in_channels, channel_list[0], strides[0])]  # S0
        in_channels = channel_list[0]

        shape = (image_size, image_size)
        for num, out_channels, stride, block in zip(num_blocks, channel_list[1:], strides[1:], block_list):
            layers.append(BlockStack(num, shape, in_channels, out_channels, stride, block))
            shape = (shape[0] // stride, shape[1] // stride)
            in_channels = out_channels

        layers.append(Head(in_channels, classes, p_drop=head_p_drop))
        super().__init__(*layers)

#Global Model
class MHC_CoAtNet(nn.Sequential):
    def __init__(self, classes, image_size, head_channels, channel_list, num_blocks, strides=None,
                 in_channels=3, trans_p_drop=0., head_p_drop=0.):

        if strides is None: strides = [2] * len(num_blocks)

        block_list = [MBConv,  # S1
                      MBConv,  # S2
                      Partial(TransformerBlock, head_channels, p_drop=trans_p_drop),  # S3
                      Partial(TransformerBlock, head_channels, p_drop=trans_p_drop)]  # S4

        layers = [Stem(in_channels, channel_list[0], strides[0])]  # S0
        in_channels = channel_list[0]

        shape = (image_size, image_size)
        for num, out_channels, stride, block in zip(num_blocks, channel_list[1:], strides[1:], block_list):
            layers.append(BlockStack(num, shape, in_channels, out_channels, stride, block))
            shape = (shape[0] // stride, shape[1] // stride)
            in_channels = out_channels


        layers.append(MultiHead(in_channels, classes, p_drop=head_p_drop))
        layers.append(TransformerHead(feature_size=classes, num_layers=2,num_classes=classes, p_drop=head_p_drop))

        super(MHC_CoAtNet, self).__init__(*layers)