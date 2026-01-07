from typing import Optional, Union, List

from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import (
    SegmentationModel,
    SegmentationHead,
    ClassificationHead,
)
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from segmentation_models_pytorch.base import modules as md
from einops import rearrange

class DropPath(nn.Module):
    def __init__(self, p: float = None):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.p == 0. or not self.training:
            return x
        kp = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = kp + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        return x.div(kp) * random_tensor

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim=None) -> None:
        super().__init__()
        out_dim = out_dim or dim
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim, heads = 8):
        super().__init__()
        
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        self.to_out = nn.Linear(dim, dim, bias = False)
    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)
        

class Transformer(nn.Module):
    def __init__(self, dim, num_heads,with_pos=False,dpr=0.) -> None:
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim*4)
        
        self.is_pos=with_pos
        # self.end = nn.Conv2d(dim,dim,1,1,0)
    def forward(self, x: Tensor) -> Tensor:
        if self.is_pos:
            x = x + self.pos_embed(x)
        B, N, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).reshape(B, N, H, W)
        return x
        
class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
        attention_type=None,
    ):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x

class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=16, in_ch=3, embed_dim=768) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Conv2d(in_ch, embed_dim, patch_size, patch_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x

class ViTUnet(nn.Module):
    def __init__(
        self,
        encoder_channels: List[int],
        decoder_channels: List[int],
        n_blocks: int = 5,
        use_batchnorm: bool = True,
        attention_type: Optional[str] = None,
        center: bool = False,
    ):
        super().__init__()
        
        encoder_channels = encoder_channels[1:]  
        encoder_channels = encoder_channels[::-1]  
 
        head_channels = encoder_channels[0]
        skip_channels = list(encoder_channels[1:]) + [0]  # 跳跃连接通道
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        self.center = CenterBlock(head_channels, head_channels, use_batchnorm) if center else nn.Identity()

        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.blocks = nn.ModuleList([
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ])
        
        self.transformer_configs = [
            (512, 4), 
            (256, 2),  
            (128, 2),  
        ]

        self.transformers = nn.ModuleList([
            nn.ModuleList([Transformer(ch, heads, with_pos=True) for _ in range(1)])
            for ch, heads in self.transformer_configs
        ])
    
    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        features = features[1:] 
        features = features[::-1]  
        
        head = features[0]  
        skips = features[1:]  
        
        x = self.center(head)
        for transformer in self.transformers[0]:
            x = transformer(x)

        if len(skips) > 0 and len(self.transformers) > 1:
            skip1 = skips[0]
            for transformer in self.transformers[1]:
                skip1 = transformer(skip1)
            x = self.blocks[0](x, skip1)
        else:
            x = self.blocks[0](x, None)
        
        if len(skips) > 1 and len(self.transformers) > 2:
            skip2 = skips[1]
            for transformer in self.transformers[2]:
                skip2 = transformer(skip2)
            x = self.blocks[1](x, skip2)

        for i, decoder_block in enumerate(self.blocks[2:], 2):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
        
        return x

class SMViT(SegmentationModel):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )
        
        self.decoder = ViTUnet(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()

if __name__ == '__main__':
    model = SMViT(
        encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights='imagenet',     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,                      # model output channels (number of classes in your dataset)
        activation='sigmoid'
    ) 
    x = torch.randn(4, 3, 224, 224)
    print(model(x).shape)



