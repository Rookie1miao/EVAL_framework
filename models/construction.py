from typing import Any, Optional, Union, Tuple, Callable

from segmentation_models_pytorch.base import (
    ClassificationHead,
    SegmentationModel,
)
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base.hub_mixin import supports_config_loading

import torch
import torch.nn as nn
from . import modules as md

class Unet(SegmentationModel):

    @supports_config_loading
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = None,
        decoder_use_batchnorm: bool = True,
        decoder_channels: Tuple[int, ...] = (512, 256, 128, 64, 32),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, Callable]] = None,
        aux_params: Optional[dict] = None,
        **kwargs: dict[str, Any],
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            **kwargs,
        )

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False
        )

        self.segmentation_head = nn.Sequential(
            nn.Conv2d(
                in_channels=decoder_channels[-1],
                out_channels=decoder_channels[-1],
                kernel_size=5,
                padding=2,  
            ),
            nn.ReLU(inplace=True), 
            nn.Conv2d(
                in_channels=decoder_channels[-1],
                out_channels=classes,
                kernel_size=3,
                padding=1,
            )
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()

    def forward(self, dmsp, inputs, instances=None):
        if instances is None:
            x = torch.cat([dmsp, inputs], dim=1)
        else:
            x = torch.cat([dmsp, inputs, instances], dim=1)

        if not torch.jit.is_tracing() or self.requires_divisible_input_shape:
            self.check_input_shape(x)

        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks

class UnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        n_blocks=5,
        use_batchnorm=True,
        center=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        encoder_channels = encoder_channels[1:]
        encoder_channels = encoder_channels[::-1]

        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
        else:
            self.center = nn.Identity()

        blocks = [
            HierarchicalFusionDecoderBlock(
                in_channels=in_ch,
                skip_channels=skip_ch,
                out_channels=out_ch,
                use_batchnorm=use_batchnorm,
            )
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):

        features = features[1:]
        features = features[::-1]

        head = features[0]
        skips = features[1:]

        x = self.center(head)

        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x

class HierarchicalFusionDecoderBlock(nn.Module):

    def __init__(self, in_channels, skip_channels, out_channels, use_batchnorm=True):
        super().__init__()
        self.upsample_conv = nn.Conv2d(in_channels, out_channels * 4, kernel_size=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        
        self.skip_content_conv = nn.Conv2d(skip_channels, out_channels, kernel_size=1)
        self.fuse_conv = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1)
        
        self.skip_structure_conv = nn.Conv2d(skip_channels, out_channels, kernel_size=5, padding=2)
        self.gate_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1) 

        self.path_d1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=not use_batchnorm),
            nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        self.path_d4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=4, dilation=4, bias=not use_batchnorm),
            nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        self.path_d9 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=9, dilation=9, bias=not use_batchnorm),
            nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
        )

        self.path_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels * 3, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, 3, kernel_size=1),
            nn.Softmax(dim=1)
        )

        self.output_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        
    def forward(self, x, skip=None):
        x_up = self.pixel_shuffle(self.upsample_conv(x))

        if skip is None:
            fused_detailed = x_up
        else:
            skip_content = self.skip_content_conv(skip)
            fused_base = self.fuse_conv(torch.cat([x_up, skip_content], dim=1))
            
            skip_structure = self.skip_structure_conv(skip)
            gate = torch.sigmoid(self.gate_conv(skip_structure))
            
            fused_detailed = fused_base + (gate * skip_structure)

        identity = fused_detailed

        path1 = self.path_d1(fused_detailed)
        path2 = self.path_d4(fused_detailed)
        path3 = self.path_d9(fused_detailed)

        path_weights = self.path_attention(torch.cat([path1, path2, path3], dim=1))
        
        refined = (
            path1 * path_weights[:, 0].unsqueeze(1) +
            path2 * path_weights[:, 1].unsqueeze(1) +
            path3 * path_weights[:, 2].unsqueeze(1)
        )

        output = self.output_conv(refined) + identity
        
        return output

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
