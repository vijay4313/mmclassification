import logging
import math
import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer, constant_init, kaiming_init
from mmcv.runner import load_checkpoint
from mmcv.utils.parrots_wrapper import _BatchNorm
from torch.nn import Conv2d, ConvTranspose2d
from torch.nn import functional as F

from ..builder import NECKS


@NECKS.register_module()
class FeaturePyramidNetwork(nn.Module):
    """
    This module implements :paper:`FPN`.
    It creates pyramid features built on top of some input feature maps.
    """

    def __init__(self, input_shapes, in_features, out_channels,
                 norm="", top_block=None, fuse_type="sum"):
        """
        Args:
            in_features (list[str]): names of the input feature maps coming
                from the backbone to which FPN is attached. For example, if the
                backbone produces ["res2", "res3", "res4"], any *contiguous* sublist
                of these may be used; order must be from high to low resolution.
            out_channels (int): number of channels in the output feature maps.
            norm (str, dict): dictionary to construct and config norm layer.
            Example: dict(type='BN')
            top_block (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list. The top_block
                further downsamples the feature map. It must have an attribute
                "num_levels", meaning the number of extra FPN levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5).
            fuse_type (str): types for fusing the top down features and the lateral
                ones. It can be "sum" (default), which sums up element-wise; or "avg",
                which takes the element-wise mean of the two.
        """
        super(FPN, self).__init__()

        # Feature map strides and channels from the bottom up network (e.g. ResNet)
        in_strides = [input_shapes[f].stride for f in in_features]
        in_channels = [input_shapes[f].channels for f in in_features]

        _assert_strides_are_log2_contiguous(in_strides)
        lateral_convs, lateral_norms = [], []
        output_convs, output_norms = [], []
        self.use_norm = norm != ""

        use_bias = norm == ""
        for idx, in_channels in enumerate(in_channels):
            lateral_conv = Conv2d(in_channels, out_channels, kernel_size=1, bias=use_bias)

            output_conv = Conv2d(out_channels,
                                 out_channels,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1,
                                 bias=use_bias)

            stage = int(math.log2(in_strides[idx]))

            if self.use_norm:
                _, lateral_norm = build_norm_layer(norm, out_channels, stage)
                _, output_norm = build_norm_layer(norm, out_channels, stage)
                self.add_module("fpn_lateral_norm{}".format(stage), lateral_norm)
                self.add_module("fpn_output_norm{}".format(stage), output_norm)
                lateral_norms.append(lateral_norm)
                output_norms.append(output_norm)

            self.add_module("fpn_lateral{}".format(stage), lateral_conv)
            self.add_module("fpn_output{}".format(stage), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]
        if self.use_norm:
            self.lateral_norms = lateral_norms[::-1]
            self.output_norms = output_norms[::-1]

        self.top_block = top_block
        self.in_features = in_features
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {"p{}".format(
            int(math.log2(s))): s for s in in_strides}
        # top block output feature maps.
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {
            k: out_channels for k in self._out_features}
        self._size_divisibility = in_strides[-1]
        assert fuse_type in {"avg", "sum"}
        self._fuse_type = fuse_type

    def init_weights(self, pretrained=None):
        """Init weights

        Args:
            pretrained (str | None): If pretrained is a string, then it
                initializes weights by loading the pretrained
                checkpoint. If pretrained is None, then it follows default
                initializer or customized initializer in subclasses.
        """
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None.'
                            f' But received {type(pretrained)}.')
    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, input):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        # Reverse feature maps into top-down order (from low to high resolution)
        x = [input[f] for f in self.in_features[::-1]]
        results = []
        prev_features = self.lateral_convs[0](x[0])
        output = self.output_convs[0](prev_features)
        if self.use_norm:
            prev_features = self.lateral_norms[0](prev_features)
            output = self.output_norms[0](output)
        results.append(output)
        for idx, features in enumerate(x[1:]):
            top_down_features = F.interpolate(prev_features, scale_factor=2,
                                              mode="bilinear",
                                              align_corners=True)
            lateral_features = self.lateral_convs[idx+1](features)
            if self.use_norm:
                lateral_features = self.lateral_norms[idx+1](lateral_features)
            # lateral_features = F.interpolate(lateral_features,
            #                                  size=top_down_features.size(
            #                                  )[-2:],
            #                                  mode="nearest")
            # print(lateral_features.shape,
            #       top_down_features.shape, prev_features.shape)
            prev_features = lateral_features + top_down_features
            if self._fuse_type == "avg":
                prev_features /= 2
            output = self.output_convs[idx+1](prev_features)
            if self.use_norm:
                output = self.output_norms[idx+1](output)
            results.insert(0, output)

        if self.top_block is not None:
            top_block_in_feature = input.get(
                self.top_block.in_feature, None)
            if top_block_in_feature is None:
                top_block_in_feature = results[self._out_features.index(
                    self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        return dict(zip(self._out_features, results))


def _assert_strides_are_log2_contiguous(strides):
    """
    Assert that each stride is 2x times its preceding stride, i.e. "contiguous in log2".
    """
    for i, stride in enumerate(strides[1:], 1):
        assert stride == 2 * strides[i - 1], "Strides {} {} are not log2 contiguous".format(
            stride, strides[i - 1]
        )