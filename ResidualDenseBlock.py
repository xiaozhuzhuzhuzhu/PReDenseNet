import torch.nn
from DenseBlock import *
from TransitionBlock import *
from se import *


GROWTH_RATE_MULTIPLIER = 4


# 无空洞、无hdc、无se
class ResidualDenseBlock(torch.nn.Module):

    def __init__(self, in_channels, num_layers_m, growth_rate_k, reduction=16):
        super(ResidualDenseBlock, self).__init__()
        # self.down_sample_fn = torch.nn.AvgPool2d(kernel_size=2)
        # self.dense_block = DenseBlock(in_channels=in_channels, num_layers_m=num_layers_m,
        #                                      growth_rate_k=growth_rate_k)
        self.dense_block = DenseBlock(in_channels=in_channels, num_layers_m=num_layers_m,
                                      growth_rate_k=growth_rate_k)
        dense_channels_out = in_channels + num_layers_m * growth_rate_k
        self.transition_block = TransitionBlock(in_channels=dense_channels_out,
                                                out_channels=growth_rate_k * GROWTH_RATE_MULTIPLIER)

    def forward(self, x):
        # residual = self.down_sample_fn(x)
        residual = x
        x = self.dense_block(x)
        x = self.transition_block(x)
        x += residual
        return x


# 有空洞、无hdc、无se
class ResidualDenseBlock_dilated(torch.nn.Module):

    def __init__(self, in_channels, num_layers_m, growth_rate_k, reduction=16):
        super(ResidualDenseBlock_dilated, self).__init__()
        # self.down_sample_fn = torch.nn.AvgPool2d(kernel_size=2)
        # self.dense_block = DenseBlock(in_channels=in_channels, num_layers_m=num_layers_m,
        #                                      growth_rate_k=growth_rate_k)
        self.dense_block = DenseBlock_dilated(in_channels=in_channels, num_layers_m=num_layers_m,
                                              growth_rate_k=growth_rate_k)
        dense_channels_out = in_channels + num_layers_m * growth_rate_k
        self.transition_block = TransitionBlock(in_channels=dense_channels_out,
                                                out_channels=growth_rate_k * GROWTH_RATE_MULTIPLIER)

    def forward(self, x):
        # residual = self.down_sample_fn(x)
        residual = x
        x = self.dense_block(x)
        x = self.transition_block(x)
        x += residual
        return x


# 有空洞、有hdc、无se
class ResidualDenseBlock_dilated_hdc(torch.nn.Module):

    def __init__(self, in_channels, num_layers_m, growth_rate_k, reduction=16):
        super(ResidualDenseBlock_dilated_hdc, self).__init__()
        # self.down_sample_fn = torch.nn.AvgPool2d(kernel_size=2)
        # self.dense_block = DenseBlock(in_channels=in_channels, num_layers_m=num_layers_m,
        #                                      growth_rate_k=growth_rate_k)
        self.dense_block = DenseBlock_dilated_hdc(in_channels=in_channels, num_layers_m=num_layers_m,
                                                  growth_rate_k=growth_rate_k)
        dense_channels_out = in_channels + num_layers_m * growth_rate_k
        self.transition_block = TransitionBlock(in_channels=dense_channels_out,
                                                out_channels=growth_rate_k * GROWTH_RATE_MULTIPLIER)

    def forward(self, x):
        # residual = self.down_sample_fn(x)
        residual = x
        x = self.dense_block(x)
        x = self.transition_block(x)
        x += residual
        return x


class ResidualDenseBlock_dilated_hdc_se(torch.nn.Module):

    def __init__(self, in_channels, num_layers_m, growth_rate_k, reduction=16):
        super(ResidualDenseBlock_dilated_hdc_se, self).__init__()
        # self.down_sample_fn = torch.nn.AvgPool2d(kernel_size=2)
        # self.dense_block = DenseBlock(in_channels=in_channels, num_layers_m=num_layers_m,
        #                                      growth_rate_k=growth_rate_k)
        self.dense_block = DenseBlock_dilated_hdc(in_channels=in_channels, num_layers_m=num_layers_m,
                                                  growth_rate_k=growth_rate_k)
        dense_channels_out = in_channels + num_layers_m * growth_rate_k
        self.transition_block = TransitionBlock(in_channels=dense_channels_out,
                                                out_channels=growth_rate_k * GROWTH_RATE_MULTIPLIER)
        # initial se
        self.se = SELayer(channel=growth_rate_k * GROWTH_RATE_MULTIPLIER, reduction=reduction)

    def forward(self, x):
        # residual = self.down_sample_fn(x)
        residual = x
        x = self.dense_block(x)
        x = self.transition_block(x)
        x = self.se(x)
        x += residual
        return x


