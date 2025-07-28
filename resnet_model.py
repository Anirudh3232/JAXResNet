import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence

class ResidualBlock(nn.Module):
    channels: int
    stride: int = 1

    @nn.compact
    def __call__(self, x, training: bool = True):
        residual = x

        # First convolution
        x = nn.Conv(
            features=self.channels,
            kernel_size=(3, 3),
            strides=(self.stride, self.stride),
            padding='SAME',
            use_bias=False
        )(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)

        # Second convolution
        x = nn.Conv(
            features=self.channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='SAME',
            use_bias=False
        )(x)
        x = nn.BatchNorm(use_running_average=not training)(x)

        # Shortcut path if needed
        if self.stride != 1 or residual.shape[-1] != self.channels:
            residual = nn.Conv(
                features=self.channels,
                kernel_size=(1, 1),
                strides=(self.stride, self.stride),
                use_bias=False
            )(residual)
            residual = nn.BatchNorm(use_running_average=not training)(residual)

        x = x + residual
        x = nn.relu(x)
        return x

class ResNet(nn.Module):
    num_classes: int = 10
    blocks: Sequence[int] = (2, 2, 2, 2)
    initial_channels: int = 64

    @nn.compact
    def __call__(self, x, training: bool = True):
        x = nn.Conv(
            features=self.initial_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='SAME',
            use_bias=False
        )(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)

        channels = self.initial_channels
        for i, num_blocks in enumerate(self.blocks):
            stride = 1 if i == 0 else 2
            x = ResidualBlock(channels=channels, stride=stride)(x, training)
            for _ in range(1, num_blocks):
                x = ResidualBlock(channels=channels, stride=1)(x, training)
            channels *= 2

        x = jnp.mean(x, axis=(1, 2))  # Global average pooling
        x = nn.Dense(features=self.num_classes)(x)
        return x

def create_model():
    """Initialize the ResNet model"""
    return ResNet(num_classes=10, blocks=(2, 2, 2, 2))
