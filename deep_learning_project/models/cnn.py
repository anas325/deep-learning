import torch
import torch.nn as nn
from typing import List, Optional, Tuple


def _get_activation(name: str) -> nn.Module:
    """Return an activation module for the given name string."""
    activations = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
    }
    if name not in activations:
        raise ValueError(
            f"Unsupported activation '{name}'. Choose from {list(activations.keys())}"
        )
    return activations[name]


class ConvBlock(nn.Module):
    """
    A single convolutional block: Conv2d -> [BatchNorm2d] -> Activation -> [MaxPool2d].

    The block is stored as ``self.block`` (nn.Sequential) so it can be
    inspected or serialised easily.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output (filter) channels.
        kernel_size: Spatial size of the convolution kernel (square).
        padding: Zero-padding added to each spatial side before convolution.
        stride: Convolution stride.
        activation: Activation function name — "relu" or "tanh".
        batch_norm: If True, insert BatchNorm2d after the convolution.
        pool_size: If set, append MaxPool2d(pool_size, pool_size) at the end.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
        activation: str = "relu",
        batch_norm: bool = False,
        pool_size: Optional[int] = None,
    ) -> None:
        super().__init__()

        layers: List[nn.Module] = []

        # Core convolution
        layers.append(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
            )
        )

        # Optional batch normalisation (before activation, standard practice)
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))

        # Non-linearity
        layers.append(_get_activation(activation))

        # Optional spatial down-sampling
        if pool_size is not None:
            layers.append(nn.MaxPool2d(kernel_size=pool_size, stride=pool_size))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (N, C_in, H, W).

        Returns:
            Output tensor of shape (N, C_out, H', W').
        """
        return self.block(x)


class LeNetCNN(nn.Module):
    """
    A configurable LeNet-style CNN.

    Architecture:
        features  : sequence of ConvBlocks (configurable depth, channels, etc.)
        classifier: sequence of fully-connected layers with optional dropout

    The flat dimension fed into the classifier is computed automatically by
    doing a single dummy forward pass through ``self.features``.

    Args:
        in_channels: Number of input image channels (e.g. 1 for greyscale, 3 for RGB).
        input_hw: (height, width) of the input image.
        num_classes: Number of output classes (logits).
        conv_channels: Output channel counts for each ConvBlock.
        kernel_sizes: Kernel size for each ConvBlock (must match len(conv_channels)).
        pool_sizes: Pool size (or None) for each ConvBlock.
        paddings: Padding for each ConvBlock.
        fc_dims: Hidden dimensions for the classifier MLP.
        activation: Activation name used in both feature and classifier stages.
        dropout_p: Dropout probability applied between classifier layers.
        batch_norm: Whether to use batch normalisation in ConvBlocks.
    """

    def __init__(
        self,
        in_channels: int = 3,
        input_hw: Tuple[int, int] = (32, 32),
        num_classes: int = 10,
        conv_channels: List[int] = None,
        kernel_sizes: List[int] = None,
        pool_sizes: List[Optional[int]] = None,
        paddings: List[int] = None,
        fc_dims: List[int] = None,
        activation: str = "relu",
        dropout_p: float = 0.0,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()

        # --- Handle None defaults for mutable arguments ---
        if conv_channels is None:
            conv_channels = [6, 16]
        if kernel_sizes is None:
            kernel_sizes = [5, 5]
        if pool_sizes is None:
            pool_sizes = [2, 2]
        if paddings is None:
            paddings = [0, 0]
        if fc_dims is None:
            fc_dims = [120, 84]

        num_conv = len(conv_channels)
        assert len(kernel_sizes) == num_conv, "kernel_sizes must match conv_channels length"
        assert len(pool_sizes) == num_conv, "pool_sizes must match conv_channels length"
        assert len(paddings) == num_conv, "paddings must match conv_channels length"

        # --- Build convolutional feature extractor ---
        conv_blocks: List[nn.Module] = []
        prev_channels = in_channels
        for i in range(num_conv):
            conv_blocks.append(
                ConvBlock(
                    in_channels=prev_channels,
                    out_channels=conv_channels[i],
                    kernel_size=kernel_sizes[i],
                    padding=paddings[i],
                    stride=1,
                    activation=activation,
                    batch_norm=batch_norm,
                    pool_size=pool_sizes[i],
                )
            )
            prev_channels = conv_channels[i]

        self.features = nn.Sequential(*conv_blocks)

        # --- Compute flattened dimension automatically ---
        # Run a single dummy tensor through features (no gradient needed)
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, *input_hw)
            flat_out = self.features(dummy).view(1, -1)
            _flat_dim = flat_out.shape[1]

        # --- Build fully-connected classifier ---
        fc_layers: List[nn.Module] = []
        in_dim = _flat_dim
        for hidden_dim in fc_dims:
            fc_layers.append(nn.Linear(in_dim, hidden_dim))
            fc_layers.append(_get_activation(activation))
            if dropout_p > 0.0:
                fc_layers.append(nn.Dropout(p=dropout_p))
            in_dim = hidden_dim

        # Final output layer — no activation after logits
        fc_layers.append(nn.Linear(in_dim, num_classes))

        self.classifier = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass.

        Args:
            x: Input image tensor of shape (N, C, H, W).

        Returns:
            Logits tensor of shape (N, num_classes).
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)   # flatten spatial dims
        x = self.classifier(x)
        return x

    def get_feature_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Collect the output activation map after each ConvBlock in self.features.

        Args:
            x: Input image tensor of shape (N, C, H, W).

        Returns:
            List of tensors, one per ConvBlock, each of shape (N, C_out, H', W').
        """
        feature_maps: List[torch.Tensor] = []
        for conv_block in self.features.children():
            x = conv_block(x)
            feature_maps.append(x)
        return feature_maps
