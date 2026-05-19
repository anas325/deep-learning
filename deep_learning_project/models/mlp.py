import torch
import torch.nn as nn
from typing import List, Tuple


def _get_activation(name: str) -> nn.Module:
    """Return an activation module for the given name string."""
    activations = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
    }
    if name not in activations:
        raise ValueError(f"Unsupported activation '{name}'. Choose from {list(activations.keys())}")
    return activations[name]


def build_sequential_mlp(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    activation: str = "relu",
    dropout_p: float = 0.0,
    batch_norm: bool = False,
) -> nn.Sequential:
    """
    Factory function that builds an MLP as an nn.Sequential.

    Each hidden layer block: Linear(in, out) -> [BatchNorm1d(out)] -> Activation -> [Dropout(p)]
    Final layer: Linear(last_hidden, output_dim) with NO activation.

    Args:
        input_dim: Number of input features.
        hidden_dims: List of hidden layer widths.
        output_dim: Number of output features (logits).
        activation: One of "relu", "tanh", "sigmoid".
        dropout_p: Dropout probability (0.0 disables dropout).
        batch_norm: Whether to insert BatchNorm1d after each linear in hidden layers.

    Returns:
        nn.Sequential representing the full MLP.
    """
    layers: List[nn.Module] = []
    in_dim = input_dim

    for out_dim in hidden_dims:
        layers.append(nn.Linear(in_dim, out_dim))
        if batch_norm:
            layers.append(nn.BatchNorm1d(out_dim))
        layers.append(_get_activation(activation))
        if dropout_p > 0.0:
            layers.append(nn.Dropout(p=dropout_p))
        in_dim = out_dim

    # Final layer — no activation, no batch norm, no dropout
    layers.append(nn.Linear(in_dim, output_dim))

    return nn.Sequential(*layers)


class MLP(nn.Module):
    """
    Multi-Layer Perceptron implemented as a proper nn.Module.

    Architecture per hidden layer: Linear -> [BatchNorm1d] -> Activation -> [Dropout]
    Output layer: Linear only (no activation).

    Attributes:
        layers: nn.ModuleList of all Linear layers (hidden + output).
        norms: nn.ModuleList of BatchNorm1d layers (one per hidden layer, or empty).
        drops: nn.ModuleList of Dropout layers (one per hidden layer, or empty).
        activation_fn: Callable activation applied after each hidden linear.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = "relu",
        dropout_p: float = 0.0,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()

        self.activation_fn = _get_activation(activation)

        # Build all Linear layers: one per hidden dim + one output layer
        all_dims = [input_dim] + list(hidden_dims) + [output_dim]
        self.layers = nn.ModuleList(
            [nn.Linear(all_dims[i], all_dims[i + 1]) for i in range(len(all_dims) - 1)]
        )

        # BatchNorm layers for hidden layers only (not the output layer)
        if batch_norm:
            self.norms = nn.ModuleList([nn.BatchNorm1d(dim) for dim in hidden_dims])
        else:
            self.norms = nn.ModuleList()

        # Dropout layers for hidden layers only (not the output layer)
        if dropout_p > 0.0:
            self.drops = nn.ModuleList([nn.Dropout(p=dropout_p) for _ in hidden_dims])
        else:
            self.drops = nn.ModuleList()

        self._num_hidden = len(hidden_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, input_dim).

        Returns:
            Logits tensor of shape (batch, output_dim).
        """
        for i in range(self._num_hidden):
            x = self.layers[i](x)
            if self.norms:
                x = self.norms[i](x)
            x = self.activation_fn(x)
            if self.drops:
                x = self.drops[i](x)

        # Output layer — no activation
        x = self.layers[self._num_hidden](x)
        return x

    def get_num_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_layer_shapes(self) -> List[Tuple[int, int]]:
        """Return a list of (in_features, out_features) for each Linear layer."""
        return [(layer.in_features, layer.out_features) for layer in self.layers]


# ---------------------------------------------------------------------------
# Initialization strategies
# ---------------------------------------------------------------------------

def init_normal(module: nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    """Initialize Linear weights with a normal distribution and biases to zero."""
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=mean, std=std)
        nn.init.zeros_(module.bias)


def init_constant(module: nn.Module, val: float = 0.01) -> None:
    """Initialize Linear weights with a constant value and biases to zero."""
    if isinstance(module, nn.Linear):
        nn.init.constant_(module.weight, val)
        nn.init.zeros_(module.bias)


def init_xavier(module: nn.Module) -> None:
    """Initialize Linear weights with Xavier uniform and biases to zero."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)


INIT_STRATEGIES = {
    "normal": init_normal,
    "constant": init_constant,
    "xavier": init_xavier,
}
