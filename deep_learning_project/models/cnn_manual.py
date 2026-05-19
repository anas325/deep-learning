"""
Manual (pure NumPy) implementations of core CNN operations.

No PyTorch — only numpy. Intended for pedagogical clarity:
every loop variable is named to match the concept it indexes.
"""

import numpy as np


def output_size_formula(
    input_size: int,
    kernel_size: int,
    padding: int = 0,
    stride: int = 1,
) -> int:
    """
    Compute the spatial output size for a convolution or pooling operation.

    Formula: (input_size + 2*padding - kernel_size) // stride + 1

    The +2*padding widens the input on both sides.
    Subtracting kernel_size accounts for the kernel needing to fit fully.
    Dividing by stride counts how many positions the kernel can step to.
    The +1 includes the starting position itself.

    Args:
        input_size: Spatial size of the input (height or width).
        kernel_size: Spatial size of the kernel/filter.
        padding: Zero-padding added to each side of the input.
        stride: Step size between consecutive kernel positions.

    Returns:
        Integer output spatial size.
    """
    return (input_size + 2 * padding - kernel_size) // stride + 1


def cross_correlate_2d(
    input_map: np.ndarray,
    kernel: np.ndarray,
    padding: int = 0,
    stride: int = 1,
) -> np.ndarray:
    """
    2-D cross-correlation of a single-channel input with a single kernel.

    Cross-correlation slides the kernel over the input and computes the
    element-wise product sum at each position (unlike convolution, the
    kernel is NOT flipped — this matches what PyTorch's Conv2d does).

    Args:
        input_map: 2-D array of shape (H, W).
        kernel: 2-D array of shape (kH, kW).
        padding: Number of zeros to pad on each side of H and W.
        stride: Step size between kernel placements.

    Returns:
        2-D output array of shape (out_H, out_W).
    """
    H, W = input_map.shape
    kH, kW = kernel.shape

    # Pad the input with zeros on all four sides if requested
    if padding > 0:
        input_map = np.pad(
            input_map,
            pad_width=((padding, padding), (padding, padding)),
            mode="constant",
            constant_values=0,
        )
        # Update H, W to reflect padded dimensions
        H, W = input_map.shape

    # Compute output spatial dimensions using the standard formula
    out_H = output_size_formula(H, kH, padding=0, stride=stride)
    out_W = output_size_formula(W, kW, padding=0, stride=stride)

    output = np.zeros((out_H, out_W), dtype=np.float64)

    # Slide the kernel over every valid (h_out, w_out) output position
    for h_out in range(out_H):
        for w_out in range(out_W):
            # Top-left corner of the input patch this output position reads from
            h_start = h_out * stride
            w_start = w_out * stride

            # Accumulate element-wise products across the kernel window
            patch_sum = 0.0
            for ki in range(kH):
                for kj in range(kW):
                    patch_sum += input_map[h_start + ki, w_start + kj] * kernel[ki, kj]

            output[h_out, w_out] = patch_sum

    return output


def cross_correlate_2d_batch(
    inputs: np.ndarray,
    kernels: np.ndarray,
    padding: int = 0,
    stride: int = 1,
) -> np.ndarray:
    """
    Multi-channel 2-D cross-correlation (batch over input and output channels).

    For each output channel, the result is the sum of cross-correlations
    between every input channel and its corresponding kernel slice.
    This is exactly what a single Conv2d filter computes.

    Args:
        inputs: Array of shape (C_in, H, W).
        kernels: Array of shape (C_out, C_in, kH, kW).
        padding: Zero-padding applied to spatial dims of each input channel.
        stride: Stride for the sliding kernel.

    Returns:
        Output array of shape (C_out, out_H, out_W).
    """
    C_in, H, W = inputs.shape
    C_out, _, kH, kW = kernels.shape

    # Determine output spatial size from the first channel (all identical)
    out_H = output_size_formula(H, kH, padding=padding, stride=stride)
    out_W = output_size_formula(W, kW, padding=padding, stride=stride)

    output = np.zeros((C_out, out_H, out_W), dtype=np.float64)

    for c_out in range(C_out):
        # Sum contributions from every input channel for this output channel
        for c_in in range(C_in):
            output[c_out] += cross_correlate_2d(
                inputs[c_in],
                kernels[c_out, c_in],
                padding=padding,
                stride=stride,
            )

    return output


def max_pool_2d(
    input_map: np.ndarray,
    pool_size: int = 2,
    stride: int = 2,
) -> np.ndarray:
    """
    2-D max pooling applied independently to each channel.

    Max pooling keeps the largest activation in each pooling window,
    providing spatial down-sampling and translation invariance.

    Args:
        input_map: Array of shape (C, H, W).
        pool_size: Height and width of the pooling window.
        stride: Step size between pooling windows.

    Returns:
        Pooled array of shape (C, out_H, out_W).
    """
    C, H, W = input_map.shape

    out_H = output_size_formula(H, pool_size, padding=0, stride=stride)
    out_W = output_size_formula(W, pool_size, padding=0, stride=stride)

    output = np.zeros((C, out_H, out_W), dtype=input_map.dtype)

    for c in range(C):
        for h_out in range(out_H):
            for w_out in range(out_W):
                h_start = h_out * stride
                w_start = w_out * stride
                # Extract the pooling window and take the maximum value
                window = input_map[
                    c,
                    h_start : h_start + pool_size,
                    w_start : w_start + pool_size,
                ]
                output[c, h_out, w_out] = np.max(window)

    return output


def avg_pool_2d(
    input_map: np.ndarray,
    pool_size: int = 2,
    stride: int = 2,
) -> np.ndarray:
    """
    2-D average pooling applied independently to each channel.

    Average pooling computes the mean activation in each pooling window,
    providing a smoother down-sampling than max pooling.

    Args:
        input_map: Array of shape (C, H, W).
        pool_size: Height and width of the pooling window.
        stride: Step size between pooling windows.

    Returns:
        Pooled array of shape (C, out_H, out_W).
    """
    C, H, W = input_map.shape

    out_H = output_size_formula(H, pool_size, padding=0, stride=stride)
    out_W = output_size_formula(W, pool_size, padding=0, stride=stride)

    output = np.zeros((C, out_H, out_W), dtype=np.float64)

    for c in range(C):
        for h_out in range(out_H):
            for w_out in range(out_W):
                h_start = h_out * stride
                w_start = w_out * stride
                # Extract the pooling window and take the mean value
                window = input_map[
                    c,
                    h_start : h_start + pool_size,
                    w_start : w_start + pool_size,
                ]
                output[c, h_out, w_out] = np.mean(window)

    return output
