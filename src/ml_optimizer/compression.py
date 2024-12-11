import torch
import numpy as np
from typing import Union, Dict, Any


class TensorCompressor:
    def __init__(self, verbose: bool = True):
        """
        Initialize tensor compressor with optional verbose mode.

        Args:
            verbose (bool): Print detailed compression information
        """
        self.verbose = verbose

    def quantization_compress(self, tensor: torch.Tensor, bits: int = 8) -> torch.Tensor:
        """
        Compress tensor using quantization technique.

        Args:
            tensor (torch.Tensor): Input tensor to compress
            bits (int): Number of bits for quantization (default 8)

        Returns:
            Quantized tensor
        """
        # Check input tensor
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Input must be a PyTorch tensor")

        # Calculate quantization parameters
        min_val = tensor.min()
        max_val = tensor.max()

        # Linear quantization
        levels = 2 ** bits
        scale = (max_val - min_val) / (levels - 1)

        # Quantization process
        quantized = torch.round((tensor - min_val) / scale)
        quantized = quantized.clamp(0, levels - 1)

        # Optional verbose logging
        if self.verbose:
            original_size = tensor.element_size() * tensor.nelement()
            compressed_size = quantized.element_size() * quantized.nelement()
            compression_ratio = original_size / compressed_size

            print(f"Quantization Compression:")
            print(f"Original Size: {original_size} bytes")
            print(f"Compressed Size: {compressed_size} bytes")
            print(f"Compression Ratio: {compression_ratio:.2f}")

        return quantized

    def pruning_compress(self, tensor: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Compress tensor by removing small magnitude weights (pruning).

        Args:
            tensor (torch.Tensor): Input tensor to compress
            threshold (float): Threshold for weight removal

        Returns:
            Pruned tensor
        """
        # Calculate standard deviation as pruning reference
        std_dev = torch.std(tensor)

        # Create mask for significant weights
        mask = torch.abs(tensor) > (std_dev * threshold)

        # Apply mask (zero out small weights)
        pruned_tensor = tensor * mask.float()

        if self.verbose:
            total_weights = tensor.numel()
            zero_weights = torch.sum(pruned_tensor == 0).item()
            sparsity = zero_weights / total_weights * 100

            print(f"Pruning Compression:")
            print(f"Total Weights: {total_weights}")
            print(f"Zeroed Weights: {zero_weights}")
            print(f"Sparsity: {sparsity:.2f}%")

        return pruned_tensor

    def svd_compress(self, tensor: torch.Tensor, k: int = None) -> Dict[str, torch.Tensor]:
        """
        Compress tensor using Singular Value Decomposition (SVD).

        Args:
            tensor (torch.Tensor): Input tensor to compress
            k (int): Number of singular values to keep

        Returns:
            Dictionary with compressed components
        """
        # Convert to 2D matrix for SVD
        original_shape = tensor.shape
        matrix = tensor.view(tensor.size(0), -1)

        # Perform SVD
        U, S, V = torch.svd(matrix)

        # Determine compression rank
        if k is None:
            k = min(matrix.shape) // 2

        # Truncate singular values
        U_compressed = U[:, :k]
        S_compressed = S[:k]
        V_compressed = V[:, :k]

        # Reconstruct compressed tensor
        compressed_matrix = U_compressed @ torch.diag(S_compressed) @ V_compressed.t()
        compressed_tensor = compressed_matrix.view(original_shape)

        if self.verbose:
            original_size = matrix.numel()
            compressed_size = compressed_matrix.numel()
            compression_ratio = original_size / compressed_size

            print(f"SVD Compression:")
            print(f"Original Shape: {original_shape}")
            print(f"Compressed Rank: {k}")
            print(f"Compression Ratio: {compression_ratio:.2f}")

        return {
            'compressed_tensor': compressed_tensor,
            'U': U_compressed,
            'S': S_compressed,
            'V': V_compressed
        }

    def quality_assessment(self, original: torch.Tensor, compressed: torch.Tensor) -> Dict[str, float]:
        """
        Assess compression quality using various metrics.

        Args:
            original (torch.Tensor): Original tensor
            compressed (torch.Tensor): Compressed tensor

        Returns:
            Dictionary with quality metrics
        """
        # Mean Squared Error
        mse = torch.nn.functional.mse_loss(original, compressed)

        # Peak Signal-to-Noise Ratio
        max_val = torch.max(original)
        psnr = 10 * torch.log10(max_val ** 2 / mse)

        # Compression Ratio
        original_size = original.element_size() * original.nelement()
        compressed_size = compressed.element_size() * compressed.nelement()
        compression_ratio = original_size / compressed_size

        return {
            'mean_squared_error': mse.item(),
            'peak_signal_noise_ratio': psnr.item(),
            'compression_ratio': compression_ratio
        }


# Example usage function
def demo_compression():
    """
    Demonstrate tensor compression techniques
    """
    # Create sample tensor
    sample_tensor = torch.randn(100, 100)

    # Initialize compressor
    compressor = TensorCompressor(verbose=True)

    # Quantization
    print("\n--- Quantization Compression ---")
    quantized = compressor.quantization_compress(sample_tensor)

    # Pruning
    print("\n--- Pruning Compression ---")
    pruned = compressor.pruning_compress(sample_tensor)

    # SVD
    print("\n--- SVD Compression ---")
    svd_result = compressor.svd_compress(sample_tensor)

    # Quality Assessment
    print("\n--- Quality Assessment ---")
    q_quant = compressor.quality_assessment(sample_tensor, quantized)
    q_pruned = compressor.quality_assessment(sample_tensor, pruned)
    q_svd = compressor.quality_assessment(sample_tensor, svd_result['compressed_tensor'])

    print("Quantization Quality:", q_quant)
    print("Pruning Quality:", q_pruned)
    print("SVD Quality:", q_svd)


# Run demo if script is executed directly
if __name__ == "__main__":
    demo_compression()