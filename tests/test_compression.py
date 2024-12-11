import torch
import unittest
from ../src/ml_optimizer import TensorCompressor


class TestTensorCompressor(unittest.TestCase):
    def setUp(self):
        self.compressor = TensorCompressor(verbose=False)
        self.sample_tensor = torch.randn(100, 100)

    def test_quantization(self):
        quantized = self.compressor.quantization_compress(self.sample_tensor)
        self.assertEqual(quantized.shape, self.sample_tensor.shape)

    def test_pruning(self):
        pruned = self.compressor.pruning_compress(self.sample_tensor)
        self.assertEqual(pruned.shape, self.sample_tensor.shape)

    def test_svd_compression(self):
        result = self.compressor.svd_compress(self.sample_tensor)
        self.assertIn('compressed_tensor', result)


if __name__ == '__main__':
    unittest.main()