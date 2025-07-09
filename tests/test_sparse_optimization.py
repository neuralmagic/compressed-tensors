import pytest
import torch
import numpy as np
from compressed_tensors.utils.helpers import pack_bitmasks, unpack_bitmasks
from compressed_tensors.compressors.sparse_compressors.sparse_24_bitmask import (
    get_24_bytemasks,
    sparse24_bitmask_compress,
    sparse24_bitmask_decompress,
    Sparse24BitMaskTensor,
)


class TestPackBitmasks:
    """Test pack_bitmasks optimizations for correctness and edge cases."""
    
    def test_pack_bitmasks_correctness(self):
        """Test PyTorch implementation matches NumPy reference."""
        # Test various shapes to ensure correctness across different scenarios
        # We specifically test:
        # - Multiple of 8 columns (no padding needed)
        # - Non-multiple of 8 columns (tests edge handling)
        # - Larger tensors (tests performance at scale)
        test_shapes = [
            (10, 8),    # Multiple of 8
            (10, 9),    # Not multiple of 8
            (128, 256), # Larger tensor
        ]
        
        for shape in test_shapes:
            mask = torch.rand(shape) > 0.5
            
            # PyTorch implementation
            packed_torch = pack_bitmasks(mask)
            
            # NumPy reference
            packed_numpy = torch.from_numpy(
                np.packbits(mask.numpy(), axis=-1, bitorder="little")
            )
            
            assert torch.equal(packed_torch, packed_numpy), \
                f"Mismatch for shape {shape}: PyTorch != NumPy"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_pack_bitmasks_gpu(self):
        """Test GPU implementation produces correct results."""
        mask = torch.rand(128, 256) > 0.5
        mask_gpu = mask.cuda()
        
        # GPU implementation
        packed_gpu = pack_bitmasks(mask_gpu)
        assert packed_gpu.is_cuda, "Result should stay on GPU"
        
        # CPU reference
        packed_cpu = pack_bitmasks(mask)
        
        assert torch.equal(packed_gpu.cpu(), packed_cpu), \
            "GPU result differs from CPU"
    
    def test_pack_unpack_roundtrip(self):
        """Test pack/unpack roundtrip preserves data."""
        shape = (128, 256)
        mask = torch.rand(shape) > 0.5
        
        packed = pack_bitmasks(mask)
        unpacked = unpack_bitmasks(packed, list(shape))
        
        assert torch.equal(mask, unpacked), "Roundtrip failed"
    
    def test_invalid_shape(self):
        """Test shape validation."""
        # The pack_bitmasks function is designed for 2D tensors only
        # This is a deliberate design choice as the compression format
        # expects row-major packing of 2D weight matrices
        
        # 1D tensor should raise error
        with pytest.raises(ValueError, match="expects a 2D tensor"):
            pack_bitmasks(torch.tensor([True, False, True]))
        
        # 3D tensor should raise error
        with pytest.raises(ValueError, match="expects a 2D tensor"):
            pack_bitmasks(torch.ones(2, 3, 4, dtype=torch.bool))
    
    def test_edge_cases(self):
        """Test edge cases for pack_bitmasks."""
        # Empty tensor
        empty = torch.empty(0, 0, dtype=torch.bool)
        packed = pack_bitmasks(empty)
        assert packed.shape == (0, 0)
        
        # Single element
        single = torch.tensor([[True]])
        packed = pack_bitmasks(single)
        assert packed.shape == (1, 1)
        assert packed[0, 0] == 1


class TestSparse24Compression:
    """Test sparse 2:4 compression functionality."""
    
    def test_compression_correctness(self):
        """Test that compression/decompression preserves correct values."""
        tensor = torch.randn(128, 256)
        
        # Get 2:4 mask and verify sparsity
        # For 2:4 sparsity, exactly 2 out of every 4 elements are kept
        # This results in exactly 50% sparsity
        mask = get_24_bytemasks(tensor)
        sparsity = (~mask).sum().item() / mask.numel()
        assert abs(sparsity - 0.5) < 0.01, "Should have ~50% sparsity"
        
        # Compress and decompress
        compressed, bitmask = sparse24_bitmask_compress(tensor)
        decompressed = sparse24_bitmask_decompress(compressed, bitmask, tensor.shape)
        
        # Check values are preserved for non-zero elements
        assert torch.allclose(tensor[mask], decompressed[mask], rtol=1e-5)
        
        # Check zeros are preserved
        assert torch.all(decompressed[~mask] == 0)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_compression(self):
        """Test compression works correctly on GPU without unnecessary transfers."""
        tensor = torch.randn(256, 512).cuda()
        
        # Compress on GPU
        compressed_tensor = Sparse24BitMaskTensor.from_dense(tensor)
        
        # Storage should be on CPU
        assert compressed_tensor.compressed.device.type == "cpu"
        assert compressed_tensor.bitmask.device.type == "cpu"
        
        # Verify correctness
        decompressed = compressed_tensor.decompress()
        mask = get_24_bytemasks(tensor.cpu())
        assert torch.allclose(tensor.cpu()[mask], decompressed[mask], rtol=1e-5)
    
    def test_invalid_tensor_size(self):
        """Test validation for tensor size."""
        # Tensor with size not multiple of 4
        tensor = torch.randn(10, 7)  # 70 elements, not divisible by 4
        
        with pytest.raises(ValueError, match="multiple of 4"):
            get_24_bytemasks(tensor)
    
    def test_various_dtypes(self):
        """Test compression with different data types."""
        dtypes = [torch.float32, torch.float16]
        if torch.cuda.is_available():
            dtypes.append(torch.bfloat16)
        
        for dtype in dtypes:
            tensor = torch.randn(64, 128, dtype=dtype)
            compressed_tensor = Sparse24BitMaskTensor.from_dense(tensor)
            decompressed = compressed_tensor.decompress()
            
            mask = get_24_bytemasks(tensor)
            assert torch.allclose(
                tensor[mask].float(), 
                decompressed[mask].float(), 
                rtol=1e-3 if dtype == torch.float16 else 1e-5
            ), f"Compression failed for dtype {dtype}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestPerformanceRegression:
    """Performance regression tests - run only when GPU is available."""
    
    def test_gpu_performance_maintained(self):
        """Ensure GPU processing doesn't regress to CPU transfers."""
        import time
        
        tensor = torch.randn(2048, 2048).cuda()
        
        # Warm up GPU to avoid initialization overhead in timing
        _ = sparse24_bitmask_compress(tensor)
        torch.cuda.synchronize()
        
        # Time compression
        start = time.time()
        compressed, bitmask = sparse24_bitmask_compress(tensor)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        
        # Performance threshold based on empirical testing
        # 100ms is a conservative upper bound for 2048x2048 on modern GPUs
        # This test will catch if someone accidentally introduces CPU transfers
        assert gpu_time < 0.1, f"GPU compression too slow: {gpu_time:.3f}s"
        
        # Verify compression stayed on GPU during processing
        # CPU transfer should only happen in Sparse24BitMaskTensor.from_dense()
        # after compression is complete
        assert compressed.is_cuda, "Compression should stay on GPU"
        assert bitmask.is_cuda, "Bitmask should stay on GPU"