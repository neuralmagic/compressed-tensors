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
    """Test pack_bitmasks optimizations."""
    
    def test_pack_bitmasks_correctness_cpu(self):
        """Test PyTorch implementation matches NumPy on CPU."""
        test_shapes = [
            (1, 8),
            (1, 16),
            (10, 7),
            (10, 8),
            (10, 9),
            (100, 100),
            (128, 256),
            (1000, 1000),
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
        test_shapes = [(128, 256), (1024, 1024)]
        
        for shape in test_shapes:
            mask = torch.rand(shape) > 0.5
            mask_gpu = mask.cuda()
            
            # GPU implementation
            packed_gpu = pack_bitmasks(mask_gpu)
            assert packed_gpu.is_cuda, "Result should stay on GPU"
            
            # CPU reference
            packed_cpu = pack_bitmasks(mask)
            
            assert torch.equal(packed_gpu.cpu(), packed_cpu), \
                f"GPU result differs from CPU for shape {shape}"
    
    def test_pack_unpack_roundtrip(self):
        """Test pack/unpack roundtrip preserves data."""
        shapes = [(10, 16), (128, 256), (100, 999)]
        
        for shape in shapes:
            mask = torch.rand(shape) > 0.5
            packed = pack_bitmasks(mask)
            unpacked = unpack_bitmasks(packed, list(shape))
            
            assert torch.equal(mask, unpacked), \
                f"Roundtrip failed for shape {shape}"
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Empty tensor
        empty = torch.empty(0, 0, dtype=torch.bool)
        packed = pack_bitmasks(empty)
        assert packed.shape == (0, 0)
        
        # Single element
        single = torch.tensor([[True]])
        packed = pack_bitmasks(single)
        assert packed.shape == (1, 1)
        assert packed[0, 0] == 1
        
        # All False
        all_false = torch.zeros(10, 16, dtype=torch.bool)
        packed = pack_bitmasks(all_false)
        assert torch.all(packed == 0)
        
        # All True
        all_true = torch.ones(10, 16, dtype=torch.bool)
        packed = pack_bitmasks(all_true)
        expected = torch.full((10, 2), 255, dtype=torch.uint8)
        assert torch.equal(packed, expected)


class TestSparse24Compression:
    """Test sparse 2:4 compression optimizations."""
    
    def test_compression_preserves_sparsity(self):
        """Test that compression preserves 2:4 sparsity pattern."""
        tensor = torch.randn(128, 256)
        
        # Get 2:4 mask
        mask = get_24_bytemasks(tensor)
        sparsity = (~mask).sum().item() / mask.numel()
        assert abs(sparsity - 0.5) < 0.01, "Should have ~50% sparsity"
        
        # Compress and decompress
        compressed, bitmask = sparse24_bitmask_compress(tensor)
        decompressed = sparse24_bitmask_decompress(compressed, bitmask, tensor.shape)
        
        # Check sparsity preserved
        decompressed_sparsity = (decompressed == 0).sum().item() / decompressed.numel()
        assert abs(decompressed_sparsity - 0.5) < 0.01, "Decompressed should maintain sparsity"
        
        # Check values preserved
        assert torch.allclose(tensor[mask], decompressed[mask], rtol=1e-5)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_compression(self):
        """Test compression works correctly on GPU."""
        tensor = torch.randn(256, 512).cuda()
        
        # Compress on GPU
        compressed_tensor = Sparse24BitMaskTensor.from_dense(tensor)
        
        # Check results moved to CPU for storage
        assert compressed_tensor.compressed.device.type == "cpu"
        assert compressed_tensor.bitmask.device.type == "cpu"
        
        # Decompress and verify
        decompressed = compressed_tensor.decompress()
        mask = get_24_bytemasks(tensor.cpu())
        
        assert torch.allclose(tensor.cpu()[mask], decompressed[mask], rtol=1e-5)
    
    def test_various_dtypes(self):
        """Test compression works with various dtypes."""
        dtypes = [torch.float32, torch.float16, torch.bfloat16]
        
        for dtype in dtypes:
            if dtype == torch.bfloat16 and not torch.cuda.is_available():
                continue
                
            tensor = torch.randn(64, 128, dtype=dtype)
            compressed_tensor = Sparse24BitMaskTensor.from_dense(tensor)
            decompressed = compressed_tensor.decompress()
            
            mask = get_24_bytemasks(tensor)
            assert torch.allclose(
                tensor[mask].float(), 
                decompressed[mask].float(), 
                rtol=1e-3 if dtype == torch.float16 else 1e-5
            )
    
    def test_deterministic_sparsity(self):
        """Test that sparsity pattern is deterministic."""
        tensor = torch.randn(128, 256)
        
        # Get mask multiple times
        mask1 = get_24_bytemasks(tensor)
        mask2 = get_24_bytemasks(tensor)
        mask3 = get_24_bytemasks(tensor)
        
        assert torch.equal(mask1, mask2)
        assert torch.equal(mask2, mask3)
    
    def test_topk_optimization(self):
        """Test that topk with sorted=False produces correct results."""
        tensor = torch.randn(128, 256)
        
        # Original implementation (sorted=True)
        reshaped = tensor.view(-1, 4)
        abs_vals = reshaped.abs()
        topk_sorted = abs_vals.topk(2, dim=1, largest=True, sorted=True).indices
        
        # Optimized implementation (sorted=False)
        topk_unsorted = abs_vals.topk(2, dim=1, largest=True, sorted=False).indices
        
        # Both should select the same elements (order doesn't matter)
        mask_sorted = torch.zeros_like(reshaped, dtype=torch.bool)
        mask_sorted.scatter_(1, topk_sorted, True)
        
        mask_unsorted = torch.zeros_like(reshaped, dtype=torch.bool)
        mask_unsorted.scatter_(1, topk_unsorted, True)
        
        assert torch.equal(mask_sorted, mask_unsorted)


class TestPerformance:
    """Performance regression tests."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_faster_than_cpu_transfer(self):
        """Test that GPU processing is faster than CPU transfer for large tensors."""
        import time
        
        tensor = torch.randn(4096, 4096).cuda()
        
        # Time GPU processing
        torch.cuda.synchronize()
        start = time.time()
        compressed, bitmask = sparse24_bitmask_compress(tensor)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        
        # Time with CPU transfer
        torch.cuda.synchronize()
        start = time.time()
        tensor_cpu = tensor.cpu()
        compressed_cpu, bitmask_cpu = sparse24_bitmask_compress(tensor_cpu)
        cpu_time = time.time() - start
        
        # GPU should be faster for large tensors
        assert gpu_time < cpu_time, \
            f"GPU ({gpu_time:.3f}s) should be faster than CPU transfer ({cpu_time:.3f}s)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])