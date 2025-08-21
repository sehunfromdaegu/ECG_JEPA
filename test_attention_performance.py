import torch
import torch.nn as nn
import time
from ecg_jepa import ecg_jepa

def benchmark_model(pos_type='rope', batch_size=32, num_iterations=10):
    """Benchmark the model with different batch sizes."""
    
    model = ecg_jepa(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=16,
        predictor_embed_dim=384,
        predictor_depth=6,
        predictor_num_heads=12,
        drop_path_rate=0.1,
        mask_scale=(0.7, 0.8),
        mask_type='random',
        pos_type=pos_type,
        c=8,
        p=50,
        t=50
    ).to('cuda')
    
    model.train()
    
    # Warm up
    x = torch.randn(batch_size, 8, 2500).cuda()
    for _ in range(3):
        loss = model(x)
        loss.backward()
    
    torch.cuda.synchronize()
    
    # Benchmark
    forward_times = []
    backward_times = []
    
    for _ in range(num_iterations):
        x = torch.randn(batch_size, 8, 2500).cuda()
        
        torch.cuda.synchronize()
        start = time.time()
        loss = model(x)
        torch.cuda.synchronize()
        forward_time = time.time() - start
        forward_times.append(forward_time)
        
        start = time.time()
        loss.backward()
        torch.cuda.synchronize()
        backward_time = time.time() - start
        backward_times.append(backward_time)
    
    avg_forward = sum(forward_times) / len(forward_times)
    avg_backward = sum(backward_times) / len(backward_times)
    avg_total = avg_forward + avg_backward
    
    return avg_forward, avg_backward, avg_total

if __name__ == "__main__":
    print("Benchmarking ECG-JEPA with scaled_dot_product_attention...")
    print("=" * 60)
    
    for batch_size in [16, 32, 64]:
        print(f"\nBatch size: {batch_size}")
        print("-" * 30)
        
        # Test with RoPE
        fwd, bwd, total = benchmark_model('rope', batch_size, 10)
        print(f"RoPE:   Forward: {fwd*1000:.2f}ms, Backward: {bwd*1000:.2f}ms, Total: {total*1000:.2f}ms")
        
        # Test with sincos
        fwd, bwd, total = benchmark_model('sincos', batch_size, 10)
        print(f"Sincos: Forward: {fwd*1000:.2f}ms, Backward: {bwd*1000:.2f}ms, Total: {total*1000:.2f}ms")
    
    print("\n" + "=" * 60)
    print("Note: Using scaled_dot_product_attention provides:")
    print("- Automatic selection of optimal backend (FlashAttention, etc.)")
    print("- Better memory efficiency for large sequences")
    print("- Potential speedups with compatible hardware")