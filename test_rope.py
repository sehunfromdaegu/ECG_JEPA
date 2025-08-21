import torch
import torch.nn as nn
from ecg_jepa import ecg_jepa

def test_rope_implementation():
    """Test the RoPE implementation in ECG-JEPA."""
    
    print("Testing RoPE implementation in ECG-JEPA...")
    
    # Test with RoPE
    print("\n1. Testing with RoPE:")
    model_rope = ecg_jepa(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=16,
        predictor_embed_dim=384,
        predictor_depth=6,
        predictor_num_heads=12,
        pos_type='rope'  # Use RoPE
    )
    
    # Create dummy input: (batch_size, channels, time)
    batch_size = 2
    channels = 8
    time_samples = 2500  # 50 patches * 50 timesteps each
    x = torch.randn(batch_size, channels, time_samples)
    
    # Test forward pass
    try:
        model_rope.train()  # Enable training mode to test coordinate shift
        loss = model_rope(x)
        print(f"  Forward pass successful with RoPE!")
        print(f"  Loss shape: {loss.shape}")
        print(f"  Loss value: {loss.item():.4f}")
    except Exception as e:
        print(f"  Error with RoPE: {e}")
        return False
    
    # Test with legacy sincos for comparison
    print("\n2. Testing with legacy sincos:")
    model_sincos = ecg_jepa(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=16,
        predictor_embed_dim=384,
        predictor_depth=6,
        predictor_num_heads=12,
        pos_type='sincos'  # Use sincos
    )
    
    try:
        loss_sincos = model_sincos(x)
        print(f"  Forward pass successful with sincos!")
        print(f"  Loss value: {loss_sincos.item():.4f}")
    except Exception as e:
        print(f"  Error with sincos: {e}")
        return False
    
    # Test representation extraction with RoPE
    print("\n3. Testing representation extraction with RoPE:")
    try:
        model_rope.eval()  # Disable training mode (no coordinate shift)
        with torch.no_grad():
            # Test the encoder's representation method
            x_repr = torch.randn(batch_size, 8, 2500)  # Direct input for representation
            repr_out = model_rope.encoder.representation(x_repr)
            print(f"  Representation extraction successful!")
            print(f"  Output shape: {repr_out.shape}")
            print(f"  Expected shape: ({batch_size}, 768)")
    except Exception as e:
        print(f"  Error in representation extraction: {e}")
        return False
    
    # Test with different batch sizes
    print("\n4. Testing with different batch sizes:")
    for bs in [1, 4, 8]:
        x_test = torch.randn(bs, channels, time_samples)
        try:
            model_rope.eval()
            with torch.no_grad():
                loss = model_rope(x_test)
            print(f"  Batch size {bs}: Success")
        except Exception as e:
            print(f"  Batch size {bs}: Failed - {e}")
            return False
    
    print("\n✓ All tests passed successfully!")
    return True

def test_rope_components():
    """Test individual RoPE components."""
    
    print("\nTesting individual RoPE components...")
    
    from rope_pos_encoding import RoPE2D, RoPE1D, apply_rotary_pos_emb
    
    # Test 2D RoPE
    print("\n1. Testing 2D RoPE:")
    rope_2d = RoPE2D(
        embed_dim=768,
        num_heads=16,
        base=10000.0,
        shift_x=0.1
    )
    
    H, W = 8, 50  # 8 leads, 50 time patches
    sin, cos = rope_2d(H, W)
    print(f"  2D RoPE output shapes: sin={sin.shape}, cos={cos.shape}")
    print(f"  Expected shape: ({H*W}, 768)")
    assert sin.shape == (H*W, 768) and cos.shape == (H*W, 768)
    
    # Test 1D RoPE
    print("\n2. Testing 1D RoPE:")
    rope_1d = RoPE1D(
        embed_dim=768,
        num_heads=16,
        base=10000.0
    )
    
    L = 50  # 50 time patches
    sin, cos = rope_1d(L)
    print(f"  1D RoPE output shapes: sin={sin.shape}, cos={cos.shape}")
    print(f"  Expected shape: ({L}, 768)")
    assert sin.shape == (L, 768) and cos.shape == (L, 768)
    
    # Test apply_rotary_pos_emb
    print("\n3. Testing apply_rotary_pos_emb:")
    B, N, H, D = 2, 50, 16, 48  # batch, seq_len, heads, head_dim (embed_dim=768=16*48)
    x = torch.randn(B, N, H, D)
    sin_batch = sin[:N].unsqueeze(0).expand(B, -1, -1)
    cos_batch = cos[:N].unsqueeze(0).expand(B, -1, -1)
    
    try:
        x_rot = apply_rotary_pos_emb(x, sin_batch, cos_batch)
        print(f"  Rotary embedding applied successfully!")
        print(f"  Input shape: {x.shape}, Output shape: {x_rot.shape}")
        assert x_rot.shape == x.shape
    except Exception as e:
        print(f"  Error applying rotary embedding: {e}")
        return False
    
    print("\n✓ All component tests passed!")
    return True

if __name__ == "__main__":
    # Run component tests first
    if test_rope_components():
        # Then run full model tests
        test_rope_implementation()
    else:
        print("\nComponent tests failed. Skipping full model tests.")