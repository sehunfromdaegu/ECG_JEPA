import torch
import torch.nn as nn
from ecg_jepa import ecg_jepa
from gram_loss import GramLoss
import numpy as np

def test_gram_anchoring():
    """Test gram anchoring implementation"""
    
    print("Testing Gram Anchoring for ECG-JEPA...")
    print("-" * 50)
    
    # Initialize model with gram anchoring
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
        pos_type='rope',
        c=8,
        p=50,
        t=50,
        use_gram=True,
        gram_loss_weight=1.0
    ).cuda()
    
    # Create dummy ECG data
    batch_size = 4
    c = 8  # leads
    T = 2500  # samples (50 patches * 50 time points)
    dummy_ecg = torch.randn(batch_size, c, T).cuda()
    
    print(f"Model initialized with gram anchoring: {model.use_gram}")
    print(f"Gram teacher initialized: {model.gram_teacher_initialized}")
    
    # Test 1: Forward pass without gram teacher (should work)
    print("\nTest 1: Forward pass without gram teacher...")
    loss1 = model(dummy_ecg)
    print(f"Loss without gram teacher: {loss1.item():.4f}")
    assert loss1.item() > 0, "Loss should be positive"
    
    # Test 2: Initialize gram teacher
    print("\nTest 2: Initializing gram teacher...")
    model.initialize_gram_teacher()
    print(f"Gram teacher initialized: {model.gram_teacher_initialized}")
    assert model.gram_teacher_initialized, "Gram teacher should be initialized"
    
    # Test 3: Forward pass with gram teacher
    print("\nTest 3: Forward pass with gram teacher...")
    loss2, gram_loss, features = model(dummy_ecg, return_features=True)
    print(f"Total loss with gram: {loss2.item():.4f}")
    print(f"Gram loss component: {gram_loss.item():.4f}")
    assert gram_loss.item() >= 0, "Gram loss should be non-negative"
    
    # Test 4: Update gram teacher
    print("\nTest 4: Updating gram teacher...")
    # Store old parameters
    old_param = next(model.gram_teacher.parameters()).clone()
    model.update_gram_teacher(momentum=0.5)
    new_param = next(model.gram_teacher.parameters())
    param_changed = not torch.allclose(old_param, new_param)
    print(f"Gram teacher parameters updated: {param_changed}")
    
    # Test 5: Test gram loss directly
    print("\nTest 5: Testing GramLoss module...")
    gram_loss_fn = GramLoss(apply_norm=True, img_level=True)
    student_feats = torch.randn(batch_size, 400, 768).cuda()  # (B, N, D)
    teacher_feats = torch.randn(batch_size, 400, 768).cuda()
    gram_loss_direct = gram_loss_fn(student_feats, teacher_feats)
    print(f"Direct gram loss: {gram_loss_direct.item():.4f}")
    
    # Test 6: Verify no gradients flow through gram teacher
    print("\nTest 6: Verifying no gradients in gram teacher...")
    for param in model.gram_teacher.parameters():
        assert not param.requires_grad, "Gram teacher should not require gradients"
    print("✓ Gram teacher parameters are frozen")
    
    print("\n" + "=" * 50)
    print("All tests passed! Gram anchoring is working correctly.")
    print("=" * 50)
    
    # Print recommended usage
    print("\nRecommended usage for training:")
    print("python pretrain_ECG_JEPA.py --use_gram --gram_start_epoch 50 --gram_update_freq 10 --gram_loss_weight 1.0")
    print("\nUpdate frequency suggestions based on total epochs:")
    print("- 100 epochs: update every 10 epochs (5 updates total)")
    print("- 200 epochs: update every 15-20 epochs (7-10 updates total)")
    print("- 300 epochs: update every 25 epochs (10 updates total)")

if __name__ == "__main__":
    test_gram_anchoring()