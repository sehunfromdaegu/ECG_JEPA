import torch
from ecg_jepa import ecg_jepa

# Test with small batch to debug
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
    t=50
).to('cuda')

# Test with batch size 128 (same as pretrain script)
x = torch.randn(128, 8, 2500).cuda()

try:
    model.train()
    loss = model(x)
    print(f"Success! Loss: {loss.item()}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()