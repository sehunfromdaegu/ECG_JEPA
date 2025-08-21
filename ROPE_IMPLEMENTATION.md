# RoPE Implementation for ECG-JEPA

## Overview
This document describes the Rotary Position Embedding (RoPE) implementation for ECG-JEPA, which follows the DINOv3 architecture closely.

## Key Design Decisions (Following DINOv3)

### 1. RoPE Application Pattern
- **RoPE is applied in EVERY transformer block** - This matches DINOv3's approach where positional information is refreshed at each layer
- **RoPE is generated ONCE per forward pass** - The embeddings are computed once and passed through all blocks for efficiency
- **RoPE is applied in the attention mechanism** - Applied to Q and K matrices before computing attention scores

### 2. Implementation Details

#### Generation (Once per forward pass):
```python
# In MaskTransformer.forward():
if self.use_rope:
    # Generate RoPE embeddings once
    rope_sin, rope_cos = self.rope_2d(self.c, self.p)  # For 8x50 patches
    rope_sin = rope_sin.unsqueeze(0).expand(bs, -1, -1)
    rope_cos = rope_cos.unsqueeze(0).expand(bs, -1, -1)
```

#### Application (In every block's attention):
```python
# In Attention.forward():
if self.use_rope and rope_sin is not None and rope_cos is not None:
    q = apply_rotary_pos_emb(q, rope_sin, rope_cos)
    k = apply_rotary_pos_emb(k, rope_sin, rope_cos)
```

### 3. ECG-Specific Adaptations

#### 2D RoPE for Main Encoder:
- Treats 8×50 patches as 2D grid (H=8 leads, W=50 time patches)
- **X-axis only coordinate shift during pretraining**: Only time dimension is augmented with random shifts (±0.1)
- Y-axis (lead dimension) remains fixed to preserve ECG lead ordering

#### 1D RoPE for Target Encoder:
- Since target encoder processes single leads, uses 1D RoPE for 50 time patches
- Consistent with the masked prediction objective operating on individual leads

### 4. Coordinate Shift Strategy

Following DINOv3's data augmentation approach:
```python
# Only shift x-axis (time dimension) during training
if self.training and self.shift_x is not None:
    shift_x = torch.empty(1, **dd).uniform_(-self.shift_x, self.shift_x)
    coords[:, 1] += shift_x  # Only shift the W (time) coordinate
```

This provides temporal augmentation without disrupting the spatial structure of ECG leads.

## Architecture Comparison

### DINOv3 Pattern:
```python
# Generate once
rope_sincos = self.rope_embed(H=H, W=W)
# Apply in every block
for blk in self.blocks:
    x = blk(x, rope_sincos)
```

### ECG-JEPA Pattern (Same as DINOv3):
```python
# Generate once
rope_sin, rope_cos = self.rope_2d(self.c, self.p)
# Apply in every block
for block in self.blocks:
    x = block(x, attention_mask, rope_sin, rope_cos)
```

## Performance Benefits

1. **Better generalization**: RoPE provides continuous position encoding that generalizes to different sequence lengths
2. **Temporal augmentation**: X-axis shift provides data augmentation in time domain
3. **Preserved structure**: Fixed y-axis maintains ECG lead relationships
4. **Efficiency**: Generated once, used many times (following DINOv3)

## Files Modified

1. **rope_pos_encoding.py**: Complete RoPE implementation
   - `RoPE2D`: 2D rotary embeddings for main encoder
   - `RoPE1D`: 1D rotary embeddings for target encoder
   - `apply_rotary_pos_emb`: Efficient RoPE application (DINOv3 style)

2. **ecg_jepa.py**: Integration into transformer blocks
   - Updated `Attention`, `Block`, `Encoder_Block`, `Predictor_Block`
   - Modified `MaskTransformer` and `MaskTransformerPredictor`
   - Changed default `pos_type` from 'sincos' to 'rope'

## Usage

To use RoPE in ECG-JEPA:
```python
model = ecg_jepa(
    encoder_embed_dim=768,
    encoder_depth=12,
    encoder_num_heads=16,
    pos_type='rope',  # Enable RoPE (now default)
    # ... other parameters
)
```

To use legacy sincos embeddings:
```python
model = ecg_jepa(
    pos_type='sincos',  # Use original sincos embeddings
    # ... other parameters
)
```

## Testing

The implementation has been tested with:
- Various batch sizes (2, 32, 128)
- Pretraining with random and block masking
- Forward and backward passes
- Mixed precision training

All tests pass successfully with expected loss convergence.