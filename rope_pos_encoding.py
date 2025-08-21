import math
import numpy as np
import torch
import torch.nn as nn
from typing import Literal, Optional, Tuple


class RoPE2D(nn.Module):
    """2D RoPE for ECG patches (8 leads x 50 time patches).
    
    During pretraining, applies coordinate shift only on x-axis (time dimension).
    The y-axis (lead dimension) remains fixed.
    """
    
    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float = 10000.0,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_x: Optional[float] = None,  # Only shift x-axis
        jitter_x: Optional[float] = None,  # Only jitter x-axis
        rescale_coords: Optional[float] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        assert embed_dim % (4 * num_heads) == 0, f"embed_dim must be divisible by 4*num_heads"
        
        D_head = embed_dim // num_heads
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.base = base
        self.D_head = D_head
        self.normalize_coords = normalize_coords
        self.shift_x = shift_x
        self.jitter_x = jitter_x
        self.rescale_coords = rescale_coords
        self.dtype = dtype
        
        # Initialize periods for frequency encoding
        self.register_buffer(
            "periods",
            torch.empty(D_head // 4, device=device, dtype=dtype),
            persistent=True,
        )
        self._init_weights()
    
    def forward(self, H: int, W: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate RoPE embeddings for H x W grid.
        
        Args:
            H: Height (number of leads = 8)
            W: Width (number of time patches = 50)
            
        Returns:
            sin, cos: Both tensors of shape [H*W, embed_dim]
        """
        device = self.periods.device
        dtype = self.dtype or torch.float32
        dd = {"device": device, "dtype": dtype}
        
        # Prepare coords in range [-1, +1]
        if self.normalize_coords == "max":
            max_HW = max(H, W)
            coords_h = torch.arange(0.5, H, **dd) / max_HW
            coords_w = torch.arange(0.5, W, **dd) / max_HW
        elif self.normalize_coords == "min":
            min_HW = min(H, W)
            coords_h = torch.arange(0.5, H, **dd) / min_HW
            coords_w = torch.arange(0.5, W, **dd) / min_HW
        elif self.normalize_coords == "separate":
            coords_h = torch.arange(0.5, H, **dd) / H
            coords_w = torch.arange(0.5, W, **dd) / W
        else:
            raise ValueError(f"Unknown normalize_coords: {self.normalize_coords}")
        
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)  # [H, W, 2]
        coords = coords.flatten(0, 1)  # [HW, 2]
        coords = 2.0 * coords - 1.0  # Shift range [0, 1] to [-1, +1]
        
        # Apply coordinate shift only to x-axis (time dimension) during training
        if self.training and self.shift_x is not None:
            shift_x = torch.empty(1, **dd).uniform_(-self.shift_x, self.shift_x)
            coords[:, 1] += shift_x  # Only shift the W (time) coordinate
        
        # Apply jitter only to x-axis during training
        if self.training and self.jitter_x is not None:
            jitter_max = np.log(self.jitter_x)
            jitter_min = -jitter_max
            jitter_x = torch.empty(1, **dd).uniform_(jitter_min, jitter_max).exp()
            coords[:, 1] *= jitter_x  # Only jitter the W (time) coordinate
        
        # Apply rescale to both coordinates if specified
        if self.training and self.rescale_coords is not None:
            rescale_max = np.log(self.rescale_coords)
            rescale_min = -rescale_max
            rescale = torch.empty(1, **dd).uniform_(rescale_min, rescale_max).exp()
            coords *= rescale
        
        # Compute angles: [HW, 2, D//4]
        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        angles = angles.flatten(1, 2)  # [HW, D//2]
        # For 2D, we need to tile to match full embed_dim
        # D//2 -> D (since we have sin and cos for each dimension)
        repeat_factor = self.embed_dim // (self.D_head // 2)
        angles = angles.tile(repeat_factor)  # [HW, embed_dim]
        
        cos = torch.cos(angles)  # [HW, D]
        sin = torch.sin(angles)  # [HW, D]
        
        return sin, cos
    
    def _init_weights(self):
        device = self.periods.device
        dtype = self.dtype or torch.float32
        # Initialize frequency periods using base
        periods = self.base ** (
            2 * torch.arange(self.D_head // 4, device=device, dtype=dtype) / (self.D_head // 2)
        )
        self.periods.data = periods


class RoPE1D(nn.Module):
    """1D RoPE for single lead processing (50 time patches).
    
    Used by target encoder when processing individual leads.
    """
    
    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float = 10000.0,
        shift_coords: Optional[float] = None,
        jitter_coords: Optional[float] = None,
        rescale_coords: Optional[float] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        assert embed_dim % (2 * num_heads) == 0, f"embed_dim must be divisible by 2*num_heads for 1D RoPE"
        
        D_head = embed_dim // num_heads
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.base = base
        self.D_head = D_head
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords
        self.dtype = dtype
        
        # Initialize periods for frequency encoding
        self.register_buffer(
            "periods",
            torch.empty(D_head // 2, device=device, dtype=dtype),
            persistent=True,
        )
        self._init_weights()
    
    def forward(self, L: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate 1D RoPE embeddings for sequence of length L.
        
        Args:
            L: Sequence length (number of time patches = 50)
            
        Returns:
            sin, cos: Both tensors of shape [L, embed_dim]
        """
        device = self.periods.device
        dtype = self.dtype or torch.float32
        dd = {"device": device, "dtype": dtype}
        
        # Prepare coords in range [-1, +1]
        coords = torch.arange(0.5, L, **dd) / L  # [L]
        coords = 2.0 * coords - 1.0  # Shift range [0, 1] to [-1, +1]
        
        # Apply coordinate shift during training
        if self.training and self.shift_coords is not None:
            shift = torch.empty(1, **dd).uniform_(-self.shift_coords, self.shift_coords)
            coords += shift
        
        # Apply jitter during training
        if self.training and self.jitter_coords is not None:
            jitter_max = np.log(self.jitter_coords)
            jitter_min = -jitter_max
            jitter = torch.empty(1, **dd).uniform_(jitter_min, jitter_max).exp()
            coords *= jitter
        
        # Apply rescale during training
        if self.training and self.rescale_coords is not None:
            rescale_max = np.log(self.rescale_coords)
            rescale_min = -rescale_max
            rescale = torch.empty(1, **dd).uniform_(rescale_min, rescale_max).exp()
            coords *= rescale
        
        # Compute angles: [L, D//2]
        angles = 2 * math.pi * coords[:, None] / self.periods[None, :]
        # For 1D, we need to tile to match full embed_dim
        repeat_factor = self.embed_dim // (self.D_head // 2)
        angles = angles.tile(repeat_factor)  # [L, embed_dim]
        
        cos = torch.cos(angles)  # [L, D]
        sin = torch.sin(angles)  # [L, D]
        
        return sin, cos
    
    def _init_weights(self):
        device = self.periods.device
        dtype = self.dtype or torch.float32
        # Initialize frequency periods using base
        periods = self.base ** (
            2 * torch.arange(self.D_head // 2, device=device, dtype=dtype) / self.D_head
        )
        self.periods.data = periods


def rope_rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input (following DINOv3).
    
    Args:
        x: Input tensor [..., D]
    
    Returns:
        Rotated tensor where first half becomes -second half, second half becomes first half
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding to input tensor (following DINOv3 implementation).
    
    Args:
        x: Input tensor of shape [B, H, N, D] where H is num_heads, D is head_dim
        sin: Sine embeddings of shape [B, N, D_total] where D_total = H * D
        cos: Cosine embeddings of shape [B, N, D_total]
        
    Returns:
        Tensor with rotary position embedding applied
    """
    B, H, N, D = x.shape
    
    # Reshape sin and cos to match x's dimensions
    sin = sin.view(B, N, H, D).transpose(1, 2)  # [B, H, N, D]
    cos = cos.view(B, N, H, D).transpose(1, 2)  # [B, H, N, D]
    
    # Apply RoPE following DINOv3: (x * cos) + (rotate_half(x) * sin)
    return (x * cos) + (rope_rotate_half(x) * sin)