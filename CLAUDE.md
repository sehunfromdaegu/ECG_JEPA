# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ECG-JEPA is a self-supervised learning framework for 12-lead ECG signal analysis using a Joint-Embedding Predictive Architecture (JEPA). The model learns representations by predicting masked ECG patches in a learned representation space.

## Common Development Commands

### Environment Setup
```bash
# Create conda environment
conda create --name ecg_jepa python=3.9
conda activate ecg_jepa

# Install PyTorch with CUDA
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge

# Install dependencies
pip install -r requirements.txt
```

### Pretraining
```bash
# Random masking strategy
python pretrain_ECG_JEPA.py --mask_type random --mask_scale 0.6 0.7 --batch_size 128 --lr 2.5e-5

# Multi-block masking strategy
python pretrain_ECG_JEPA.py --mask_type block --mask_scale 0.175 0.225 --batch_size 64 --lr 5.5e-5
```

### Downstream Evaluation
```bash
# Linear evaluation on PTB-XL
cd downstream_tasks
python linear_eval.py --ckpt_dir ../weights/multiblock_epoch100.pth --dataset ptbxl --task multilabel

# Fine-tuning on PTB-XL
python finetuning.py --ckpt_dir ../weights/multiblock_epoch100.pth --dataset ptbxl --task multiclass
```

### Testing
```bash
# Run performance metrics evaluation
python perf_metrics.py
```

## Architecture Overview

### Core Model Structure
The JEPA model consists of three main components:

1. **Context Encoder** (`ecg_jepa.py`): Processes visible ECG patches through a 12-layer transformer with cross-attention masking between leads and time
2. **Target Encoder**: EMA-updated copy of context encoder that processes all patches
3. **Predictor Network**: 6-layer transformer that predicts target representations from context

### Key Design Decisions
- **Input Processing**: 12-lead ECG → 8-lead (I, II, V1-V6) → reshape to 8×50 patches
- **Masking**: Supports random (60-70%) and multi-block (17.5-22.5%) masking strategies
- **Cross-Attention**: Custom attention pattern allowing within-lead temporal and across-lead spatial attention
- **EMA Updates**: Target encoder updated with momentum scheduling (0.996 → 1.0)

### Data Flow
```
ecg_data.py → ECG loading/preprocessing → 8×2500 samples
    ↓
augmentation.py → Data augmentation pipeline
    ↓
ecg_jepa.py → JEPA model training
    ↓
downstream_tasks/*.py → Linear eval or fine-tuning
```

## Key Files and Their Purposes

- **`ecg_jepa.py`**: Core JEPA model implementation with encoder, predictor, and training logic
- **`pretrain_ECG_JEPA.py`**: Main pretraining script with distributed training support
- **`ecg_data.py`**: Dataset classes for PTB-XL, CPSC2018, Shaoxing, and Code15 datasets
- **`augmentation.py`**: ECG-specific augmentation strategies (filtering, masking, noise)
- **`models.py`**: Model loading utilities and encoder extraction
- **`pos_encoding.py`**: 2D sinusoidal positional encoding for spatial-temporal representation
- **`downstream_tasks/linear_eval.py`**: Linear probing evaluation
- **`downstream_tasks/finetuning.py`**: Full model fine-tuning
- **`ptbxl_utils.py`**: PTB-XL specific utilities and evaluation metrics

## Important Implementation Details

### ECG Signal Processing
- All ECGs resampled to 500Hz, 2500 samples (10 seconds)
- Lead reduction: 12-lead → 8-lead (dropping III, aVR, aVL, aVF)
- Patch creation: 50-sample patches, resulting in 8×50 patch grid
- Data cleaning: NaN removal and zero-padding detection

### Model Parameters
- Encoder: 768 dim, 12 depth, 16 heads (~95M parameters)
- Predictor: 384 dim, 6 depth, 12 heads
- Batch sizes: 128 (random masking), 64 (multi-block masking)
- Learning rates: 2.5e-5 (random), 5.5e-5 (multi-block)

### Dataset Configuration
- PTB-XL: Located at `./data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/`
- CPSC2018: Located at `./data/CPSC/`
- Shaoxing: Located at `./data/shaoxing/`
- Code15: HDF5 file at `./data/code15/`

### Pretrained Weights
Download pretrained models and place in `./weights/`:
- Random masking: `random_epoch100.pth`
- Multi-block masking: `multiblock_epoch100.pth`