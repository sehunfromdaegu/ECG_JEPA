import os
import logging
from datetime import datetime
import torch
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from scipy.signal import resample
import time
from torch.cuda.amp import GradScaler, autocast
from ecg_jepa import ecg_jepa
from timm.scheduler import CosineLRScheduler
from ecg_data import *
import argparse

def downsample_waves(waves, new_size):
    return np.array([resample(wave, new_size, axis=1) for wave in waves])

# Argument parser
parser = argparse.ArgumentParser(description="Pretrain the JEPA model with ECG data")
parser.add_argument('--mask_scale', type=float, nargs=2, default=(0.7, 0.8), help="Scale of masking")
parser.add_argument('--batch_size', type=int, default=192, help="Batch size")
parser.add_argument('--lr', type=float, default=3.75e-5, help="Learning rate")
parser.add_argument('--mask_type', type=str, default='random', help="Type of masking") # 'block' or 'random'
parser.add_argument('--epochs', type=int, default=100, help="Number of epochs")
parser.add_argument('--wd', type=float, default=0.05, help="Weight decay")
# parser.add_argument('--data_dir_shao', type=str, default='/mount/ecg/physionet.org/files/ecg-arrhythmia/1.0.0/WFDBRecords/', help="Directory for Shaoxing data")
parser.add_argument('--data_dir_shao', type=str, default='/mount/ecg/WFDB_ShaoxingUniv/', help="Directory for Shaoxing data")
parser.add_argument('--data_dir_code15', type=str, default='/mount/ecg/code15', help="Directory for Code15 data")
# Gram anchoring arguments
parser.add_argument('--use_gram', action='store_true', default=False, help="Enable gram anchoring")
parser.add_argument('--gram_start_epoch', type=int, default=50, help="Epoch to start gram anchoring")
parser.add_argument('--gram_update_freq', type=int, default=1, help="Update gram teacher every N epochs")
parser.add_argument('--gram_loss_weight', type=float, default=1.0, help="Weight for gram loss")
parser.add_argument('--gram_loss_schedule', action='store_true', help="Use cosine schedule for gram loss weight")

args = parser.parse_args()

# Access the arguments like this
mask_scale = tuple(args.mask_scale)
batch_size = args.batch_size
lr = args.lr
mask_type = args.mask_type
epochs = args.epochs
wd = args.wd
data_dir_shao = args.data_dir_shao
data_dir_code15 = args.data_dir_code15
use_gram = args.use_gram
gram_start_epoch = args.gram_start_epoch
gram_update_freq = args.gram_update_freq
gram_loss_weight = args.gram_loss_weight
gram_loss_schedule = args.gram_loss_schedule

# Generate timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Create logs directory if it doesn't exist
save_dir = f'./weights/ecg_jepa_{timestamp}_{mask_scale}'
os.makedirs(save_dir, exist_ok=True)
log_file = os.path.join(save_dir, f'training_{timestamp}.log')

# Configure logging
logging.basicConfig(filename=log_file, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def log_params(params_dict):
    for key, value in params_dict.items():
        logging.info(f'{key}: {value}')
        print(f'{key}: {value}')  
        
os.makedirs(save_dir, exist_ok=True)

start_time = time.time()

# Shaoxing (Ningbo + Chapman)
waves_shaoxing = waves_shao(data_dir_shao)
waves_shaoxing = downsample_waves(waves_shaoxing, 2500)
# waves_shaoxing = np.random.randn(5000, 8, 2500)
print(f'Shao waves shape: {waves_shaoxing.shape}')
logging.info(f'Shao waves shape: {waves_shaoxing.shape}')

dataset = ECGDataset_pretrain(waves_shaoxing)

# Code15
dataset_code15 = Code15Dataset(data_dir_code15)
print(f'Code15 waves shape: ({len(dataset_code15.file_indices)}, 8, 2500)')
logging.info(f'Code15 waves shape: ({len(dataset_code15.file_indices)}, 8, 2500)')

loading_time = time.time() - start_time
print(f'Data loading time: {loading_time:.2f}s')

dataset = ConcatDataset([dataset, dataset_code15])
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16)
del waves_shaoxing

model = ecg_jepa(encoder_embed_dim=768, 
                encoder_depth=12, 
                encoder_num_heads=16,
                predictor_embed_dim=384,
                predictor_depth=6,
                predictor_num_heads=12,
                drop_path_rate=0.1,
                mask_scale=mask_scale,
                mask_type=mask_type,
                pos_type='rope',
                c=8,
                p=50,
                t=50,
                use_gram=use_gram,
                gram_loss_weight=gram_loss_weight).to('cuda')


total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params_million = total_params / 1000000
logging.info(f'Total number of learnable parameters: {total_params_million:.2f} million')

param_groups = [{'params': (p for n, p in model.named_parameters() if (p.requires_grad) and ('bias' not in n) and (len(p.shape) != 1))},
                {'params': (p for n, p in model.named_parameters() if (p.requires_grad) and (('bias' in n) or (len(p.shape) == 1))), 
                'WD_exclude': True, 
                'weight_decay': 0}]

iterations_per_epoch = len(train_loader)
optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=wd)
scheduler = CosineLRScheduler(optimizer,
        t_initial=iterations_per_epoch*epochs,
        cycle_mul=1,
        lr_min=1e-6,
        cycle_decay=0.1,
        warmup_lr_init=1e-6,
        warmup_t=5*iterations_per_epoch,
        cycle_limit=1,
        t_in_epochs=True)

ema = [0.996,1.0]
momentum_target_encoder_scheduler = (ema[0] + i*(ema[1]-ema[0])/(iterations_per_epoch*epochs) for i in range(int(iterations_per_epoch*epochs)+1))

# Log hyperparameters and model arguments
hyperparameters = vars(args)
log_params(hyperparameters)

# Create gram loss weight schedule if enabled
if use_gram and gram_loss_schedule:
    from gram_loss import linear_warmup_cosine_decay
    # Schedule: 0 until gram_start_epoch, then warmup to gram_loss_weight over 10 epochs, then cosine decay
    warmup_epochs = 10
    gram_weight_schedule = [0.0] * (gram_start_epoch * iterations_per_epoch)
    remaining_iterations = (epochs - gram_start_epoch) * iterations_per_epoch
    if remaining_iterations > 0:
        schedule_values = linear_warmup_cosine_decay(
            start=0.0,
            peak=gram_loss_weight,
            end=gram_loss_weight * 0.5,  # Decay to half the peak value
            warmup_iters=warmup_epochs * iterations_per_epoch,
            total_iters=remaining_iterations
        )
        gram_weight_schedule.extend(schedule_values)
    logging.info(f'Gram loss weight schedule enabled: warmup from epoch {gram_start_epoch} to {gram_start_epoch + warmup_epochs}')
else:
    gram_weight_schedule = None

scaler = GradScaler()

for epoch in range(epochs):
    start_time = time.time()
    model.train()
    total_loss = 0.
    total_gram_loss = 0.
    
    # Initialize gram teacher at specified epoch
    if use_gram and epoch == gram_start_epoch and not model.gram_teacher_initialized:
        model.initialize_gram_teacher()
        logging.info(f'Gram teacher initialized at epoch {epoch}')
    
    # Update gram teacher periodically
    if use_gram and model.gram_teacher_initialized and epoch > gram_start_epoch:
        if (epoch - gram_start_epoch) % gram_update_freq == 0:
            model.update_gram_teacher(momentum=0.0)  # Full update from target encoder
            logging.info(f'Gram teacher updated at epoch {epoch}')
    
    for minibatch, wave in enumerate(train_loader):
        scheduler.step(epoch * iterations_per_epoch + minibatch)
        bs, c, t = wave.shape
        wave = wave.to('cuda')

        optimizer.zero_grad()
        
        # Update gram loss weight if using schedule
        if gram_weight_schedule is not None:
            current_iteration = epoch * iterations_per_epoch + minibatch
            if current_iteration < len(gram_weight_schedule):
                model.gram_loss_weight = gram_weight_schedule[current_iteration]
        
        with autocast():  # Enable mixed precision
            if use_gram and model.gram_teacher_initialized:
                loss, gram_loss, _ = model(wave, return_features=True)
                total_gram_loss += gram_loss.item()
            else:
                loss = model(wave)    

        # Scale the loss and backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()    

        total_loss += loss.item()

        with torch.no_grad():
            m = next(momentum_target_encoder_scheduler)
            for param_q, param_k in zip(model.encoder.parameters(), model.target_encoder.parameters()):
                param_k.data.mul_(m).add_((1.0 - m) * param_q.detach().data)

    total_loss /= len(train_loader)
    epoch_time = time.time() - start_time
    
    if use_gram and model.gram_teacher_initialized:
        total_gram_loss /= len(train_loader)
        print(f'epoch={epoch:04d}/{epochs:04d}  loss={total_loss:.4f}  gram_loss={total_gram_loss:.4f}  time={epoch_time:.2f}s')
        logging.info(f'epoch={epoch:04d}/{epochs:04d}  loss={total_loss:.4f}  gram_loss={total_gram_loss:.4f}  time={epoch_time:.2f}s')
    else:
        print(f'epoch={epoch:04d}/{epochs:04d}  loss={total_loss:.4f}  time={epoch_time:.2f}s')
        logging.info(f'epoch={epoch:04d}/{epochs:04d}  loss={total_loss:.4f}  time={epoch_time:.2f}s')

    if epoch > 1 and (epoch + 1) % 5 == 0:
        model.to('cpu')
        torch.save({'encoder': model.encoder.state_dict(),
                    'epoch': epoch,
                    }, f'{save_dir}/epoch{epoch + 1}.pth')
        model.to('cuda')
