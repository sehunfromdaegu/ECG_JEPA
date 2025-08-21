import torch
import torch.nn as nn
import torch.nn.functional as F


class GramLoss(nn.Module):
    """Gram matrix loss for ECG-JEPA, adapted from DINOv3"""

    def __init__(
        self,
        apply_norm=True,
        img_level=True,
        remove_neg=True,
        remove_only_teacher_neg=False,
    ):
        super().__init__()

        # Loss
        self.mse_loss = torch.nn.MSELoss()

        # Parameters
        self.apply_norm = apply_norm
        self.remove_neg = remove_neg
        self.remove_only_teacher_neg = remove_only_teacher_neg

        if self.remove_neg or self.remove_only_teacher_neg:
            assert self.remove_neg != self.remove_only_teacher_neg

    def forward(self, student_feats, teacher_feats, img_level=True):
        """Compute the MSE loss between the gram matrix of student and teacher features.

        Args:
            student_feats: Pytorch tensor (B, N, dim) or (B*N, dim) if img_level == False
            teacher_feats: Pytorch tensor (B, N, dim) or (B*N, dim) if img_level == False
            img_level: bool, if true gram computed at the sample level only else over the entire batch
        Returns:
            loss: scalar
        """

        # Dimensions of the tensor should be (B, N, dim)
        if img_level:
            assert len(teacher_feats.shape) == 3 and len(student_feats.shape) == 3

        # Float casting
        student_feats = student_feats.float()
        teacher_feats = teacher_feats.float()

        # Teacher correlation
        if self.apply_norm:
            teacher_feats = F.normalize(teacher_feats, dim=-1)

        if not img_level and len(teacher_feats.shape) == 3:
            # Flatten (B, N, D) into  (B*N, D)
            teacher_feats = teacher_feats.flatten(0, 1)

        # Compute similarities
        teacher_sim = torch.matmul(teacher_feats, teacher_feats.transpose(-1, -2))

        # Student correlation
        if self.apply_norm:
            student_feats = F.normalize(student_feats, dim=-1)

        if not img_level and len(student_feats.shape) == 3:
            # Flatten (B, N, D) into  (B*N, D)
            student_feats = student_feats.flatten(0, 1)

        # Compute similarities
        student_sim = torch.matmul(student_feats, student_feats.transpose(-1, -2))

        if self.remove_neg:
            teacher_sim[teacher_sim < 0] = 0.0
            student_sim[student_sim < 0] = 0.0

        elif self.remove_only_teacher_neg:
            # Remove only the negative sim values of the teacher
            teacher_sim[teacher_sim < 0] = 0.0
            student_sim[(student_sim < 0) & (teacher_sim < 0)] = 0.0

        return self.mse_loss(student_sim, teacher_sim)


def linear_warmup_cosine_decay(start, peak, end, warmup_iters, total_iters):
    """Create a schedule that linearly warms up and then cosine decays"""
    schedule = []
    for i in range(total_iters):
        if i < warmup_iters:
            # Linear warmup
            value = start + (peak - start) * i / warmup_iters
        else:
            # Cosine decay
            progress = (i - warmup_iters) / (total_iters - warmup_iters)
            value = end + (peak - end) * (1 + torch.cos(torch.tensor(progress * 3.14159))) / 2
        schedule.append(value)
    return schedule