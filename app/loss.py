import torch
import torch.nn as nn

# Cross Entropy Loss for segmentation (if segmentation output is multi-class)
segmentation_loss_fn = nn.CrossEntropyLoss()

# Cross Entropy Loss for classification
classification_loss_fn = nn.CrossEntropyLoss()


def combined_loss(seg_output, class_output, target_segmentation, target_class, alpha=1.0, beta=1.0):
    """
    Compute combined loss for segmentation and classification.

    Parameters:
    - seg_output: Output from the segmentation head (shape: [batch_size, n_segmentation_classes, H, W])
    - class_output: Output from the classification head (shape: [batch_size, n_classes])
    - target_segmentation: Ground truth segmentation map (shape: [batch_size, H, W])
    - target_class: Ground truth class labels (shape: [batch_size])
    - alpha: Weight for the segmentation loss
    - beta: Weight for the classification loss

    Returns:
    - total_loss: Combined loss from both segmentation and classification tasks
    """

    # 1. Segmentation loss
    # For binary segmentation, you can use Dice Loss or combine it with Cross Entropy
    if seg_output.size(1) == 1:  # Binary segmentation
        seg_loss = dice_loss(seg_output, target_segmentation)  # Use dice loss for binary segmentation
    else:  # Multi-class segmentation
        seg_loss = segmentation_loss_fn(seg_output,
                                        target_segmentation)  # Use cross-entropy loss for multi-class segmentation

    # 2. Classification loss
    class_loss = classification_loss_fn(class_output, target_class)

    # 3. Combine the losses
    total_loss = alpha * seg_loss + beta * class_loss

    return total_loss, seg_loss, class_loss
