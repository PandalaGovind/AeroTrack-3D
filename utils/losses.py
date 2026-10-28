import torch
import torch.nn as nn

class SparseFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(SparseFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = 1e-5  # Safely within float32 precision limits

    def forward(self, predictions, targets):
        # Clamp to prevent log(0) using a safe float32 epsilon
        predictions = torch.clamp(predictions, self.eps, 1.0 - self.eps)
        
        bce_pos = -targets * torch.log(predictions)
        bce_neg = -(1.0 - targets) * torch.log(1.0 - predictions)
        
        focal_pos = self.alpha * torch.pow(1.0 - predictions, self.gamma) * bce_pos
        focal_neg = (1.0 - self.alpha) * torch.pow(predictions, self.gamma) * bce_neg
        
        loss = focal_pos + focal_neg
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class CIoULoss(nn.Module):
    """
    Complete IoU (CIoU) Loss for anchor-free target bounding box regression.
    Evaluates overlap, center-point distance, and aspect ratio discrepancies.
    """
    def __init__(self, eps=1e-7):
        super(CIoULoss, self).__init__()
        self.eps = eps

    def forward(self, pred_boxes, target_boxes):
        """
        Args:
            pred_boxes (Tensor): [N_pos, 4] -> (x_center, y_center, w, h)
            target_boxes (Tensor): [N_pos, 4] -> (x_center, y_center, w, h)
        """
        if pred_boxes.shape[0] == 0:
            return torch.tensor(0.0, device=pred_boxes.device, requires_grad=True)

        # Convert center (x, y, w, h) to corner (x1, y1, x2, y2)
        pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2.0
        pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2.0
        pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2.0
        pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2.0

        target_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2.0
        target_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2.0
        target_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2.0
        target_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2.0

        # Intersection area
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)

        inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
        inter_area = inter_w * inter_h

        # Union area
        pred_area = pred_boxes[:, 2] * pred_boxes[:, 3]
        target_area = target_boxes[:, 2] * target_boxes[:, 3]
        union_area = pred_area + target_area - inter_area + self.eps

        iou = inter_area / union_area

        # Distance penalty (center distance / enclosing diagonal)
        center_dist = (pred_boxes[:, 0] - target_boxes[:, 0]) ** 2 + (pred_boxes[:, 1] - target_boxes[:, 1]) ** 2

        enc_x1 = torch.min(pred_x1, target_x1)
        enc_y1 = torch.min(pred_y1, target_y1)
        enc_x2 = torch.max(pred_x2, target_x2)
        enc_y2 = torch.max(pred_y2, target_y2)
        enclose_diag = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + self.eps

        # Aspect ratio consistency penalty
        v = (4.0 / (torch.pi ** 2)) * torch.pow(
            torch.atan(target_boxes[:, 2] / (target_boxes[:, 3] + self.eps)) -
            torch.atan(pred_boxes[:, 2] / (pred_boxes[:, 3] + self.eps)), 2
        )
        
        with torch.no_grad():
            alpha = v / (1.0 - iou + v + self.eps)

        ciou = iou - (center_dist / enclose_diag) - (alpha * v)
        return (1.0 - ciou).mean()


class PISTTNLoss(nn.Module):
    """
    Combined Loss: L_total = c1 * L_ss + c2 * L_iou
    Default Paper Weights: c1 = 0.1 (Focal Loss), c2 = 2.0 (CIoU Loss)
    """
    def __init__(self, c1=0.1, c2=2.0):
        super(PISTTNLoss, self).__init__()
        self.c1 = c1
        self.c2 = c2
        self.focal = SparseFocalLoss()
        self.ciou = CIoULoss()

    def forward(self, pred_cls, target_cls, pred_boxes=None, target_boxes=None):
        loss_ss = self.focal(pred_cls, target_cls)
        
        # Calculate CIoU loss exclusively on positive target candidates
        if pred_boxes is not None and target_boxes is not None:
            pos_mask = (target_cls.squeeze(-1) == 1.0)
            if pos_mask.sum() > 0:
                loss_iou = self.ciou(pred_boxes[pos_mask], target_boxes[pos_mask])
            else:
                loss_iou = torch.tensor(0.0, device=pred_cls.device, requires_grad=True)
        else:
            loss_iou = torch.tensor(0.0, device=pred_cls.device, requires_grad=True)
            
        total_loss = (self.c1 * loss_ss) + (self.c2 * loss_iou)
        return total_loss, loss_ss, loss_iou