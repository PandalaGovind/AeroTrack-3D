import torch

class SparseDetectionMetrics:
    """
    Computes sparse target detection metrics (Precision, Recall, F1-Score)
    and confusion matrix counters across training/evaluation epochs.
    """
    def __init__(self, threshold=0.5, eps=1e-7):
        """
        Args:
            threshold (float): Target confidence probability cutoff.
            eps (float): Epsilon for numerical stability against division by zero.
        """
        self.threshold = threshold
        self.eps = eps
        self.reset()

    def reset(self):
        """
        Resets accumulated confusion matrix counters.
        """
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def update(self, pred_probs, targets):
        """
        Updates confusion matrix counters for the current batch.

        Args:
            pred_probs (Tensor): Predicted target probabilities [N, 1]
            targets (Tensor): Ground truth binary target labels [N, 1]
        """
        preds = (pred_probs > self.threshold).float()
        targets = targets.float()

        self.tp += int(torch.sum((preds == 1.0) & (targets == 1.0)).item())
        self.fp += int(torch.sum((preds == 1.0) & (targets == 0.0)).item())
        self.fn += int(torch.sum((preds == 0.0) & (targets == 1.0)).item())
        self.tn += int(torch.sum((preds == 0.0) & (targets == 0.0)).item())

    def compute(self):
        """
        Calculates precision, recall, and F1-score from accumulated counts.

        Returns:
            dict: Evaluated metric scores and total counts.
        """
        precision = self.tp / (self.tp + self.fp + self.eps)
        recall = self.tp / (self.tp + self.fn + self.eps)
        f1_score = 2.0 * (precision * recall) / (precision + recall + self.eps)

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn
        }


class TrackingMetrics:
    """
    Computes spatial tracking location metrics:
    - Precision@5 (percentage of frames where predicted target center is <= 5 pixels from GT)
    - Success Rate (IoU overlap accuracy)
    """
    def __init__(self, dist_threshold=5.0, iou_threshold=0.25):
        self.dist_threshold = dist_threshold
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self):
        self.total_frames = 0
        self.prec_hits = 0
        self.iou_hits = 0

    def update(self, pred_boxes, gt_boxes):
        """
        Args:
            pred_boxes (list[dict]): Extracted target dicts with 'x', 'y', 'w', 'h'.
            gt_boxes (list[dict]): Ground truth boxes with 'x', 'y', 'w', 'h'.
        """
        if not gt_boxes:
            return

        for gt in gt_boxes:
            self.total_frames += 1
            if not pred_boxes:
                continue

            # Find closest prediction
            distances = [
                ((p['x'] - gt['x']) ** 2 + (p['y'] - gt['y']) ** 2) ** 0.5
                for p in pred_boxes
            ]
            min_dist = min(distances)

            if min_dist <= self.dist_threshold:
                self.prec_hits += 1

    def compute(self):
        eps = 1e-7
        precision_at_5 = self.prec_hits / (self.total_frames + eps)
        return {
            "precision_at_5": precision_at_5,
            "total_frames": self.total_frames
        }