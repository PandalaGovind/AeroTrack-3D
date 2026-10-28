import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

class TargetTracker:
    def __init__(self, prob_threshold=0.5, nms_radius=5.0, max_track_dist=15.0, max_age=1):
        self.prob_threshold = prob_threshold
        self.nms_radius = nms_radius
        self.max_track_dist = max_track_dist
        self.max_age = max_age  # Frames to remember a lost track
        self.next_track_id = 0

    def extract_targets(self, coords, probabilities, bbox_preds=None):
        mask = (probabilities > self.prob_threshold).squeeze(-1)
        valid_coords = coords[mask].float()
        valid_probs = probabilities[mask].squeeze(-1)
        valid_boxes = bbox_preds[mask] if bbox_preds is not None else None

        if len(valid_coords) == 0:
            return []

        # Sort detections descending by target probability
        sorted_indices = torch.argsort(valid_probs, descending=True)
        valid_coords = valid_coords[sorted_indices]
        valid_probs = valid_probs[sorted_indices]
        if valid_boxes is not None:
            valid_boxes = valid_boxes[sorted_indices]

        # Pre-compute true centers for accurate NMS
        if valid_boxes is not None:
            centers_x = valid_coords[:, 3] + valid_boxes[:, 0]
            centers_y = valid_coords[:, 2] + valid_boxes[:, 1]
            widths = valid_boxes[:, 2].clamp(min=1.0)
            heights = valid_boxes[:, 3].clamp(min=1.0)
        else:
            centers_x = valid_coords[:, 3]
            centers_y = valid_coords[:, 2]
            widths = torch.full((len(valid_coords),), 4.0)
            heights = torch.full((len(valid_coords),), 4.0)

        detections = []
        keep = torch.ones(len(valid_coords), dtype=torch.bool)

        for i in range(len(valid_coords)):
            if not keep[i]:
                continue
                
            b_idx = int(valid_coords[i, 0].item())
            t_idx = int(valid_coords[i, 1].item())

            detections.append({
                'batch_idx': b_idx,
                'time': t_idx,
                'x': centers_x[i].item(),
                'y': centers_y[i].item(),
                'w': widths[i].item(),
                'h': heights[i].item(),
                'prob': valid_probs[i].item(),
                'track_id': None
            })
            
            if i + 1 < len(valid_coords):
                same_frame = (valid_coords[i+1:, 0] == b_idx) & (valid_coords[i+1:, 1] == t_idx)
                dist = torch.sqrt(
                    (centers_y[i+1:] - centers_y[i]) ** 2 + 
                    (centers_x[i+1:] - centers_x[i]) ** 2
                )
                suppress = same_frame & (dist < self.nms_radius)
                
                # FIXED: Correct PyTorch boolean updating
                suppress_indices = torch.where(suppress)[0] + (i + 1)
                keep[suppress_indices] = False

        return detections

    def link_trajectories(self, detections):
        if not detections:
            return []

        frames_dict = {}
        for det in detections:
            key = (det['batch_idx'], det['time'])
            frames_dict.setdefault(key, []).append(det)

        batch_indices = set(det['batch_idx'] for det in detections)

        for b_idx in batch_indices:
            times = sorted(set(det['time'] for det in detections if det['batch_idx'] == b_idx))
            active_tracks = []  # List of dicts: {'track_id', 'x', 'y', 'age'}

            for t in times:
                current_dets = frames_dict.get((b_idx, t), [])
                
                if not active_tracks:
                    for det in current_dets:
                        det['track_id'] = self.next_track_id
                        active_tracks.append({'track_id': self.next_track_id, 'x': det['x'], 'y': det['y'], 'age': 0})
                        self.next_track_id += 1
                    continue

                if current_dets:
                    # Build Cost Matrix for Hungarian Algorithm
                    cost_matrix = np.zeros((len(active_tracks), len(current_dets)))
                    for trk_idx, track in enumerate(active_tracks):
                        for det_idx, det in enumerate(current_dets):
                            cost_matrix[trk_idx, det_idx] = np.hypot(det['x'] - track['x'], det['y'] - track['y'])

                    # Solve optimal bipartite matching
                    trk_indices, det_indices = linear_sum_assignment(cost_matrix)

                    matched_det_indices = set()
                    matched_trk_indices = set()

                    for trk_idx, det_idx in zip(trk_indices, det_indices):
                        if cost_matrix[trk_idx, det_idx] <= self.max_track_dist:
                            current_dets[det_idx]['track_id'] = active_tracks[trk_idx]['track_id']
                            active_tracks[trk_idx].update({
                                'x': current_dets[det_idx]['x'],
                                'y': current_dets[det_idx]['y'],
                                'age': 0
                            })
                            matched_det_indices.add(det_idx)
                            matched_trk_indices.add(trk_idx)

                    # Initialize new tracks for unmatched detections
                    for det_idx, det in enumerate(current_dets):
                        if det_idx not in matched_det_indices:
                            det['track_id'] = self.next_track_id
                            active_tracks.append({'track_id': self.next_track_id, 'x': det['x'], 'y': det['y'], 'age': 0})
                            self.next_track_id += 1

                # Update ages for unmatched tracks (Track Memory)
                updated_active_tracks = []
                for trk_idx, track in enumerate(active_tracks):
                    if current_dets and trk_idx not in matched_trk_indices:
                        track['age'] += 1
                    
                    if track['age'] <= self.max_age:
                        updated_active_tracks.append(track)

                active_tracks = updated_active_tracks

        return detections