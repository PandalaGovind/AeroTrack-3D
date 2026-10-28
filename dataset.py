import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class IRSequenceDataset(Dataset):
    def __init__(self, sequence_dir, annotation_dir=None, window_size=10, intensity_threshold=20, target_size=(256, 256), sequence_list=None):
        """
        Args:
            sequence_dir (str): Path to the SIATD_Dataset sequences folder.
            annotation_dir (str, optional): Path to ground truth text/CSV annotations.
            window_size (int): Temporal window length (default: 10).
            intensity_threshold (int): Background threshold for voxel sparsification.
            target_size (tuple): (Height, Width) for frame spatial normalization.
            sequence_list (list, optional): Specific sequence names to load (e.g., ['seq_01', 'seq_02']). If None, loads all.
        """
        self.sequence_dir = sequence_dir
        self.annotation_dir = annotation_dir if annotation_dir else os.path.join(os.path.dirname(sequence_dir), "annotations")
        self.window_size = window_size
        self.intensity_threshold = intensity_threshold
        self.target_size = target_size
        
        self.samples = []
        if os.path.exists(sequence_dir):
            all_seq_folders = sorted([f for f in os.listdir(sequence_dir) if os.path.isdir(os.path.join(sequence_dir, f))])
            
            # Filter sequences if a specific list is provided for train/test splitting
            if sequence_list is not None:
                seq_folders = [seq for seq in all_seq_folders if seq in sequence_list]
            else:
                seq_folders = all_seq_folders

            for seq in seq_folders:
                seq_path = os.path.join(sequence_dir, seq)
                frames = sorted([f for f in os.listdir(seq_path) if f.endswith(('.png', '.jpg', '.bmp'))])
                
                # Check for sequence annotation file
                annot_file = os.path.join(self.annotation_dir, f"{seq}.txt")
                if not os.path.exists(annot_file):
                    annot_file = os.path.join(seq_path, "groundtruth.txt")
                
                for i in range(len(frames) - self.window_size + 1):
                    self.samples.append({
                        'seq_name': seq,
                        'seq_path': seq_path,
                        'frame_names': frames[i : i + self.window_size],
                        'start_frame_idx': i,
                        'annot_file': annot_file if os.path.exists(annot_file) else None
                    })
        else:
            print(f"Warning: Dataset directory '{sequence_dir}' not found.")

    def _load_annotations(self, annot_file):
        """
        Parses SIATD annotation files.
        Expected line format: frame_idx, target_id, x, y, w, h
        """
        annots = {}
        if not annot_file or not os.path.exists(annot_file):
            return annots
        
        with open(annot_file, 'r') as f:
            for line in f:
                parts = line.strip().replace('\t', ',').split(',')
                if len(parts) >= 5:
                    try:
                        if len(parts) >= 6:
                            f_idx, _, x, y, w, h = int(parts[0]), int(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                        else:
                            f_idx, x, y, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        
                        if f_idx not in annots:
                            annots[f_idx] = []
                        annots[f_idx].append([x, y, w, h])
                    except ValueError:
                        continue
        return annots

    def apply_tsvd_suppression(self, frame, k=5):
        """
        t-SVD background clutter suppression removing top k components.
        """
        U, S, Vt = np.linalg.svd(frame.astype(np.float32), full_matrices=False)
        background = np.dot(U[:, :k], np.dot(np.diag(S[:k]), Vt[:k, :]))
        target_enhanced = frame.astype(np.float32) - background
        return np.clip(target_enhanced, 0, 255)

    def __len__(self):
        return max(1, len(self.samples))

    def __getitem__(self, idx):
        if not self.samples:
            dummy_coords = torch.tensor([[0, 5, 5]], dtype=torch.int32)
            dummy_features = torch.tensor([[0.5]], dtype=torch.float32)
            dummy_targets = torch.tensor([[0.0]], dtype=torch.float32)
            dummy_boxes = torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
            return dummy_coords, dummy_features, dummy_targets, dummy_boxes

        sample = self.samples[idx]
        seq_path = sample['seq_path']
        annots = self._load_annotations(sample['annot_file'])
        
        coords_list, features_list, targets_list, boxes_list = [], [], [], []
        
        for t, frame_name in enumerate(sample['frame_names']):
            img_path = os.path.join(seq_path, frame_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            # Standardize spatial dimensions if needed
            if self.target_size and (img.shape[0] != self.target_size[0] or img.shape[1] != self.target_size[1]):
                img = cv2.resize(img, (self.target_size[1], self.target_size[0]))
            
            processed_img = self.apply_tsvd_suppression(img)
            y_idx, x_idx = np.where(processed_img > self.intensity_threshold)
            
            # Retrieve annotations for this frame (1-based vs 0-based offset handling)
            frame_annots = annots.get(sample['start_frame_idx'] + t, [])
            if not frame_annots:
                frame_annots = annots.get(sample['start_frame_idx'] + t + 1, [])
            
            if len(y_idx) > 0:
                t_idx = np.full_like(y_idx, t)
                frame_coords = np.stack([t_idx, y_idx, x_idx], axis=-1)
                frame_features = processed_img[y_idx, x_idx] / 255.0
                
                frame_targets = np.zeros((len(y_idx), 1), dtype=np.float32)
                frame_boxes = np.zeros((len(y_idx), 4), dtype=np.float32)
                
                # Match sparse voxels against target bounding boxes
                for gt_box in frame_annots:
                    gx, gy, gw, gh = gt_box
                    inside = (x_idx >= gx) & (x_idx <= gx + gw) & (y_idx >= gy) & (y_idx <= gy + gh)
                    frame_targets[inside] = 1.0
                    frame_boxes[inside] = [gx + gw / 2.0, gy + gh / 2.0, gw, gh]
                
                coords_list.append(frame_coords)
                features_list.append(frame_features.reshape(-1, 1))
                targets_list.append(frame_targets)
                boxes_list.append(frame_boxes)

        if coords_list:
            coords = torch.tensor(np.concatenate(coords_list, axis=0), dtype=torch.int32)
            features = torch.tensor(np.concatenate(features_list, axis=0), dtype=torch.float32)
            targets = torch.tensor(np.concatenate(targets_list, axis=0), dtype=torch.float32)
            boxes = torch.tensor(np.concatenate(boxes_list, axis=0), dtype=torch.float32)
        else:
            coords = torch.tensor([[0, 0, 0]], dtype=torch.int32)
            features = torch.tensor([[0.0]], dtype=torch.float32)
            targets = torch.tensor([[0.0]], dtype=torch.float32)
            boxes = torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

        return coords, features, targets, boxes