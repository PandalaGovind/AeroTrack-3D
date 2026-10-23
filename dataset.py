import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset

class IRSequenceDataset(Dataset):
    def __init__(self, sequence_dir, sequence_length=10, threshold=20):
        """
        sequence_dir: Path to the directory containing sequence folders (e.g., SIATD_Dataset/sequences)
        sequence_length: Number of frames to stack (PISTTN uses 10)
        threshold: Hard thresholding value for the sparse component (lambda_seg = 20)
        """
        self.sequence_dir = sequence_dir
        self.sequence_length = sequence_length
        self.threshold = threshold
        
        # Collect all valid frame sequences here
        self.samples = self._build_sequence_samples()

    def _build_sequence_samples(self):
        samples = []
        
        # Iterate over each sequence folder (e.g., seq_001, seq_002)
        for seq_folder in sorted(os.listdir(self.sequence_dir)):
            seq_path = os.path.join(self.sequence_dir, seq_folder)
            if not os.path.isdir(seq_path):
                continue
                
            # Grab all image files and sort them to ensure chronological order
            frames = sorted([f for f in os.listdir(seq_path) if f.endswith(('.png', '.jpg'))])
            
            # Create sliding windows of 10 frames
            # E.g., Window 1: frames 0-9, Window 2: frames 1-10
            for i in range(len(frames) - self.sequence_length + 1):
                chunk = [os.path.join(seq_path, frames[i + j]) for j in range(self.sequence_length)]
                samples.append(chunk)
                
        return samples

    def _apply_tsvd(self, frame_stack):
        """
        Applies truncated SVD to separate low-rank background from sparse targets.
        """
        # Flatten the spatial dimensions for SVD: Shape becomes (Time, Height * Width)
        t, h, w = frame_stack.shape
        matrix = frame_stack.reshape(t, -1)
        
        # Perform SVD
        U, S, V = np.linalg.svd(matrix, full_matrices=False)
        
        # Reconstruct background using the top singular value (low-rank component)
        # S_trunc keeps only the first singular value
        S_trunc = np.zeros_like(S)
        S_trunc[0] = S[0] 
        background = np.dot(U * S_trunc, V)
        
        # Extract sparse target component
        sparse_target = matrix - background
        sparse_target = sparse_target.reshape(t, h, w)
        
        # Apply hard thresholding (lambda_seg = 20)
        sparse_target[sparse_target < self.threshold] = 0
        
        return sparse_target

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths = self.samples[idx]
        
        # Load the 10 frames
        frames = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in frame_paths]
        frame_stack = np.stack(frames, axis=0).astype(np.float32)
        
        # Process the stack to get the sparse tensor
        sparse_tensor = self._apply_tsvd(frame_stack)
        
        # Convert to PyTorch tensor
        return torch.tensor(sparse_tensor).unsqueeze(0) # Add channel dimension