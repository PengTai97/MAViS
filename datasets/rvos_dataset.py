import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import cv2


class RvosDataset(Dataset):
    def __init__(self, ann_file, img_folder, subset):
        """
        Args:
            ann_file (str or Path): Path to meta_expressions.json
            img_folder (str or Path): Root path to data (contains JPEGImages and Annotations)
            subset (str): 'valid' or 'test'
        """
        self.ann_file = Path(ann_file)
        self.img_folder = Path(img_folder)
        self.subset = subset
        self.prepare_metas()

    def prepare_metas(self):
        with open(self.ann_file, 'r') as f:
            data = json.load(f)
            videos = data['videos']

        self.metas = []
        for video_id, video_data in videos.items():
            expressions = video_data.get("expressions", {})
            frames = video_data.get("frames", [])
            if not frames:
                continue

            for expr_id, expr_data in expressions.items():
                caption = expr_data.get("exp", "")
                self.metas.append({
                    'video_id': video_id,
                    'expr_id': expr_id,
                    'caption': caption,
                    'frames': frames,
                })

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        meta = self.metas[idx]
        video_id = meta['video_id']
        expr_id = meta['expr_id']
        caption = " ".join(meta['caption'].lower().split())
        frames = meta['frames']
        sample_indices = list(range(len(frames)))

        base_folder = self.img_folder
        # ✅ 修正 Path 處理
        if base_folder.name == self.subset:
            base_folder = base_folder.parent

        jpeg_root = base_folder / self.subset / "JPEGImages" / video_id

        # 取得圖片大小（以第一張圖為準）
        first_img_path = jpeg_root / f"{frames[0]}.jpg"
        img_bgr = cv2.imread(str(first_img_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found: {first_img_path}")
        height, width = img_bgr.shape[:2]

        frame_paths = [str(jpeg_root / f"{fname}.jpg") for fname in frames]

        # ✅ 不輸出 masks
        target = {
            'frame_paths': frame_paths,
            'caption': caption,
            'path': str(jpeg_root),
            'num_frames': torch.tensor(len(frames)),
            'orig_size': torch.tensor([height, width]),
            'video_id': video_id,
            'expr_id': expr_id,
            'frames_idx': torch.tensor(sample_indices),
        }

        return target

















