import os
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2


class MeViSDataset(Dataset):
    def __init__(self, img_folder: Path, ann_file: Path, num_frames: int, mode="val"):
        self.mode = mode
        self.img_folder = img_folder
        self.ann_file = ann_file
        self.num_frames = num_frames
        self.prepare_metas()

        print(f"\n視頻數量： {len(self.videos)}, clip 數量： {len(self.metas)}\n")
        self.video_sizes = {}

    def prepare_metas(self):
        with open(str(self.ann_file), 'r') as f:
            data = json.load(f)
        subset_expressions_by_video = data['videos']
        self.videos = list(subset_expressions_by_video.keys())
        self.metas = []
        for vid in self.videos:
            vid_data = subset_expressions_by_video[vid]
            vid_frames = sorted(vid_data['frames'])
            for exp_id, exp_dict in vid_data['expressions'].items():
                meta = {}
                meta['video'] = vid
                meta['exp_id'] = exp_id
                meta['exp'] = " ".join(exp_dict['exp'].lower().split())
                meta['frames'] = vid_frames
                meta['frame_id'] = 0
                self.metas.append(meta)

    def __len__(self):
        return len(self.metas)

    def get_video_size(self, video, frames):
        if video in self.video_sizes:
            return self.video_sizes[video]
        else:
            v_path = os.path.join(str(self.img_folder), 'JPEGImages', video)
            first_frame_file = os.path.join(v_path, frames[0] + '.jpg')
            im = cv2.imread(first_frame_file)
            if im is None:
                raise ValueError(f"無法讀取圖像： {first_frame_file}")
            h, w = im.shape[:2]
            self.video_sizes[video] = (h, w)
            return h, w

    def __getitem__(self, idx):
        meta = self.metas[idx]
        video = meta['video']
        exp_id = meta['exp_id']
        exp = meta['exp']
        frames = meta['frames']
        vid_len = len(frames)

        sample_indx = list(range(vid_len))
        num_frames = vid_len

        v_path = os.path.join(str(self.img_folder), 'JPEGImages', video)
        frame_paths = [os.path.join(v_path, frame_name + '.jpg') for frame_name in frames]

        h, w = self.get_video_size(video, frames)
        orig_size = torch.as_tensor([h, w])

        target = {
            'frames_idx': torch.tensor(sample_indx),
            'num_frames': num_frames,
            'orig_size': orig_size,
            'path': v_path,
            'caption': exp,
            'video_id': video,
            'exp_id': exp_id,
            'frame_paths': frame_paths,
        }
        return target


# 測試
if __name__ == '__main__':
    root_dir = Path("/home/ai2lab/sam24mllm/Data/MeViS")
    img_folder = root_dir / "valid_u"
    ann_file = img_folder / "meta_expressions.json"

    dataset = MeViSDataset(img_folder=img_folder, ann_file=ann_file, num_frames=20, mode="val")
    sample = dataset[0]

    print("Sample keys:", sample.keys())
    print("video_id:", sample['video_id'])
    print("exp_id:", sample['exp_id'])
    print("caption:", sample['caption'])
    print("第一幀路徑:", sample['frame_paths'][0])
