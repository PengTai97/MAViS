import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import json
import numpy as np
import random

class RefDAVISDataset(Dataset):
    """
    只使用 meta_expressions.json，讀取 mask 和語句，輸出 target，不返回圖片。
    """
    def __init__(self,
                 img_folder,
                 ann_file,
                 num_frames,
                 transforms=None,
                 mode='train'):
        self.img_folder = img_folder  # e.g. .../ref-davis/valid
        self.ann_file = ann_file      # e.g. .../ref-davis/meta_expressions/valid/meta_expressions.json
        self.num_frames = num_frames
        self.mode = mode
        self._transforms = transforms

        self.prepare_metas()
        print('\n video num: ', len(self.videos), ' clip num: ', len(self.metas), '\n')

    def prepare_metas(self):
        # 只讀取 meta_expressions.json
        with open(self.ann_file, 'r') as f:
            subset_expressions_by_video = json.load(f)['videos']

        self.videos = list(subset_expressions_by_video.keys())
        self.metas = []

        for vid in self.videos:
            vid_data = subset_expressions_by_video[vid]
            vid_frames = sorted(vid_data['frames'])
            vid_len = len(vid_frames)

            for exp_id, exp_dict in vid_data['expressions'].items():
                for frame_id in range(0, vid_len, self.num_frames):
                    meta = {
                        'video': vid,
                        'exp': exp_dict['exp'],
                        'obj_id': int(exp_dict['obj_id']),
                        'frames': vid_frames,
                        'frame_id': frame_id,
                        'unique_id': f"{vid}_{exp_id}_{frame_id}"
                    }
                    self.metas.append(meta)
                    if self.mode == "val":
                        # 在驗證模式下只取每個 expression 的第一個 meta
                        break

    @staticmethod
    def bounding_box(img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax  # y1, y2, x1, x2

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        meta = self.metas[idx]
        video = meta['video']
        #exp = " ".join(meta['exp'].lower().split())
        exp = meta['exp']
        obj_id = meta['obj_id']
        frames = meta['frames']
        frame_id = meta['frame_id']
        unique_id = meta['unique_id']

        vid_len = len(frames)

        # 根據 mode 區分采樣策略
        if self.mode == 'val':
            # 驗證模式下不做采樣，直接返回所有 frame 的索引
            sample_indx = list(range(vid_len))
        else:
            # 訓練模式下按照原邏輯采樣 num_frames 個 frame
            sample_indx = [frame_id]
            if self.num_frames != 1:
                sample_id_before = random.randint(1, 3)
                sample_id_after  = random.randint(1, 3)
                local_indx = [
                    max(0, frame_id - sample_id_before),
                    min(vid_len - 1, frame_id + sample_id_after)
                ]
                sample_indx.extend(local_indx)
                if self.num_frames > 3:
                    all_inds = list(range(vid_len))
                    global_inds = [i for i in all_inds if i not in sample_indx]
                    global_n = self.num_frames - len(sample_indx)
                    if len(global_inds) > global_n:
                        select_id = random.sample(global_inds, global_n)
                    else:
                        need_more = global_n - len(global_inds)
                        select_id = global_inds + random.choices(all_inds, k=need_more)
                    sample_indx.extend(select_id)
            sample_indx = sorted(list(set(sample_indx)))[:self.num_frames]

        labels = []
        valid = []
        boxes_dict = {}
        masks_dict = {}
        groundtruth_dict = {}
        frame_paths = []

        H, W = None, None
        for frame_i in sample_indx:
            frame_name = frames[frame_i]
            img_path = os.path.join(self.img_folder, 'JPEGImages', video, frame_name + '.jpg')
            mask_path = os.path.join(self.img_folder, 'Annotations', video, frame_name + '.png')
            frame_paths.append(img_path)

            mask_im = Image.open(mask_path).convert("P")
            mask_np = np.array(mask_im)

            if H is None or W is None:
                H, W = mask_np.shape

            obj_mask = (mask_np == obj_id).astype(np.float32)
            if (obj_mask > 0).any():
                y1, y2, x1, x2 = self.bounding_box(obj_mask)
                norm_box = np.array([x1/W, y1/H, x2/W, y2/H], dtype=np.float32)
                valid.append(1)
                box_str = f"[{norm_box[0]:.2f}, {norm_box[1]:.2f}, {norm_box[2]:.2f}, {norm_box[3]:.2f}]"
                groundtruth_dict[frame_i] = [box_str]
            else:
                norm_box = np.zeros(4, dtype=np.float32)
                valid.append(0)
                groundtruth_dict[frame_i] = []

            boxes_dict[frame_i] = norm_box
            masks_dict[frame_i] = obj_mask
            labels.append(torch.tensor(0))  # 類別統一設為 0

        labels_tensor = torch.stack(labels, dim=0)

        try:
            anchor_frame = next(i for i, v in enumerate(valid) if v == 1)
        except StopIteration:
            anchor_frame = -1

        target = {
            'frames_idx': torch.tensor(sample_indx),
            'num_frames': vid_len,
            'labels': labels_tensor,
            'boxes': boxes_dict,
            'masks': masks_dict,
            'valid': torch.tensor(valid),
            'caption': exp,
            'orig_size': torch.as_tensor([H, W]),
            'size': torch.as_tensor([H, W]),
            'path': os.path.join(self.img_folder, 'JPEGImages', video),
            'groundtruth': groundtruth_dict,
            'obj_id': [obj_id],
            'anchor_frame': anchor_frame,
            'unique_id': unique_id,
            'frame_paths': frame_paths
        }

        return target


