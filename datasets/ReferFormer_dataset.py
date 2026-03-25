###########################################################################
# Created by: NTU
# Email: heshuting555@gmail.com
# Copyright (c) 2023
###########################################################################
"""
MeViS data loader
"""
from pathlib import Path

import torch
# from torch.autograd.grad_mode import F
from torch.utils.data import Dataset
#import datasets.transforms_video as T
import torchvision.transforms as T
import os
from PIL import Image
import json
import numpy as np
import random


from pycocotools import mask as coco_mask


class MeViSDataset(Dataset):
    """
    A dataset class for the MeViS dataset which was first introduced in the paper:
    "MeViS: A Large-scale Benchmark for Video Segmentation with Motion Expressions"
    """

    def __init__(self, img_folder: Path, ann_file: Path, transforms, return_masks: bool,
                 num_frames: int, max_skip: int , mode='train'):
        self.mode = mode
        self.img_folder = img_folder
        self.ann_file = ann_file
        self._transforms = transforms
        self.return_masks = return_masks  # not used
        self.num_frames = num_frames
        self.max_skip = max_skip
        # create video meta data
        self.prepare_metas()

        mask_json = os.path.join(str(self.img_folder) + '/mask_dict.json')
        print(f'Loading masks form {mask_json} ...')
        with open(mask_json) as fp:
            self.mask_dict = json.load(fp)

        print('\n video num: ', len(self.videos), ' clip num: ', len(self.metas))
        print('\n')
    

    def prepare_metas(self):
        # read expression data
        with open(str(self.ann_file), 'r') as f:
            subset_expressions_by_video = json.load(f)['videos']
        self.videos = list(subset_expressions_by_video.keys())

        self.metas = []
        for vid in self.videos:
            # vid_meta = subset_metas_by_video[vid]
            vid_data = subset_expressions_by_video[vid]
            vid_frames = sorted(vid_data['frames'])
            vid_len = len(vid_frames)

            for exp_id, exp_dict in vid_data['expressions'].items():
                for frame_id in range(0, vid_len, self.num_frames):
                    meta = {}
                    meta['video'] = vid
                    meta['exp'] = exp_dict['exp']
                    meta['obj_id'] = [int(x) for x in exp_dict['obj_id']]
                    meta['anno_id'] = [str(x) for x in exp_dict['anno_id']]
                    meta['frames'] = vid_frames
                    meta['frame_id'] = frame_id
                    meta['category'] = 0
                    # 添加唯一标识符 unique_id
                    meta['unique_id'] = f"{vid}_{exp_id}_{frame_id}"
                    self.metas.append(meta)
        #print(f"加载的视频数量：{len(self.videos)}")
        #print("视频列表：", self.videos)


    @staticmethod
    def bounding_box(img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax  # y1, y2, x1, x2

    def __len__(self):
        return len(self.metas)
    
    def sample_frames(self, vid_len, frame_id, num_frames):
        #print(f"Entering sample_frames with vid_len={vid_len}, frame_id={frame_id}, num_frames={num_frames}")
        sample_indx = [frame_id]

        if num_frames == 1:
            return sample_indx

        # 处理 sample_id_before
        sample_id_before_max = min(3, frame_id)
        if sample_id_before_max >= 1:
            sample_id_before = random.randint(1, sample_id_before_max)
        else:
            sample_id_before = 0

        # 处理 sample_id_after
        sample_id_after_max = min(3, vid_len - frame_id - 1)
        if sample_id_after_max >= 1:
            sample_id_after = random.randint(1, sample_id_after_max)
        else:
            sample_id_after = 0

        # 本地采样
        local_indx = [frame_id - sample_id_before, frame_id + sample_id_after]
        sample_indx.extend(local_indx)

        # 全局采样部分保持不变
        global_n = num_frames - len(sample_indx)
        all_inds = list(range(vid_len))
        excluded_inds = set(sample_indx)
        global_inds = [i for i in all_inds if i not in excluded_inds]

        while len(sample_indx) < num_frames:
            #print(f"In loop: len(sample_indx)={len(sample_indx)}, num_frames={num_frames}, global_n={global_n}")
            if global_inds:
                select_id = random.sample(global_inds, min(global_n, len(global_inds)))
            else:
                select_id = random.choices(all_inds, k=global_n)
            sample_indx.extend(select_id)
            sample_indx = list(set(sample_indx))  # 去重
            global_n = num_frames - len(sample_indx)

        sample_indx = sample_indx[:num_frames]
        sample_indx.sort()
        return sample_indx


    # def __getitem__(self, idx):
    #     try:
    #         instance_check = False
    #         attempt = 0
    #         max_attempts = 10  # 设定最大尝试次数
    #         while not instance_check and attempt < max_attempts:
    #             attempt += 1
    #             #print(f"Attempt {attempt}: Processing idx={idx}")
    #             meta = self.metas[idx]  # dict

    #             video, exp, anno_id, obj_id, category, frames, frame_id = \
    #                 meta['video'], meta['exp'], meta['anno_id'],meta['obj_id'], meta['category'], meta['frames'], meta['frame_id']
    #             # clean up the caption
    #             #print(f"Loaded meta data for idx={idx}")
    #             unique_id = meta['unique_id']
    #             exp = " ".join(exp.lower().split())
    #             category_id = 0
    #             vid_len = len(frames)
    #             # num_frames = self.num_frames
    #             # if vid_len < num_frames:
    #             #     num_frames = vid_len

    #             #print(f"vid_len: {vid_len}, frames: {frames}")
    #             if self.mode == 'val':
    #                 sample_indx = list(range(vid_len))  # 使用所有帧
    #                 num_frames = vid_len  # 使用完整帧长度
    #             else:
    #                 num_frames = self.num_frames if vid_len >= self.num_frames else vid_len
    #                 sample_indx = self.sample_frames(vid_len, frame_id, num_frames)
    #             # num_frames = self.num_frames if vid_len >= self.num_frames else vid_len
    #             # sample_indx = self.sample_frames(vid_len, frame_id, num_frames)

                
    #             # random sparse sample
    #             """ sample_indx = [frame_id]
    #             if self.num_frames != 1:
    #                 # local sample
    #                 sample_id_before = random.randint(1, 3)
    #                 sample_id_after = random.randint(1, 3)
    #                 local_indx = [max(0, frame_id - sample_id_before), min(vid_len - 1, frame_id + sample_id_after)]
    #                 sample_indx.extend(local_indx)

    #                 # global sampling
    #                 if num_frames > 3:
    #                     all_inds = list(range(vid_len))
    #                     global_inds = all_inds[:min(sample_indx)] + all_inds[max(sample_indx):]
    #                     global_n = num_frames - len(sample_indx)
    #                     if len(global_inds) > global_n:
    #                         select_id = random.sample(range(len(global_inds)), global_n)
    #                         for s_id in select_id:
    #                             sample_indx.append(global_inds[s_id])
    #                     elif vid_len >= global_n:  # sample long range global frames
    #                         select_id = random.sample(range(vid_len), global_n)
    #                         for s_id in select_id:
    #                             sample_indx.append(all_inds[s_id])
    #                     else:
    #                         select_count = max(0, global_n - vid_len)
    #                         select_id = random.sample(range(vid_len), select_count) + list(range(vid_len))
    #                         #select_id = random.sample(range(vid_len), global_n - vid_len) + list(range(vid_len))
    #                         for s_id in select_id:
    #                             sample_indx.append(all_inds[s_id])
    #             sample_indx.sort() """
    #             #sample_indx = self.sample_frames(vid_len, frame_id, num_frames)
    #             #print(f"Sampled frames: {sample_indx}")
    #             #GT_string = None
    #             GT_dict = {}
    #             masks_dict = {}  # 用于存储每个 anno_id 的 mask
    #             boxes_dict = {}
    #             # read frames and masks
    #             frame_paths = []
    #             imgs, labels, boxes, masks, valid,  = [], [], [], [], [], 
    #             v_path = os.path.join(str(self.img_folder), 'JPEGImages', video)
    #             #print(f"Video path: {v_path}")
    #             frame_indx_size = sample_indx[0]
    #             frame_name_size = frames[frame_indx_size]
    #             img_path_size = os.path.join(str(self.img_folder), 'JPEGImages', video, frame_name_size + '.jpg')
    #             with Image.open(img_path_size) as img:
    #                 w, h = img.size
    #             for j in range(num_frames):
    #                 frame_indx = sample_indx[j]
    #                 frame_name = frames[frame_indx]
    #                 img_path = os.path.join(str(self.img_folder), 'JPEGImages', video, frame_name + '.jpg')
    #                 #print(f"Loading image: {img_path}")
    #                 frame_paths.append(img_path)
    #                 # img = Image.open(img_path).convert('RGB')
    #                 # w, h = img.size
    #                 #frame_masks = {}
    #                 merged_mask = np.zeros((h, w), dtype=np.float32)
    #                 frame_boxes = {}
    #                 frame_bounding_boxes = []
    #                 # 遍历每个 anno_id，解码 mask 并计算 box
    #                 for anno in anno_id:
    #                     frm_anno = self.mask_dict[anno][frame_indx]
    #                     if frm_anno is not None:
    #                         decoded_mask = coco_mask.decode(frm_anno)
    #                         #frame_masks[anno] = decoded_mask.astype(np.float32)
    #                         merged_mask += decoded_mask.astype(np.float32)
    #                         # 计算该物体的 bounding box
    #                         if (decoded_mask > 0).any():
    #                             y1, y2, x1, x2 = self.bounding_box(decoded_mask)
    #                             # 将 box 转换到 [0, 1] 范围
    #                             norm_box = np.array([
    #                                 x1/w,
    #                                 y1/h, 
    #                                 x2/w,
    #                                 y2/h, 
    #                             ], dtype=np.float32)
    #                             frame_boxes[anno] = norm_box
    #                             bbox_str = f"[{', '.join(f'{x:.2f}' for x in norm_box)}]"
    #                         else:
    #                             frame_boxes[anno] = np.zeros(4, dtype=np.float32)
    #                             bbox_str ="[]"
    #                     else:
    #                         #frame_masks[anno] = np.zeros(img.size[::-1], dtype=np.float32)
    #                         frame_boxes[anno] = np.zeros(4, dtype=np.float32)
    #                         bbox_str = "[]"

    #                     frame_bounding_boxes.append(bbox_str)

    #                 #masks_dict[frame_indx] = frame_masks  # 存储当前帧的所有 mask
    #                 merged_mask = np.clip(merged_mask, 0, 1)
    #                 masks_dict[frame_indx] = merged_mask.astype(np.float32)  # 存储当前帧的合并掩码
    #                 boxes_dict[frame_indx] = frame_boxes
    #                 GT_dict[frame_indx] = bbox_str
    #                 # 检查是否有有效的物体
    #                 #valid_mask = any((mask > 0).any() for mask in frame_masks.values())
    #                 valid_mask = (merged_mask > 0).any()
    #                 valid.append(1 if valid_mask else 0)

    #                 # 如果任何一个 bbox_str 不是 None，则存储在 GT_dict 中
    #                 if any(bbox_str is not None for bbox_str in frame_bounding_boxes):
    #                     # 存储当前帧的边界框字符串
    #                     GT_dict[frame_indx] = frame_bounding_boxes
    #                 else:
    #                     GT_dict[frame_indx] = []  # 当前帧没有有效的边界框

    #                 # mask = np.zeros(img.size[::-1], dtype=np.float32)
    #                 # for x in anno_id:
    #                 #     frm_anno = self.mask_dict[x][frame_indx]
    #                 #     if frm_anno is not None:
    #                 #         mask += coco_mask.decode(frm_anno)

    #                 # append
                    
    #                 #imgs.append(img)# gvideo test
    #                 label = torch.tensor(category_id)
    #                 labels.append(label)
    #             # 如果这是第一个有物体的帧，生成 GT_string
    #             try:
    #             #     # 使用 next() 和 enumerate() 查找第一个值为 1 的索引
    #                 anchor_frame = next(i for i, v in enumerate(valid) if v == 1)
                    
    #             #     # 找到该帧的边界框
    #             #     frame_boxes = boxes_dict[sample_indx[anchor_frame]]
    #             #     #time_percentage = anchor_frame / self.num_frames  # 計算時間佔比
    #             #     bounding_boxes = [
    #             #         f"[{', '.join(f'{x:.2f}' for x in frame_boxes[anno])}]" for anno in anno_id
    #             #     ]
    #             #     #GT_string = f"{anchor_frame} {time_percentage:.2f} " + " ".join(bounding_boxes)
    #             #     GT_string = f"{anchor_frame} " + " ".join(bounding_boxes)
    #             except StopIteration:
    #             #     # 如果没有找到值为 1 的元素，则处理这种情况
    #                 anchor_frame = -1
    #                 #GT_string = None
                    
        
    #             frame_paths = [str(p) if not isinstance(p, (tuple, list)) else p[0] for p in frame_paths]    
    #             # transform
    #             labels = torch.stack(labels, dim=0)

    #             target = {
    #                 'frames_idx': torch.tensor(sample_indx),  # [T,]
    #                 'num_frames':vid_len,
    #                 'labels': labels,  # [T,]
    #                 'boxes': boxes_dict,  # [T, 4], xyxy, normalize to 0-1
    #                 'masks': masks_dict,  # [T, H, W]
    #                 'valid': torch.tensor(valid),  # [T,]
    #                 'caption': exp,
    #                 'orig_size': torch.as_tensor([int(h), int(w)]),
    #                 'size': torch.as_tensor([int(h), int(w)]),
    #                 'path': v_path,
    #                 'groundtruth':GT_dict,# key:frame_id, ['anno1', 'anno2']
    #                 'obj_id' : anno_id,
    #                 'anchor_frame':anchor_frame,
    #                 'unique_id': unique_id,
    #                 'frame_paths': frame_paths,
    #             }

    #             # "boxes" normalize to [0, 1] and transform from xyxy to cxcywh in self._transform
    #             #imgs, target = self._transforms(imgs, target)

    #             # imgs = [self._transforms(img) for img in imgs] # gvideo test
    #             # imgs = torch.stack(imgs, dim=0)  # [T, 3, H, W] # gvideo test

    #             # FIXME: handle "valid", since some box may be removed due to random crop
    #             if torch.any(target['valid'] == 1):  # at leatst one instance
    #                 instance_check = True
    #                 #print(f"Instance found for idx={idx}")
    #             else:
    #                 #raise ValueError(f"No valid instance found for idx={idx}")
    #                 #print(f"No valid instance for idx={idx}, selecting new idx")
    #                 idx = random.randint(0, self.__len__() - 1)

    #         #return imgs, target
    #         return  target  # gvideo test
    #     except Exception as e:
    #         print(f"Error in __getitem__ at index {idx}: {e}")
    #         raise e
    #     #return  target # gvideo test

    def __getitem__(self, idx):
        """
        只回傳 main() 需要的 8 個欄位，並保留與舊版相同的 dtype / 值域。
        """
        attempt, max_attempts = 0, 10
        while attempt < max_attempts:
            attempt += 1
            meta = self.metas[idx]

            video      = meta["video"]
            frames     = meta["frames"]
            frame_id   = meta["frame_id"]
            anno_ids   = meta["anno_id"]
            vid_len    = len(frames)

            # 1) 抽 frame
            if self.mode == "val":
                sample_indx = list(range(vid_len))
            else:
                n_frames    = min(self.num_frames, vid_len)
                sample_indx = self.sample_frames(vid_len, frame_id, n_frames)

            # 2) 只開第一張取 (w, h)
            first_name = frames[sample_indx[0]]
            first_path = os.path.join(str(self.img_folder), "JPEGImages", video, f"{first_name}.jpg")
            with Image.open(first_path) as img:
                w, h = img.size

            # 3) frame 路徑
            v_path      = os.path.join(str(self.img_folder), "JPEGImages", video)
            frame_paths = [os.path.join(v_path, f"{frames[i]}.jpg") for i in sample_indx]

            # 4) 合併 mask
            masks_dict, has_instance = {}, False
            for f_idx in sample_indx:
                merged = np.zeros((h, w), dtype=np.uint8)
                for anno in anno_ids:
                    rle = self.mask_dict[anno][f_idx]
                    if rle is not None:
                        merged |= coco_mask.decode(rle).astype(np.uint8)
                if merged.any():
                    has_instance = True
                # clip 到 0/1 後轉 float32 —— 與舊版一致
                masks_dict[f_idx] = np.clip(merged, 0, 1).astype(np.float32)

            if not has_instance:
                idx = random.randint(0, self.__len__() - 1)
                continue

            # 5) target
            target = {
                "frames_idx": torch.tensor(sample_indx, dtype=torch.long),
                "num_frames": vid_len,
                "masks":      masks_dict,
                "caption":    " ".join(meta["exp"].lower().split()),
                "orig_size":  torch.as_tensor([h, w], dtype=torch.long),   # 與舊版一致 (int64)
                "path":       v_path,
                "unique_id":  meta["unique_id"],
                "frame_paths": frame_paths,
            }
            return target

        # 若連續 max_attempts 都失敗，回傳最後一次結果
        return target


def make_coco_transforms(image_set, max_size=640):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [288, 320, 352, 392, 416, 448, 480, 512]

    #if image_set == 'train':
        #return T.Compose([
            #T.RandomHorizontalFlip(),
            #T.PhotometricDistort(),
            #T.RandomSelect(
                #T.Compose([
                    #T.RandomResize(scales, max_size=max_size),
                    #T.Check(),
                #]),
                #T.Compose([
                    #T.RandomResize([400, 500, 600]),
                    #T.RandomSizeCrop(384, 600),
                    #T.RandomResize(scales, max_size=max_size),
                    #T.Check(),
                #])
            #),
            #normalize,
        #])
    if image_set == 'train':
        return T.Compose([
           #T.RandomResize([360], max_size=640),
            T.Resize((384,384)),
            normalize,
        ])

    # we do not use the 'val' set since the annotations are inaccessible
    if image_set == 'val':
        return T.Compose([
           #T.RandomResize([360], max_size=640),
           T.Resize((384,384)),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.rovos_path)
    assert root.exists(), f'provided YTVOS path {root} does not exist'
    PATHS = {
        "train": (root / "train", root / "train" / "meta_expressions.json"),
        "val": (root / "valid", root / "meta_expressions" / "val" / "meta_expressions.json"),  # not used actually
    }
    img_folder, ann_file = PATHS['train']
    dataset = MeViSDataset(img_folder, ann_file, transforms=make_coco_transforms(image_set, max_size=args.max_size), return_masks=args.masks,
                           num_frames=args.num_frames, max_skip=args.max_skip)
    return dataset


if __name__ == '__main__':
    root = Path('data/mevis_release')
    image_set = 'train'
    PATHS = {
        "train": (root / "train", root / "train" / "meta_expressions.json"),
    }
    img_folder, ann_file = PATHS['train']

    dataset = MeViSDataset(img_folder, ann_file, transforms=T.ToTensor(), return_masks=True, num_frames=5, max_skip=3)

    img, meta = dataset[0]