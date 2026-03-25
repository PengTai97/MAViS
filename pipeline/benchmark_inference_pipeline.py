from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from sam2.build_sam import build_sam2_video_predictor

from video_summarization_agent import VideoSummarizationAgent, load_qwen2_5_vl_model
from keyframe_selection_agent import KeyframeSelectionAgent
from object_grounding_agent import ObjectGroundingAgent


# -----------------------------
# Core MAViS pipeline
# -----------------------------


def load_predictor(
    sam2_checkpoint: str,
    model_cfg: str,
    device: str = "cuda",
    vos_optimized: bool = True,
):
    return build_sam2_video_predictor(
        config_file=model_cfg,
        ckpt_path=sam2_checkpoint,
        device=device,
        vos_optimized=vos_optimized,
        apply_postprocessing=True,
    )


def reset_inference_state(predictor, video_dir: str):
    inference_state = predictor.init_state(
        video_path=video_dir,
        offload_video_to_cpu=True,
        async_loading_frames=True,
    )
    predictor.reset_state(inference_state)
    return inference_state


def uniform_sample_indices(num_frames: int, num_samples: int = 10) -> List[int]:
    if num_frames <= 0:
        return []
    if num_frames <= num_samples:
        return list(range(num_frames))
    return list(np.linspace(0, num_frames - 1, num_samples, dtype=int))


def get_image_size(image_path: str) -> Tuple[int, int]:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    height, width = image.shape[:2]
    return height, width


def add_boxes_to_sam2(predictor, inference_state, frame_idx: int, boxes_xyxy: Sequence[Sequence[int]]) -> bool:
    any_box_added = False
    for box_idx, box in enumerate(boxes_xyxy):
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=box_idx + 1,
            box=box,
        )
        any_box_added = True
    return any_box_added


def propagate_masks_bidirectionally(predictor, inference_state, start_frame: int):
    video_segments: Dict[int, Dict[int, np.ndarray]] = {}

    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state,
        reverse=False,
    ):
        video_segments[out_frame_idx] = {
            int(out_obj_id): (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state,
        reverse=True,
    ):
        if out_frame_idx < start_frame:
            break
        video_segments[out_frame_idx] = {
            int(out_obj_id): (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    return video_segments


@dataclass
class MavisResult:
    description: str
    summary: str
    start_frame: int
    key_frame_indices: List[int]
    ranked_indices: List[int]
    frame_scores: List[float]
    boxes_by_frame: Dict[int, List[List[int]]]
    masks_by_frame: Dict[int, Dict[int, np.ndarray]]


class MavisInferencePipeline:
    def __init__(
        self,
        sam2_checkpoint: str,
        sam2_config: str,
        qwen_model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda",
    ):
        model, processor = load_qwen2_5_vl_model(qwen_model_name)
        self.vs_agent = VideoSummarizationAgent(model, processor)
        self.ks_agent = KeyframeSelectionAgent(model, processor)
        self.og_agent = ObjectGroundingAgent(model, processor)
        self.predictor = load_predictor(
            sam2_checkpoint=sam2_checkpoint,
            model_cfg=sam2_config,
            device=device,
        )

    def run(
        self,
        frame_paths: Sequence[str],
        description: str,
        video_dir: Optional[str] = None,
        num_summary_samples: int = 10,
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> MavisResult:
        all_frame_paths = list(frame_paths)
        if not all_frame_paths:
            raise ValueError("frame_paths is empty.")

        if video_dir is None:
            video_dir = str(Path(all_frame_paths[0]).parent)

        sampled_indices = uniform_sample_indices(len(all_frame_paths), num_summary_samples)
        sampled_paths = [all_frame_paths[idx] for idx in sampled_indices]

        summary, summary_history = self.vs_agent(sampled_paths, description)
        ks_result = self.ks_agent(
            frame_paths=all_frame_paths,
            description=description,
            video_summary=summary,
            top_k=top_k,
            threshold=threshold,
            frame_indices=list(range(len(all_frame_paths))),
        )

        height, width = get_image_size(all_frame_paths[0])
        boxes_by_frame: Dict[int, List[List[int]]] = {}
        inference_state = reset_inference_state(self.predictor, video_dir)

        any_box_added = False
        for key_idx in ks_result["key_frame_indices"]:
            boxes = self.og_agent(
                image_path=all_frame_paths[key_idx],
                description=description,
                image_width=width,
                image_height=height,
                conversation_history=summary_history,
            )
            boxes_by_frame[int(key_idx)] = boxes
            if boxes:
                any_box_added = add_boxes_to_sam2(
                    predictor=self.predictor,
                    inference_state=inference_state,
                    frame_idx=int(key_idx),
                    boxes_xyxy=boxes,
                ) or any_box_added

        masks_by_frame = (
            propagate_masks_bidirectionally(
                predictor=self.predictor,
                inference_state=inference_state,
                start_frame=int(ks_result["start_frame"]),
            )
            if any_box_added
            else {}
        )

        return MavisResult(
            description=description,
            summary=summary,
            start_frame=int(ks_result["start_frame"]),
            key_frame_indices=list(map(int, ks_result["key_frame_indices"])),
            ranked_indices=list(map(int, ks_result["ranked_indices"])),
            frame_scores=[float(x) for x in ks_result["scores"]],
            boxes_by_frame=boxes_by_frame,
            masks_by_frame=masks_by_frame,
        )


# -----------------------------
# Dataset adapters
# -----------------------------


def build_dataset(dataset_name: str, dataset_root: str, split: str, num_frames: int = 20, max_skip: int = 3):
    root = Path(dataset_root)
    name = dataset_name.lower()

    if name == "referformer":
        from ReferFormer_dataset import build as build_referformer

        args = SimpleNamespace(
            rovos_path=str(root),
            max_size=640,
            masks=True,
            num_frames=num_frames,
            max_skip=max_skip,
        )
        return build_referformer(split, args)

    if name == "davis":
        from davis_dataset import RefDAVISDataset

        split_dir = root / split
        ann_file = root / "meta_expressions" / split / "meta_expressions.json"
        return RefDAVISDataset(
            img_folder=str(split_dir),
            ann_file=str(ann_file),
            num_frames=num_frames,
            transforms=None,
            mode="val" if split != "train" else "train",
        )

    if name == "rvos":
        from rvos_dataset import RvosDataset

        ann_file = root / "meta_expressions" / split / "meta_expressions.json"
        img_folder = root / split
        return RvosDataset(
            ann_file=str(ann_file),
            img_folder=str(img_folder),
            subset=split,
        )

    if name == "mevis":
        from mevis_dataset2 import MeViSDataset

        split_dir = root / split
        ann_file = split_dir / "meta_expressions.json"
        return MeViSDataset(
            img_folder=split_dir,
            ann_file=ann_file,
            num_frames=num_frames,
            mode="val",
        )

    raise ValueError(f"Unsupported dataset: {dataset_name}")


# -----------------------------
# Output and evaluation helpers
# -----------------------------


def combine_object_masks(mask_dict: Mapping[int, np.ndarray], shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    merged = np.zeros((h, w), dtype=np.uint8)
    for mask in mask_dict.values():
        mask_np = np.asarray(mask)
        if mask_np.ndim == 3:
            mask_np = mask_np.squeeze(0)
        merged |= (mask_np > 0).astype(np.uint8)
    return np.clip(merged, 0, 1).astype(np.uint8)



def save_mask_sequence(
    result: MavisResult,
    frame_paths: Sequence[str],
    output_dir: str,
    dataset_name: str,
    target: Mapping[str, Any],
):
    os.makedirs(output_dir, exist_ok=True)
    height, width = get_image_size(frame_paths[0])
    name = dataset_name.lower()

    if name == "mevis":
        num_frames_val = int(target["num_frames"])
        for frame_idx in range(num_frames_val):
            pred_mask = result.masks_by_frame.get(frame_idx)
            if pred_mask:
                merged = combine_object_masks(pred_mask, (height, width))
            else:
                merged = np.zeros((height, width), dtype=np.uint8)
            out_path = os.path.join(output_dir, f"{frame_idx:05d}.png")
            cv2.imwrite(out_path, (merged * 255).astype(np.uint8))
        return

    if name == "rvos":
        frame_indices = target.get("frames_idx", list(range(len(frame_paths))))
        if hasattr(frame_indices, "tolist"):
            frame_indices = frame_indices.tolist()
        for current_frame_idx in frame_indices:
            current_frame_idx = int(current_frame_idx)
            pred_mask = result.masks_by_frame.get(current_frame_idx)
            if pred_mask:
                merged = combine_object_masks(pred_mask, (height, width))
            else:
                merged = np.zeros((height, width), dtype=np.uint8)
            frame_file_name = os.path.basename(frame_paths[current_frame_idx])
            frame_stem = Path(frame_file_name).stem
            out_path = os.path.join(output_dir, f"{frame_stem}.png")
            cv2.imwrite(out_path, (merged * 255).astype(np.uint8))
        return

    for frame_idx, frame_path in enumerate(frame_paths):
        pred_mask = result.masks_by_frame.get(frame_idx)
        if pred_mask:
            merged = combine_object_masks(pred_mask, (height, width))
        else:
            merged = np.zeros((height, width), dtype=np.uint8)
        stem = Path(frame_path).stem
        out_path = os.path.join(output_dir, f"{stem}.png")
        cv2.imwrite(out_path, merged)



def serialize_result_without_masks(result: MavisResult) -> Dict[str, Any]:
    payload = asdict(result)
    payload["masks_by_frame"] = {
        int(frame_idx): [int(obj_id) for obj_id in obj_dict.keys()]
        for frame_idx, obj_dict in result.masks_by_frame.items()
    }
    payload["boxes_by_frame"] = {int(k): v for k, v in result.boxes_by_frame.items()}
    return payload



def save_metadata_json(result: MavisResult, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(serialize_result_without_masks(result), f, ensure_ascii=False, indent=2)



def mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_bin = pred.astype(bool)
    gt_bin = gt.astype(bool)
    union = np.logical_or(pred_bin, gt_bin).sum()
    if union == 0:
        return 1.0
    inter = np.logical_and(pred_bin, gt_bin).sum()
    return float(inter / union)



def evaluate_sequence_iou(result: MavisResult, target: Mapping[str, Any]) -> Dict[str, float]:
    frame_indices = list(map(int, target["frames_idx"].tolist() if hasattr(target["frames_idx"], "tolist") else target["frames_idx"]))
    gt_masks: Mapping[int, np.ndarray] = target["masks"]
    h, w = map(int, target["orig_size"].tolist() if hasattr(target["orig_size"], "tolist") else target["orig_size"])

    per_frame = []
    for local_idx, dataset_frame_idx in enumerate(frame_indices):
        pred_mask_dict = result.masks_by_frame.get(local_idx, {})
        pred_mask = combine_object_masks(pred_mask_dict, (h, w)) if pred_mask_dict else np.zeros((h, w), dtype=np.uint8)
        gt_mask = np.asarray(gt_masks[dataset_frame_idx]).astype(np.uint8)
        per_frame.append(mask_iou(pred_mask, gt_mask))

    mean_iou = float(np.mean(per_frame)) if per_frame else 0.0
    return {
        "mean_iou": mean_iou,
        "num_frames_eval": len(per_frame),
    }



def sample_identifier(dataset_name: str, target: Mapping[str, Any]) -> str:
    name = dataset_name.lower()
    if name in {"referformer", "davis"}:
        return str(target.get("unique_id", "sample"))
    if name == "rvos":
        return f"{target['video_id']}__{target['expr_id']}"
    if name == "mevis":
        return f"{target['video_id']}__{target['exp_id']}"
    return "sample"



def prediction_output_dir(dataset_name: str, output_root: str, target: Mapping[str, Any]) -> str:
    name = dataset_name.lower()
    if name == "rvos":
        return os.path.join(output_root, "Annotations", str(target["video_id"]), str(target["expr_id"]))
    if name == "mevis":
        return os.path.join(output_root, str(target["video_id"]), str(target["exp_id"]))
    return os.path.join(output_root, sample_identifier(dataset_name, target))


# -----------------------------
# Benchmark runner
# -----------------------------


@dataclass
class BenchmarkSummary:
    dataset: str
    split: str
    num_samples: int
    evaluated_samples: int
    mean_iou: Optional[float]
    prediction_root: str
    metadata_root: str


class BenchmarkRunner:
    def __init__(
        self,
        pipeline: MavisInferencePipeline,
        dataset_name: str,
        prediction_root: str,
        metadata_root: str,
        num_summary_samples: int = 10,
        top_k: int = 5,
        threshold: float = 0.0,
    ):
        self.pipeline = pipeline
        self.dataset_name = dataset_name.lower()
        self.prediction_root = prediction_root
        self.metadata_root = metadata_root
        self.num_summary_samples = num_summary_samples
        self.top_k = top_k
        self.threshold = threshold

    def run_dataset(self, dataset, max_samples: Optional[int] = None, split: str = "val") -> BenchmarkSummary:
        os.makedirs(self.prediction_root, exist_ok=True)
        os.makedirs(self.metadata_root, exist_ok=True)

        metric_values: List[float] = []
        total = len(dataset) if max_samples is None else min(len(dataset), max_samples)

        for idx in range(total):
            target = dataset[idx]
            frame_paths = list(target["frame_paths"])
            video_dir = str(target.get("path", Path(frame_paths[0]).parent))
            description = str(target["caption"])
            sample_id = sample_identifier(self.dataset_name, target)

            result = self.pipeline.run(
                frame_paths=frame_paths,
                description=description,
                video_dir=video_dir,
                num_summary_samples=self.num_summary_samples,
                top_k=self.top_k,
                threshold=self.threshold,
            )

            pred_dir = prediction_output_dir(self.dataset_name, self.prediction_root, target)
            meta_path = os.path.join(self.metadata_root, f"{sample_id}.json")
            save_metadata_json(result, meta_path)

            if self.dataset_name in {"rvos", "mevis"}:
                save_mask_sequence(result, frame_paths, pred_dir, dataset_name=self.dataset_name, target=target)
            else:
                metrics = evaluate_sequence_iou(result, target)
                metric_values.append(metrics["mean_iou"])
                sample_report = {
                    "sample_id": sample_id,
                    "metrics": metrics,
                    "prediction_dir": pred_dir,
                }
                save_mask_sequence(result, frame_paths, pred_dir, dataset_name=self.dataset_name, target=target)
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta_data = json.load(f)
                meta_data["local_eval"] = metrics
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta_data, f, ensure_ascii=False, indent=2)
                print(json.dumps(sample_report, ensure_ascii=False))

            print(json.dumps({
                "dataset": self.dataset_name,
                "sample": idx + 1,
                "sample_id": sample_id,
                "summary": result.summary,
                "start_frame": result.start_frame,
                "key_frames": result.key_frame_indices,
            }, ensure_ascii=False))

        summary = BenchmarkSummary(
            dataset=self.dataset_name,
            split=split,
            num_samples=total,
            evaluated_samples=len(metric_values),
            mean_iou=float(np.mean(metric_values)) if metric_values else None,
            prediction_root=self.prediction_root,
            metadata_root=self.metadata_root,
        )

        with open(os.path.join(self.metadata_root, "benchmark_summary.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(summary), f, ensure_ascii=False, indent=2)
        return summary


# -----------------------------
# CLI
# -----------------------------


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified MAViS benchmark inference runner.")
    parser.add_argument("--dataset", required=True, choices=["referformer", "davis", "rvos", "mevis"])
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--split", default="valid")
    parser.add_argument("--sam2_checkpoint", required=True)
    parser.add_argument("--sam2_config", required=True)
    parser.add_argument("--qwen_model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--prediction_root", required=True)
    parser.add_argument("--metadata_root", required=True)
    parser.add_argument("--num_frames", type=int, default=20)
    parser.add_argument("--max_skip", type=int, default=3)
    parser.add_argument("--num_summary_samples", type=int, default=10)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--max_samples", type=int, default=None)
    return parser


if __name__ == "__main__":
    args = build_cli().parse_args()

    dataset = build_dataset(
        dataset_name=args.dataset,
        dataset_root=args.dataset_root,
        split=args.split,
        num_frames=args.num_frames,
        max_skip=args.max_skip,
    )

    pipeline = MavisInferencePipeline(
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_config=args.sam2_config,
        qwen_model_name=args.qwen_model_name,
    )

    runner = BenchmarkRunner(
        pipeline=pipeline,
        dataset_name=args.dataset,
        prediction_root=args.prediction_root,
        metadata_root=args.metadata_root,
        num_summary_samples=args.num_summary_samples,
        top_k=args.top_k,
        threshold=args.threshold,
    )
    summary = runner.run_dataset(dataset, max_samples=args.max_samples, split=args.split)
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
