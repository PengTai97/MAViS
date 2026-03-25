from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from sam2.build_sam import build_sam2_video_predictor

from video_summarization_agent import VideoSummarizationAgent, load_qwen2_5_vl_model
from keyframe_selection_agent import KeyframeSelectionAgent
from object_grounding_agent import ObjectGroundingAgent


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


def list_video_frames(video_dir: str | Path) -> List[str]:
    video_dir = Path(video_dir)
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    frames: List[str] = []
    for ext in exts:
        frames.extend(str(p.resolve()) for p in sorted(video_dir.glob(ext)))
    if not frames:
        raise FileNotFoundError(f"No frames found under: {video_dir}")
    return frames


def uniform_sample_indices(num_frames: int, num_samples: int = 10) -> List[int]:
    if num_frames <= 0:
        return []
    if num_frames <= num_samples:
        return list(range(num_frames))
    return list(np.linspace(0, num_frames - 1, num_samples, dtype=int))


def get_image_size(image_path: str) -> tuple[int, int]:
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
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state,
        reverse=True,
    ):
        if out_frame_idx < start_frame:
            break
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    return video_segments


class MavisInferencePipeline:
    """Single-file inference pipeline that wires the three extracted agents together."""

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
        video_dir: str,
        description: str,
        num_summary_samples: int = 10,
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> MavisResult:
        all_frame_paths = list_video_frames(video_dir)
        all_indices = list(range(len(all_frame_paths)))
        sampled_indices = uniform_sample_indices(len(all_frame_paths), num_summary_samples)
        sampled_paths = [all_frame_paths[idx] for idx in sampled_indices]

        summary, summary_history = self.vs_agent(sampled_paths, description)
        ks_result = self.ks_agent(
            frame_paths=all_frame_paths,
            description=description,
            video_summary=summary,
            top_k=top_k,
            threshold=threshold,
            frame_indices=all_indices,
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
            boxes_by_frame[key_idx] = boxes
            if boxes:
                any_box_added = add_boxes_to_sam2(
                    predictor=self.predictor,
                    inference_state=inference_state,
                    frame_idx=key_idx,
                    boxes_xyxy=boxes,
                ) or any_box_added

        masks_by_frame = (
            propagate_masks_bidirectionally(
                predictor=self.predictor,
                inference_state=inference_state,
                start_frame=ks_result["start_frame"],
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


def save_result_json(result: MavisResult, output_json: str):
    serializable = asdict(result)
    serializable["masks_by_frame"] = {
        int(frame_idx): {int(obj_id): None for obj_id in obj_dict.keys()}
        for frame_idx, obj_dict in result.masks_by_frame.items()
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", required=True, help="Directory containing ordered video frames.")
    parser.add_argument("--description", required=True)
    parser.add_argument("--sam2_checkpoint", required=True)
    parser.add_argument("--sam2_config", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--num_summary_samples", type=int, default=10)
    parser.add_argument("--output_json", default="mavis_result.json")
    return parser


if __name__ == "__main__":
    args = _build_cli().parse_args()
    pipeline = MavisInferencePipeline(
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_config=args.sam2_config,
    )
    result = pipeline.run(
        video_dir=args.video_dir,
        description=args.description,
        top_k=args.top_k,
        threshold=args.threshold,
        num_summary_samples=args.num_summary_samples,
    )
    save_result_json(result, args.output_json)
    print(json.dumps({
        "summary": result.summary,
        "start_frame": result.start_frame,
        "key_frame_indices": result.key_frame_indices,
        "boxes_by_frame": result.boxes_by_frame,
        "output_json": args.output_json,
    }, ensure_ascii=False, indent=2))
