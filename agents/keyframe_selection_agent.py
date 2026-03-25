from __future__ import annotations

import argparse
import math
from typing import Any, Dict, List, Optional, Sequence

import torch
from qwen_vl_utils import process_vision_info


MIN_LOG = -1e9


def _apply_chat_template(processor, convos):
    texts = [processor.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in convos]
    img_inputs, _, video_kwargs = process_vision_info(convos, return_video_kwargs=True)
    return processor(
        text=texts,
        images=img_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )


def gather_single_token_ids(tokenizer, word: str) -> List[int]:
    variants = [
        f" {word}", word, f"\n{word}",
        f" {word.lower()}", word.lower(),
        f"▁{word}", f"▁{word.lower()}",
        f"Ġ{word}", f"Ġ{word.lower()}",
    ]
    ids = []
    for variant in variants:
        try:
            token_ids = tokenizer.encode(variant, add_special_tokens=False)
            if len(token_ids) == 1:
                ids.append(token_ids[0])
        except Exception:
            continue
    return sorted(set(ids))


def _group_logsumexp(last_logits: torch.Tensor, ids: List[int]) -> torch.Tensor:
    if not ids:
        return torch.full((last_logits.size(0),), MIN_LOG, device=last_logits.device)
    return torch.logsumexp(last_logits.float()[:, ids], dim=1)


class KeyframeSelectionAgent:
    """MAViS Keyframe Selection Agent using Binary-Logit Frame Scoring."""

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.yes_ids = gather_single_token_ids(processor.tokenizer, "Yes")
        self.no_ids = gather_single_token_ids(processor.tokenizer, "No")

    def score_frames(
        self,
        frame_paths: Sequence[str],
        description: str,
        video_summary: Optional[str] = None,
        chunk_size: int = 20,
    ) -> List[float]:
        if not frame_paths:
            return []

        all_scores: List[float] = []
        for start in range(0, len(frame_paths), chunk_size):
            chunk = list(frame_paths[start : start + chunk_size])
            convos = []
            for frame_path in chunk:
                user_content: List[Dict[str, Any]] = []
                if video_summary:
                    user_content.append({"type": "text", "text": f"Video summary: {video_summary}"})
                user_content.extend(
                    [
                        {"type": "image", "image": frame_path},
                        {
                            "type": "text",
                            "text": (
                                f"Description: '{description}'. "
                                "Does this frame contain all described objects? "
                                "Answer exactly 'Yes' or 'No'."
                            ),
                        },
                    ]
                )
                convos.append(
                    [
                        {
                            "role": "system",
                            "content": (
                                "You are a binary classification assistant. "
                                "Given one image frame and a description, answer exactly 'Yes' or 'No'."
                            ),
                        },
                        {"role": "user", "content": user_content},
                    ]
                )

            batch_inputs = _apply_chat_template(self.processor, convos).to(self.model.device)
            with torch.inference_mode():
                outputs = self.model(**batch_inputs, return_dict=True, use_cache=False)
            last_logits = outputs.logits[:, -1, :]
            lse_yes = _group_logsumexp(last_logits, self.yes_ids)
            lse_no = _group_logsumexp(last_logits, self.no_ids)
            scores = torch.nan_to_num(lse_yes - lse_no, nan=0.0, neginf=MIN_LOG, posinf=-MIN_LOG)
            all_scores.extend(scores.cpu().tolist())
        return all_scores

    def __call__(
        self,
        frame_paths: Sequence[str],
        description: str,
        video_summary: Optional[str] = None,
        top_k: int = 5,
        threshold: float = 0.0,
        frame_indices: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        if frame_indices is None:
            frame_indices = list(range(len(frame_paths)))
        if len(frame_indices) != len(frame_paths):
            raise ValueError("frame_indices and frame_paths must have the same length.")

        scores = self.score_frames(frame_paths, description, video_summary=video_summary)
        frame_scores = list(zip(frame_indices, scores))
        ranked = sorted(frame_scores, key=lambda x: x[1], reverse=True)
        ranked_indices = [idx for idx, _ in ranked]
        start_frame = next((idx for idx, score in zip(frame_indices, scores) if score > threshold), 0)
        return {
            "key_frame_indices": ranked_indices[:top_k],
            "ranked_indices": ranked_indices,
            "scores": scores,
            "start_frame": int(start_frame),
        }


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--description", required=True)
    parser.add_argument("--frames", nargs="+", required=True)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.0)
    return parser


if __name__ == "__main__":
    from video_summarization_agent import load_qwen2_5_vl_model

    args = _build_cli().parse_args()
    model, processor = load_qwen2_5_vl_model()
    agent = KeyframeSelectionAgent(model, processor)
    result = agent(
        frame_paths=args.frames,
        description=args.description,
        video_summary=args.summary,
        top_k=args.top_k,
        threshold=args.threshold,
    )
    print(result)
