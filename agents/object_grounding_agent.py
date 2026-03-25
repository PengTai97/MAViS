from __future__ import annotations

import argparse
import ast
import copy
import json
import re
from typing import Any, Dict, List, Sequence

from qwen_vl_utils import process_vision_info


Conversation = List[Dict[str, Any]]


def qwen2_5_vl_generate_multi_turn(
    model,
    processor,
    conversation_history: Conversation,
    new_user_message: Dict[str, Any],
    mode: str = "image",
    max_new_tokens: int = 512,
    temperature: float = 0.0,
):
    conversation_history = list(conversation_history)
    conversation_history.append(new_user_message)
    messages_for_inference = [conversation_history]

    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages_for_inference
    ]
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages_for_inference,
        return_video_kwargs=True,
    )
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    ).to(model.device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=temperature,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    if mode.lower() == "video":
        return output_text
    input_height = int(inputs["image_grid_thw"][0][1] * 14)
    input_width = int(inputs["image_grid_thw"][0][2] * 14)
    return output_text, input_height, input_width


def parse_bboxes_to_pixel_xyxy(
    image_height: int,
    image_width: int,
    raw_response: str,
    input_width: int,
    input_height: int,
) -> List[List[int]]:
    lines = raw_response.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    cleaned_text = "\n".join(lines).strip()

    if "there are none" in cleaned_text.lower():
        return []

    parsed = None
    for parser in (
        lambda x: json.loads(x),
        lambda x: ast.literal_eval(x),
    ):
        try:
            parsed = parser(cleaned_text)
            break
        except Exception:
            pass

    if parsed is None:
        match = re.search(r"\[.*\]", cleaned_text, re.DOTALL)
        if match:
            extracted = match.group(0)
            for parser in (
                lambda x: json.loads(x),
                lambda x: ast.literal_eval(x),
            ):
                try:
                    parsed = parser(extracted)
                    break
                except Exception:
                    continue

    if not parsed:
        return []

    boxes: List[List[int]] = []
    for item in parsed:
        if not isinstance(item, dict) or "bbox_2d" not in item:
            continue
        x1 = int(item["bbox_2d"][0] / input_width * image_width)
        y1 = int(item["bbox_2d"][1] / input_height * image_height)
        x2 = int(item["bbox_2d"][2] / input_width * image_width)
        y2 = int(item["bbox_2d"][3] / input_height * image_height)
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        boxes.append([x1, y1, x2, y2])
    return boxes


class ObjectGroundingAgent:
    """MAViS Object Grounding Agent.

    It predicts bounding boxes for a selected keyframe and converts them to
    pixel-space xyxy boxes for SAM2 prompting.
    """

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    def __call__(
        self,
        image_path: str,
        description: str,
        image_width: int,
        image_height: int,
        conversation_history: Conversation | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> List[List[int]]:
        local_history = copy.deepcopy(conversation_history or [])
        local_history.insert(
            0,
            {
                "role": "system",
                "content": (
                    "As an AI assistant, you specialize in accurate image object detection, "
                    "delivering coordinates. Output a JSON list. If the target does not appear, "
                    "reply with 'There are none.'."
                ),
            },
        )
        user_message = {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {
                    "type": "text",
                    "text": (
                        f"Locate {description}, and report bbox coordinates in JSON format. "
                        "Use items like {'bbox_2d':[x1,y1,x2,y2]}."
                    ),
                },
            ],
        }
        response, input_height, input_width = qwen2_5_vl_generate_multi_turn(
            model=self.model,
            processor=self.processor,
            conversation_history=local_history,
            new_user_message=user_message,
            mode="image",
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        return parse_bboxes_to_pixel_xyxy(
            image_height=image_height,
            image_width=image_width,
            raw_response=response,
            input_width=input_width,
            input_height=input_height,
        )


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    return parser


if __name__ == "__main__":
    from video_summarization_agent import load_qwen2_5_vl_model

    args = _build_cli().parse_args()
    model, processor = load_qwen2_5_vl_model()
    agent = ObjectGroundingAgent(model, processor)
    print(agent(args.image, args.description, args.width, args.height))
