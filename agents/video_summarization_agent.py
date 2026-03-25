from __future__ import annotations

import argparse
from typing import Any, Dict, List, Sequence, Tuple

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


Conversation = List[Dict[str, Any]]


def load_qwen2_5_vl_model(
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    min_pixels: int = 384 * 384,
    max_pixels: int = 384 * 384,
):
    """Load Qwen2.5-VL once and reuse across all agents."""
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(
        model_name,
        use_fast=True,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    return model, processor


def qwen2_5_vl_generate_multi_turn(
    model,
    processor,
    conversation_history: Conversation,
    new_user_message: Dict[str, Any],
    mode: str = "image",
    max_new_tokens: int = 512,
    temperature: float = 0.0,
):
    """Shared helper kept inside this file so the agent can be called directly."""
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


class VideoSummarizationAgent:
    """MAViS Video Summarization Agent.

    Paper-aligned behavior:
    - input: uniformly sampled frames + text reference
    - output: 1-2 sentence semantic summary used by downstream agents
    """

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    def __call__(
        self,
        sampled_frame_paths: Sequence[str],
        description: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> Tuple[str, Conversation]:
        history: Conversation = [
            {
                "role": "system",
                "content": "You are a helpful video-understanding assistant.",
            }
        ]
        user_msg = {
            "role": "user",
            "content": [
                {"type": "video", "video": list(sampled_frame_paths)},
                {
                    "type": "text",
                    "text": (
                        f"The target is: {description}\n"
                        "In 1-2 sentences, summarize what the video actually shows about it "
                        "without merely restating the description."
                    ),
                },
            ],
        }
        summary = qwen2_5_vl_generate_multi_turn(
            model=self.model,
            processor=self.processor,
            conversation_history=history,
            new_user_message=user_msg,
            mode="video",
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        return summary.strip(), history + [{"role": "user", "content": user_msg["content"]}, {"role": "assistant", "content": summary}]


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--description", required=True)
    parser.add_argument("--frames", nargs="+", required=True)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    return parser


if __name__ == "__main__":
    args = _build_cli().parse_args()
    model, processor = load_qwen2_5_vl_model(args.model_name)
    agent = VideoSummarizationAgent(model, processor)
    summary, _ = agent(args.frames, args.description)
    print(summary)
