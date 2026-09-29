# run Honeycomb video inference with a learned memory writer
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate video with Honeycomb.")
    parser.add_argument("--geometry-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-config", action="append", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--writer-checkpoint", type=Path, required=True)
    parser.add_argument("--vace-checkpoint", type=Path, required=True)
    parser.add_argument("--lora-checkpoint", type=Path)
    parser.add_argument("--lora-alpha", type=float, default=1.0)
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt")
    prompt.add_argument("--prompt-path", type=Path)
    parser.add_argument("--num-frames", type=int, default=33)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--infer-steps", type=int, default=40)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument(
        "--negative-prompt",
        default="bright colors, overexposed, static, blurred details, subtitles, style, artwork, painting, picture, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, still picture, cluttered background, three legs, many people in the background, walking backwards",
    )
    parser.add_argument(
        "--torch-dtype", choices=("bf16", "fp16", "fp32"), default="bf16"
    )
    parser.add_argument("--timestep-shift", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--max-reference-frames", type=int, default=8)
    parser.add_argument("--preceding-pixel-frames", type=int, default=8)
    parser.add_argument("--ref-iou-threshold", type=float, default=0.04)
    parser.add_argument("--ref-iou-voxel-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--tile-size", type=int, nargs=2, default=(30, 52))
    parser.add_argument("--tile-stride", type=int, nargs=2, default=(15, 26))
    parser.add_argument(
        "--depth-python",
        default=sys.executable,
        help="Python executable in the ViPE environment for depth estimation.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ("num_frames", "infer_steps", "fps"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    for name in ("start_frame", "max_reference_frames", "preceding_pixel_frames"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative.")
    if args.guidance_scale < 1.0:
        raise ValueError("--guidance-scale must be at least 1.")
    if (args.height is None) != (args.width is None):
        raise ValueError("--height and --width must be provided together.")


def load_pipeline_from_args(args: argparse.Namespace):
    import torch
    from diffsynth.core import ModelConfig
    from lsm.inference.pipeline import (
        HoneycombPipeline,
        InferenceConfig,
        load_lora_checkpoint,
        load_vace_checkpoint,
        validate_pipe,
    )
    from lsm.spatia.vace_init import build_scratch_vace_from_dit
    from lsm.spatia.wan_video_new import WanVideoPipeline

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        args.torch_dtype
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=dtype,
        device=device,
        model_configs=[ModelConfig(path=spec) for spec in args.model_config],
        tokenizer_config=ModelConfig(path=args.tokenizer_path),
    )
    if pipe.vace is None:
        if pipe.dit is None:
            raise ValueError("The model configuration did not load a DiT model.")
        pipe.vace = build_scratch_vace_from_dit(
            pipe.dit,
            use_reentrant=False,
            device=device,
            dtype=dtype,
        )
    load_vace_checkpoint(pipe, args.vace_checkpoint)
    validate_pipe(pipe)
    config = InferenceConfig(
        num_frames=args.num_frames,
        start_frame=args.start_frame,
        infer_steps=args.infer_steps,
        timestep_shift=args.timestep_shift,
        guidance_scale=args.guidance_scale,
        no_cfg=args.guidance_scale == 1.0,
        negative_prompt=args.negative_prompt,
        fps=args.fps,
        max_reference_frames=args.max_reference_frames,
        preceding_pixel_frames=args.preceding_pixel_frames,
        ref_iou_threshold=args.ref_iou_threshold,
        ref_iou_voxel_size=args.ref_iou_voxel_size,
        seed=args.seed,
        height=args.height,
        width=args.width,
        tiled=args.tiled,
        tile_size=tuple(args.tile_size),
        tile_stride=tuple(args.tile_stride),
        depth_python=args.depth_python,
    )
    runner = HoneycombPipeline(pipe, config)
    if args.lora_checkpoint is not None:
        runner.lora_state_dict = load_lora_checkpoint(
            pipe,
            args.lora_checkpoint,
            alpha=args.lora_alpha,
            fuse=False,
        )
        runner.lora_alpha = args.lora_alpha
    return runner


def main() -> None:
    args = parse_args()
    validate_args(args)

    import torch
    from adapter.rollout_memory import make_memory_factory
    from lsm.latent_point_cloud import LatentPointCloud
    from shared_writer.field import load_writer
    from recurrent_writer.incremental import load_incremental_writer

    prompt = (
        args.prompt_path.read_text(encoding="utf-8").strip()
        if args.prompt_path
        else args.prompt
    )
    checkpoint = torch.load(
        args.writer_checkpoint, map_location="cpu", weights_only=True
    )
    recurrent = "fusion" in checkpoint["args"]
    del checkpoint
    writer = (load_incremental_writer if recurrent else load_writer)(
        args.writer_checkpoint
    )
    backend = "recurrent" if recurrent else "replacement"
    pipeline = load_pipeline_from_args(args)
    created = []
    factory = make_memory_factory(LatentPointCloud, writer=writer)

    def capturing_factory(**kwargs):
        memory = factory(**kwargs)
        created.append(memory)
        return memory

    pipeline.memory_factory = capturing_factory
    pipeline.generate(
        geometry_path=args.geometry_path,
        prompt=prompt,
        output_dir=args.output_dir,
        run_metadata={
            "model_configs": args.model_config,
            "tokenizer_path": args.tokenizer_path,
            "vace_checkpoint": str(args.vace_checkpoint),
            "lora_checkpoint": str(args.lora_checkpoint)
            if args.lora_checkpoint
            else None,
            "memory_backend": backend,
            "memory_writer_ckpt": writer.ckpt_meta,
        },
    )
    if len(created) != 1:
        raise RuntimeError(f"Expected one memory instance, got {len(created)}.")
    memory = created[0]
    report = {
        "written_times": memory.written_times,
        "n_points": int(memory.points_world.shape[0]),
        "n_valid_points": int(memory.valid_mask.sum()),
        "num_time_steps": memory.num_time_steps,
        "bounds": memory.bounds.to_json(),
        "events": memory.events,
        "read_log": memory.read_log,
    }
    (args.output_dir / "hexmemory.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
