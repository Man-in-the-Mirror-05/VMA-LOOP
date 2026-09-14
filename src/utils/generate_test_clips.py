# v6.0 - batch与可视化
"""
生成合成测试视频 clip，用于验证 batch pipeline。

运行方式：
    conda activate vma-loop
    cd vma_loop
    python src/utils/generate_test_clips.py --num 20 --out data/clips
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np


ACTIONS = [
    ("pour", "water"),
    ("cut", "bread"),
    ("stir", "soup"),
    ("open", "bottle"),
    ("close", "door"),
    ("pick", "cup"),
    ("put", "book"),
    ("wash", "plate"),
]


def generate_clip(
    output_path: Path,
    duration_sec: float = 3.0,
    fps: int = 5,
    width: int = 320,
    height: int = 240,
) -> None:
    """生成一个带简单动画的合成视频。"""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    total_frames = int(duration_sec * fps)
    action_verb, action_noun = random.choice(ACTIONS)

    for i in range(total_frames):
        # 背景色随帧变化
        bg_color = (i * 5 % 180 + 50, i * 3 % 180 + 50, 200)
        frame = np.full((height, width, 3), bg_color, dtype=np.uint8)

        # 移动方块模拟对象
        x = int((i / total_frames) * (width - 60)) + 30
        y = height // 2
        color = (0, 0, 255) if i % 2 == 0 else (255, 0, 0)
        cv2.rectangle(frame, (x, y - 30), (x + 40, y + 30), color, -1)

        # 叠加文字
        text = f"{action_verb} {action_noun}"
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        out.write(frame)

    out.release()


def main():
    parser = argparse.ArgumentParser(description="生成合成测试视频")
    parser.add_argument("--num", type=int, default=20, help="生成视频数量")
    parser.add_argument("--out", type=str, default="data/clips", help="输出目录")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.num):
        clip_path = out_dir / f"test_clip_{i+1:03d}.mp4"
        generate_clip(clip_path)
        print(f"Generated: {clip_path}")

    print(f"Done. Generated {args.num} clips in {out_dir}")


if __name__ == "__main__":
    main()
