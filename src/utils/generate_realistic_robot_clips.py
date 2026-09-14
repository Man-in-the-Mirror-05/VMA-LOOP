"""
生成更贴近真实机器人操作场景的合成视频。

相比早期版本（纯色背景 + 移动方块），本版本包含：
- 工作台背景与桌面纹理
- 二连杆机械臂 + 夹爪
- 多种几何形状与颜色的操作对象
- pick-and-place 轨迹与夹爪开合动画
- 简单的阴影效果

用于在无法下载真实开源机器人数据集时，作为更高保真的验证代理。
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import cv2
import numpy as np


OBJECT_SHAPES = ["cube", "cylinder", "sphere"]
OBJECT_COLORS = [
    (180, 120, 70),   # 蓝色
    (70, 180, 120),   # 绿色
    (120, 70, 180),   # 紫色
    (80, 160, 220),   # 橙色
    (200, 200, 80),   # 青色
]
ACTIONS = [
    "pick and place",
    "push",
    "stack",
    "slide",
    "rotate",
]


def draw_table(frame: np.ndarray) -> None:
    """绘制工作台背景。"""
    h, w = frame.shape[:2]
    # 墙面
    cv2.rectangle(frame, (0, 0), (w, int(h * 0.55)), (220, 220, 220), -1)
    # 桌面
    cv2.rectangle(frame, (0, int(h * 0.55)), (w, h), (180, 160, 140), -1)
    # 桌面边缘阴影
    cv2.line(frame, (0, int(h * 0.55)), (w, int(h * 0.55)), (140, 120, 100), 3)


def draw_robot_base(frame: np.ndarray, base_pos: tuple[int, int]) -> None:
    """绘制机械臂底座。"""
    x, y = base_pos
    cv2.rectangle(frame, (x - 40, y - 20), (x + 40, y + 20), (80, 80, 80), -1)
    cv2.rectangle(frame, (x - 40, y - 20), (x + 40, y + 20), (50, 50, 50), 2)


def draw_link(frame: np.ndarray, p1: tuple[int, int], p2: tuple[int, int], width: int = 12) -> None:
    """绘制机械臂连杆。"""
    cv2.line(frame, p1, p2, (200, 200, 200), width + 4)
    cv2.line(frame, p1, p2, (160, 160, 160), width)


def draw_gripper(frame: np.ndarray, pos: tuple[int, int], angle: float, open_width: int) -> None:
    """绘制夹爪。"""
    x, y = pos
    dx = int(math.cos(angle) * 25)
    dy = int(math.sin(angle) * 25)
    # 垂直于 angle 的方向
    perp_x = int(math.cos(angle + math.pi / 2) * open_width)
    perp_y = int(math.sin(angle + math.pi / 2) * open_width)

    # 左爪
    p1 = (x + perp_x, y + perp_y)
    p2 = (x + perp_x + dx, y + perp_y + dy)
    cv2.line(frame, p1, p2, (100, 100, 100), 6)

    # 右爪
    p3 = (x - perp_x, y - perp_y)
    p4 = (x - perp_x + dx, y - perp_y + dy)
    cv2.line(frame, p3, p4, (100, 100, 100), 6)


def draw_object(
    frame: np.ndarray,
    shape: str,
    pos: tuple[int, int],
    size: int,
    color: tuple[int, int, int],
) -> None:
    """绘制操作对象，带简单阴影。"""
    x, y = pos
    shadow_offset = 6
    shadow_color = (100, 100, 100)

    if shape == "cube":
        # 阴影
        cv2.rectangle(
            frame,
            (x - size + shadow_offset, y - size + shadow_offset),
            (x + size + shadow_offset, y + size + shadow_offset),
            shadow_color,
            -1,
        )
        cv2.rectangle(
            frame,
            (x - size, y - size),
            (x + size, y + size),
            color,
            -1,
        )
        cv2.rectangle(
            frame,
            (x - size, y - size),
            (x + size, y + size),
            tuple(max(0, c - 40) for c in color),
            2,
        )
    elif shape == "cylinder":
        cv2.ellipse(
            frame,
            (x + shadow_offset, y + size + shadow_offset),
            (size, size // 3),
            0,
            0,
            360,
            shadow_color,
            -1,
        )
        cv2.rectangle(frame, (x - size, y - size), (x + size, y + size), color, -1)
        cv2.ellipse(frame, (x, y - size), (size, size // 3), 0, 0, 360, color, -1)
        cv2.ellipse(
            frame,
            (x, y - size),
            (size, size // 3),
            0,
            0,
            360,
            tuple(max(0, c - 40) for c in color),
            2,
        )
    else:  # sphere
        cv2.circle(frame, (x + shadow_offset, y + shadow_offset), size, shadow_color, -1)
        cv2.circle(frame, (x, y), size, color, -1)
        # 高光
        cv2.circle(frame, (x - size // 3, y - size // 3), size // 4, (255, 255, 255), -1)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def generate_robot_clip(
    output_path: Path,
    duration_sec: float = 4.0,
    fps: int = 5,
    width: int = 480,
    height: int = 360,
) -> dict:
    """生成一个机器人 pick-and-place 操作视频。"""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    total_frames = int(duration_sec * fps)

    # 随机选择对象与动作
    shape = random.choice(OBJECT_SHAPES)
    color = random.choice(OBJECT_COLORS)
    action = random.choice(ACTIONS)
    obj_size = 25

    # 机械臂底座位置
    base_x = 80
    base_y = int(height * 0.55) - 10

    # 对象起始与目标位置
    start_x = random.randint(180, 260)
    start_y = int(height * 0.55) + 40 + obj_size
    target_x = random.randint(340, 420)
    target_y = int(height * 0.55) + 40 + obj_size

    # 轨迹关键点：接近 → 抓取 → 提升 → 移动 → 下降 → 放置 → 离开
    keyframes = [
        (start_x - 60, start_y - 80, 20),   # 接近
        (start_x, start_y, 8),               # 抓取
        (start_x, start_y - 60, 8),          # 提升
        (target_x, target_y - 60, 8),        # 移动
        (target_x, target_y, 8),             # 下降放置
        (target_x, target_y - 60, 20),       # 离开
        (target_x + 60, target_y - 80, 20),  # 归位
    ]

    holding_object = False
    obj_pos = (start_x, start_y)

    for frame_idx in range(total_frames):
        frame = np.full((height, width, 3), (240, 240, 240), dtype=np.uint8)
        draw_table(frame)
        draw_robot_base(frame, (base_x, base_y))

        # 计算当前进度对应的 keyframe 位置
        progress = frame_idx / (total_frames - 1) if total_frames > 1 else 1.0
        segment_count = len(keyframes) - 1
        segment_idx = min(int(progress * segment_count), segment_count - 1)
        local_t = progress * segment_count - segment_idx

        x1, y1, grip1 = keyframes[segment_idx]
        x2, y2, grip2 = keyframes[segment_idx + 1]
        cur_x = int(lerp(x1, x2, local_t))
        cur_y = int(lerp(y1, y2, local_t))
        cur_grip = int(lerp(grip1, grip2, local_t))

        # 连杆 IK 近似：肩关节 -> 肘关节 -> 腕关节
        shoulder = (base_x, base_y - 20)
        # 简单二连杆，肘部在中点上方拱起
        mid_x = (shoulder[0] + cur_x) // 2
        mid_y = min(shoulder[1], cur_y) - 80
        elbow = (mid_x, mid_y)

        draw_link(frame, shoulder, elbow)
        draw_link(frame, elbow, (cur_x, cur_y))
        draw_gripper(frame, (cur_x, cur_y), math.atan2(cur_y - elbow[1], cur_x - elbow[0]), cur_grip)

        # 更新对象位置：被抓取时跟随夹爪，否则在目标位置
        if segment_idx >= 2 and segment_idx <= 4:
            holding_object = True
        elif segment_idx > 4:
            holding_object = False
            obj_pos = (target_x, target_y)

        if holding_object:
            obj_pos = (cur_x, cur_y + obj_size + 5)

        draw_object(frame, shape, obj_pos, obj_size, color)

        # 文字说明
        cv2.putText(
            frame,
            f"Action: {action}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (50, 50, 50),
            2,
        )

        out.write(frame)

    out.release()

    return {
        "action": action,
        "shape": shape,
        "start": (start_x, start_y),
        "target": (target_x, target_y),
    }


def main():
    parser = argparse.ArgumentParser(description="生成高保真机器人操作合成视频")
    parser.add_argument("--num", type=int, default=20, help="生成视频数量")
    parser.add_argument("--out", type=str, default="data/clips", help="输出目录")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    for i in range(args.num):
        clip_path = out_dir / f"robot_clip_{i+1:03d}.mp4"
        info = generate_robot_clip(clip_path)
        info["path"] = str(clip_path)
        info["clip_id"] = clip_path.stem
        metadata.append(info)
        print(f"Generated: {clip_path} | action={info['action']} | shape={info['shape']}")

    # 保存元数据
    import json

    meta_path = out_dir / "robot_clips_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Metadata saved to: {meta_path}")
    print(f"Done. Generated {args.num} realistic robot clips in {out_dir}")


if __name__ == "__main__":
    main()
