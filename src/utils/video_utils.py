# v2.0 - 单轮tagger
"""
视频预处理工具：抽帧、缩放、质量压缩。
"""

from pathlib import Path

import cv2
import yaml
from PIL import Image


def load_api_config(config_path: str | Path | None = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "configs" / "api_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_frames(
    video_path: str | Path,
    *,
    target_fps: int | None = None,
    max_frames: int | None = None,
    resize_max_side: int | None = None,
) -> list[Image.Image]:
    """
    从视频中提取帧并返回 PIL Image 列表。

    Args:
        video_path: 视频文件路径
        target_fps: 目标采样帧率，None 则读取配置
        max_frames: 最大返回帧数，None 则读取配置
        resize_max_side: 缩放后长边最大像素
    """
    config = load_api_config().get("video_preprocessing", {})
    target_fps = target_fps or config.get("target_fps", 1)
    max_frames = max_frames or config.get("max_frames", 8)
    resize_max_side = resize_max_side or config.get("resize_max_side", 512)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0.0

    if duration_sec <= 0 or total_frames <= 0:
        cap.release()
        return []

    # 均匀采样：最多 max_frames 帧
    if total_frames <= max_frames:
        frame_indices = list(range(total_frames))
    else:
        step = total_frames / max_frames
        frame_indices = [int(i * step) for i in range(max_frames)]

    frames: list[Image.Image] = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        # BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame)
        # 等比缩放
        w, h = pil_image.size
        if max(w, h) > resize_max_side:
            scale = resize_max_side / max(w, h)
            new_size = (int(w * scale), int(h * scale))
            pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
        frames.append(pil_image)

    cap.release()
    return frames


def get_video_duration(video_path: str | Path) -> float:
    """获取视频时长（秒）。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames / fps if fps > 0 else 0.0


if __name__ == "__main__":
    # 用于本地测试，不依赖真实视频
    print("Video preprocessing utilities loaded.")
