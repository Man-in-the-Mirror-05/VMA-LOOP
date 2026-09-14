# v2.0 - 单轮tagger
"""
Milestone 2: Planner Agent

【老师】讲解
------------
Planner 负责把「一个视频 clip」加工成 Tagger 能消费的输入：
- 决定抽取多少帧、什么分辨率（控制成本）
- 构建最终 prompt（系统 prompt + 用户 prompt + JSON Schema 约束）
- 管理 clip 元数据，如 duration_sec

Planner 不调用模型，只负责编排输入。这样 Tagger 只关注「调用模型并解析输出」。
"""

from pathlib import Path

import yaml

from utils.video_utils import extract_frames, get_video_duration


class Planner:
    """规划一次标注任务：抽帧、拼 prompt、准备输入。"""

    def __init__(
        self,
        prompt_path: str | Path | None = None,
        *,
        target_fps: int | None = None,
        max_frames: int | None = None,
    ):
        if prompt_path is None:
            prompt_path = Path(__file__).parent.parent.parent / "prompts" / "tagger_prompt.yaml"
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_cfg = yaml.safe_load(f)
        self.system_prompt = prompt_cfg.get("system", "")
        self.user_prompt_template = prompt_cfg.get("user", "请根据以下视频帧给出结构化标注。")
        self.target_fps = target_fps
        self.max_frames = max_frames

    def plan(self, clip_path: str | Path, clip_id: str | None = None) -> dict:
        """
        返回一个 dict，包含 Tagger 需要的所有输入。
        """
        clip_path = Path(clip_path)
        if clip_id is None:
            clip_id = clip_path.stem

        duration_sec = get_video_duration(clip_path)
        frames = extract_frames(
            clip_path,
            target_fps=self.target_fps,
            max_frames=self.max_frames,
        )

        user_prompt = self.user_prompt_template.format(
            clip_id=clip_id,
            duration_sec=duration_sec,
        )

        return {
            "clip_id": clip_id,
            "duration_sec": duration_sec,
            "frames": frames,
            "system_prompt": self.system_prompt,
            "user_prompt": user_prompt,
        }

    def plan_from_frames(
        self,
        frames: list,
        clip_id: str,
        duration_sec: float,
    ) -> dict:
        """直接传入已抽好的帧，用于测试或批量服务。"""
        user_prompt = self.user_prompt_template.format(
            clip_id=clip_id,
            duration_sec=duration_sec,
        )
        return {
            "clip_id": clip_id,
            "duration_sec": duration_sec,
            "frames": frames,
            "system_prompt": self.system_prompt,
            "user_prompt": user_prompt,
        }
