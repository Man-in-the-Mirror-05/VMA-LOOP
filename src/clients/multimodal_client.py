# v2.0 - 单轮tagger
"""
Milestone 2: 统一多模态客户端

【老师】讲解
------------
MultiModalClient 是 VMA-Loop 对外的「感官接口」：
- 负责把本地视频/图片编码成 API 能接受的格式
- 负责调用云端多模态大模型
- 负责重试、超时、限流、错误降级
- 对上暴露统一的 annotate(...) 接口，方便以后切换供应商

【学生】易错点
-------------
不要直接把整个视频文件上传（贵且慢），应先抽帧再发送。
不要把 API key 打印到日志。
"""

from __future__ import annotations

import base64
import io
import os
import time
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class MultiModalClient:
    """
    统一多模态 API 客户端。

    当前默认实现基于阿里云百炼 OpenAI 兼容接口，
    但接口设计保持供应商无关：
    - annotate(prompt, media_paths_or_frames, schema) -> dict
    """

    def __init__(self, config_path: str | Path | None = None):
        self.config = self._load_config(config_path)
        self.model = os.getenv("BAILIAN_MODEL", self.config.get("model"))
        self.base_url = os.getenv("BAILIAN_BASE_URL", self.config.get("base_url"))
        self.api_key = os.getenv("BAILIAN_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "BAILIAN_API_KEY 未设置。请将 key 写入 .env 文件并确保不提交到 git。"
            )

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.max_retries = self.config.get("retry", {}).get("max_retries", 3)
        self.backoff_base = self.config.get("retry", {}).get("backoff_base_sec", 2.0)
        self.max_backoff = self.config.get("retry", {}).get("max_backoff_sec", 30.0)
        self.timeout = self.config.get("timeout_sec", 60)

    def _load_config(self, config_path: str | Path | None) -> dict:
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "configs" / "api_config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @staticmethod
    def encode_image(image_path: str | Path) -> str:
        """将单张图片编码为 base64 JPEG data URL。"""
        with open(image_path, "rb") as f:
            data = f.read()
        encoded = base64.b64encode(data).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

    @staticmethod
    def encode_frame(pil_image) -> str:
        """将 PIL Image 编码为 base64 JPEG data URL。"""
        buffer = io.BytesIO()
        pil_image.convert("RGB").save(buffer, format="JPEG", quality=85)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

    def annotate(
        self,
        prompt: str,
        media: list[str | Path | Any] | str | Path | Any,
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """
        调用多模态模型。

        Args:
            prompt: 文本 prompt
            media: 图片路径列表、PIL Image 列表，或单张图片
            model: 可临时覆盖模型名称
            max_tokens: 最大输出 token 数
            temperature: 采样温度

        Returns:
            模型返回的字符串内容
        """
        if not isinstance(media, list):
            media = [media]

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for item in media:
            if isinstance(item, (str, Path)):
                content.append({"type": "image_url", "image_url": {"url": self.encode_image(item)}})
            else:
                # 假设是 PIL Image
                content.append({"type": "image_url", "image_url": {"url": self.encode_frame(item)}})

        model = model or self.model
        max_tokens = max_tokens or self.config.get("max_tokens", 2048)
        temperature = temperature if temperature is not None else self.config.get("temperature", 0.2)

        last_exception: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=self.timeout,
                )
                return response.choices[0].message.content
            except Exception as e:
                last_exception = e
                if attempt == self.max_retries:
                    break
                wait = min(self.backoff_base * (2**attempt), self.max_backoff)
                time.sleep(wait)

        raise RuntimeError(
            f"多模态 API 调用失败（已重试 {self.max_retries} 次）: {last_exception}"
        )

    def safe_api_key_hint(self) -> str:
        """仅返回 key 前 8 位，用于调试确认，不暴露完整 key。"""
        if not self.api_key:
            return "<未设置>"
        return self.api_key[:8] + "..."


if __name__ == "__main__":
    client = MultiModalClient()
    print("API key hint:", client.safe_api_key_hint())
    print("Model:", client.model)
    print("Base URL:", client.base_url)
