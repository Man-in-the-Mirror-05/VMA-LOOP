# v2.0 - 单轮tagger
"""
Milestone 2: Tagger Agent

【老师】讲解
------------
Tagger 是 VMA-Loop 的「生成器」：
- 接收 Planner 准备好的 prompt 和 frames
- 调用 MultiModalClient 获取模型输出
- 解析 JSON，用 Pydantic schema 校验
- 返回 Annotation 对象或错误信息

关键原则：
1. 输出必须结构化（Pydantic）
2. 解析失败时要有明确的 fallback（至少返回 raw text 供 Reflector 诊断）
3. 不把所有状态塞进 LLM 上下文——只传当前轮次需要修正的字段
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from clients.multimodal_client import MultiModalClient
from core.schema import Annotation


class Tagger:
    """单轮标注生成器。"""

    def __init__(self, client: MultiModalClient | None = None):
        self.client = client or MultiModalClient()

    def tag(self, plan: dict, *, partial_fields: list[str] | None = None) -> dict[str, Any]:
        """
        根据 Planner 输出执行一次标注。

        Args:
            plan: Planner.plan() 的输出
            partial_fields: 可选。若指定，则只对缺失/错误的字段进行补充标注。

        Returns:
            {
                "success": bool,
                "annotation": Annotation | None,
                "raw_text": str,
                "error": str | None,
            }
        """
        frames = plan["frames"]
        system_prompt = plan["system_prompt"]
        user_prompt = plan["user_prompt"]

        if partial_fields:
            user_prompt += (
                "\n\n上一轮标注存在以下字段问题，请仅针对这些字段重新给出正确值，"
                "并返回完整 JSON：\n- " + "\n- ".join(partial_fields)
            )

        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        try:
            raw_text = self.client.annotate(full_prompt, frames)
        except Exception as e:
            return {
                "success": False,
                "annotation": None,
                "raw_text": "",
                "error": f"API 调用失败: {e}",
            }

        # 尝试从模型输出中提取 JSON
        parsed = self._extract_json(raw_text)
        if parsed is None:
            return {
                "success": False,
                "annotation": None,
                "raw_text": raw_text,
                "error": "无法从模型输出中解析 JSON",
            }

        # 注入 clip_id 和 duration_sec（防止模型编造）
        parsed["clip_id"] = plan["clip_id"]
        parsed["duration_sec"] = plan["duration_sec"]

        try:
            annotation = Annotation.model_validate(parsed)
            return {
                "success": True,
                "annotation": annotation,
                "raw_text": raw_text,
                "error": None,
            }
        except Exception as e:
            return {
                "success": False,
                "annotation": None,
                "raw_text": raw_text,
                "error": f"Schema 校验失败: {e}",
            }

    @staticmethod
    def _extract_json(text: str) -> dict | None:
        """
        从模型输出中提取 JSON。
        支持：1) markdown 代码块；2) 第一个 {...} 块。
        """
        # 先尝试匹配 markdown JSON 块
        code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if code_block_match:
            candidate = code_block_match.group(1).strip()
        else:
            # 找第一个 { 和最后一个 }
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return None
            candidate = text[start : end + 1]

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None


if __name__ == "__main__":
    # 仅做解析逻辑测试，不触发真实 API
    tagger = Tagger.__new__(Tagger)
    sample = '{"clip_id": "x", "duration_sec": 1.0, "description": "test", "actions": [], "overall_confidence": 0.9}'
    print(tagger._extract_json(f"```json\n{sample}\n```"))
