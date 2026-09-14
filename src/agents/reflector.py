# v4.0 - reflector循环
"""
Milestone 4: Reflector Agent

【老师】讲解
------------
Reflector 是 verifier 与 tagger 之间的「翻译官」：
- 输入：VerificationReport（包含每个 verifier 的分数和反馈）
- 输出：下一轮 Tagger 需要修正的字段列表 + 具体修正指令

关键设计：
1. 默认走规则反射，不调用 LLM（省钱、快速、可复现）
2. 只把 affected_fields 传给下一轮，避免让模型重新标注全部字段
3. 避免无限循环：Loop 自己维护 turn_count，Reflector 只负责生成目标

【学生】疑问
------------
Q: 为什么 Reflector 不直接自己修改 JSON？
A: 修改权应该保留给 Tagger（生成器），Reflector 只负责定位问题。
   这样符合 generator-verifier-reflector 的职责分离。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from agents.verifier import VerificationReport


class CorrectionTarget(BaseModel):
    """下一轮需要修正的目标。"""

    field: str = Field(..., description="字段路径，如 actions[0].verb")
    issue: str = Field(..., description="问题描述")
    instruction: str = Field(..., description="给 Tagger 的具体修正指令")
    priority: int = Field(default=1, ge=1, le=3, description="1=高，3=低")


class ReflectorOutput(BaseModel):
    """Reflector 的输出。"""

    reflection_summary: str
    correction_targets: list[CorrectionTarget]
    continue_loop: bool


class Reflector:
    """根据 verifier 反馈生成下一轮修正目标。"""

    def __init__(self, max_turns: int = 3):
        self.max_turns = max_turns

    def reflect(
        self,
        report: VerificationReport,
        current_turn: int,
        *,
        force_llm: bool = False,
    ) -> ReflectorOutput:
        """
        生成下一轮修正目标。

        Args:
            report: verifier 报告
            current_turn: 当前轮次（从 1 开始）
            force_llm: 是否强制使用 LLM 生成反射（默认 False）
        """
        if report.stop_recommendation:
            return ReflectorOutput(
                reflection_summary="当前标注已通过所有 verifier，无需修正。",
                correction_targets=[],
                continue_loop=False,
            )

        if current_turn >= self.max_turns:
            return ReflectorOutput(
                reflection_summary=f"已达到最大轮次 {self.max_turns}，停止迭代。",
                correction_targets=[],
                continue_loop=False,
            )

        targets: list[CorrectionTarget] = []
        for verdict in report.verdicts:
            if verdict.passed:
                continue
            for fb, field in zip(verdict.feedback, verdict.fields_affected):
                # 去重：相同 field 只保留一个 target
                if any(t.field == field for t in targets):
                    continue
                instruction = self._field_to_instruction(field, fb)
                targets.append(
                    CorrectionTarget(
                        field=field,
                        issue=fb,
                        instruction=instruction,
                        priority=1 if "缺失" in fb or "未定义" in fb else 2,
                    )
                )

        if not targets:
            return ReflectorOutput(
                reflection_summary="Verifier 未通过但没有定位到具体字段，终止循环。",
                correction_targets=[],
                continue_loop=False,
            )

        summary = (
            f"第 {current_turn} 轮 verifier 结果："
            f"coverage={report.coverage:.2f}, "
            f"consistency={report.consistency:.2f}, "
            f"plausibility={report.plausibility:.2f}。"
            f"需修正 {len(targets)} 个字段。"
        )

        return ReflectorOutput(
            reflection_summary=summary,
            correction_targets=targets,
            continue_loop=True,
        )

    @staticmethod
    def _field_to_instruction(field: str, issue: str) -> str:
        """把字段路径转成给 Tagger 的自然语言指令。"""
        if "description" in field:
            return "请重新观察视频帧，给出更完整、客观的自然语言描述。"
        if "actions" in field and "verb" in field:
            return "请核对动作动词是否准确，避免使用模糊动词。"
        if "actions" in field and "segment" in field:
            return "请核对动作发生的时间区间是否合理且不超过视频总时长。"
        if "involved_object_ids" in field:
            return "请确保动作涉及的对象都在 objects 中定义，引用使用正确的 object_id。"
        if "confidence" in field:
            return "请给出 0-1 之间的合理置信度。"
        return f"请修正以下问题：{issue}"


if __name__ == "__main__":
    from agents.verifier import VerifierPipeline
    from core.schema import Annotation
    import json
    from pathlib import Path

    sample = json.loads(Path("data/sample_annotation_good.json").read_text())
    ann = Annotation.model_validate(sample)
    # 注入一个错误
    ann.actions[0].involved_object_ids.append("ghost_obj")
    report = VerifierPipeline().verify(ann)
    reflector = Reflector()
    out = reflector.reflect(report, current_turn=1)
    print(out.model_dump_json(indent=2, ensure_ascii=False))
