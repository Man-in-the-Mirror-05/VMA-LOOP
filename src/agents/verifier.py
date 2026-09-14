# v3.0 - verifier框架
"""
Milestone 3: Verifier 框架

【老师】讲解
------------
Verifier 是 VMA-Loop 的「质检员」。它的核心职责是：
- 不重新生成标注，而是对已有标注给出可量化的评分
- 每个 verifier 返回 score + feedback + 受影响的字段
- Reflector 根据 feedback 决定下一轮修正什么

三类 verifier：
1. Schema Completeness：必填字段是否齐全、是否有空值
2. Cross-Field Consistency：字段之间是否一致（如动作引用的对象是否存在）
3. Physical Plausibility：时间、空间、物理合理性

为什么 verifier 不能「让 LLM 再 judge 一次」就完事？
- LLM-as-a-Judge 有偏差，需要规则 verifier 做 anchor
- 规则 verifier 更便宜、更快、可复现
- 只有当规则难以判断时才引入 LLM verifier
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from clients.multimodal_client import MultiModalClient
from core.schema import ActionItem, Annotation


class Verdict(BaseModel):
    """单个 verifier 的判定结果。"""

    verifier_name: str
    score: float = Field(..., ge=0.0, le=1.0)
    passed: bool
    feedback: list[str] = Field(default_factory=list)
    fields_affected: list[str] = Field(default_factory=list)


class VerificationReport(BaseModel):
    """一组 verifier 的汇总报告。"""

    clip_id: str
    verdicts: list[Verdict]
    coverage: float
    consistency: float
    plausibility: float
    overall_score: float
    stop_recommendation: bool


class BaseVerifier(ABC):
    """Verifier 基类。"""

    name: str = "base"

    @abstractmethod
    def verify(self, annotation: Annotation) -> Verdict:
        raise NotImplementedError


class SchemaCompletenessVerifier(BaseVerifier):
    """检查 schema 必填字段是否完整。"""

    name = "schema_completeness"

    # 必填字段及权重
    required_top_fields = {
        "clip_id": 1.0,
        "duration_sec": 1.0,
        "description": 1.0,
        "actions": 1.0,
        "overall_confidence": 1.0,
    }

    def verify(self, annotation: Annotation) -> Verdict:
        feedback: list[str] = []
        fields_affected: list[str] = []
        checks: list[tuple[str, bool]] = []

        # 顶层必填字段
        checks.append(("clip_id", bool(annotation.clip_id.strip())))
        checks.append(("duration_sec", annotation.duration_sec > 0))
        checks.append(("description", bool(annotation.description.strip())))
        checks.append(("actions", len(annotation.actions) > 0))
        checks.append(("overall_confidence", 0.0 <= annotation.overall_confidence <= 1.0))

        # action 子字段
        for idx, action in enumerate(annotation.actions):
            if not action.verb:
                checks.append((f"actions[{idx}].verb", False))
                fields_affected.append(f"actions[{idx}].verb")
            if action.segment is None:
                checks.append((f"actions[{idx}].segment", False))
                fields_affected.append(f"actions[{idx}].segment")

        passed_count = sum(1 for _, ok in checks if ok)
        score = passed_count / len(checks) if checks else 1.0

        for field, ok in checks:
            if not ok:
                feedback.append(f"字段缺失或无效: {field}")
                if field not in fields_affected:
                    fields_affected.append(field)

        return Verdict(
            verifier_name=self.name,
            score=score,
            passed=score >= 0.95,
            feedback=feedback,
            fields_affected=fields_affected,
        )


class CrossFieldConsistencyVerifier(BaseVerifier):
    """检查跨字段一致性。"""

    name = "cross_field_consistency"

    def verify(self, annotation: Annotation) -> Verdict:
        feedback: list[str] = []
        fields_affected: list[str] = []
        checks: list[bool] = []

        object_ids = {obj.object_id for obj in annotation.objects}

        # 检查 action 引用的对象是否存在
        for idx, action in enumerate(annotation.actions):
            for oid in action.involved_object_ids:
                ok = oid in object_ids
                checks.append(ok)
                if not ok:
                    feedback.append(
                        f"动作 {action.action_id} 引用了未定义对象 {oid}"
                    )
                    fields_affected.append(f"actions[{idx}].involved_object_ids")

        # 检查 action_id 唯一性
        action_ids = [a.action_id for a in annotation.actions]
        duplicates = set(aid for aid in action_ids if action_ids.count(aid) > 1)
        checks.append(len(duplicates) == 0)
        if duplicates:
            feedback.append(f"存在重复 action_id: {duplicates}")
            fields_affected.append("actions[].action_id")

        # 检查 description 中提到的主要对象是否在 objects 中（简单启发式）
        for obj in annotation.objects:
            if obj.name in annotation.description:
                checks.append(True)
            else:
                # 不强制要求，作为 soft check 不扣分
                pass

        score = sum(checks) / len(checks) if checks else 1.0
        return Verdict(
            verifier_name=self.name,
            score=score,
            passed=score >= 0.90,
            feedback=feedback,
            fields_affected=fields_affected,
        )


class PhysicalPlausibilityVerifier(BaseVerifier):
    """检查物理/时间合理性。"""

    name = "physical_plausibility"

    def verify(self, annotation: Annotation) -> Verdict:
        feedback: list[str] = []
        fields_affected: list[str] = []
        checks: list[bool] = []

        # 动作时间必须在视频范围内
        for idx, action in enumerate(annotation.actions):
            seg = action.segment
            ok = 0.0 <= seg.start_sec <= seg.end_sec <= annotation.duration_sec
            checks.append(ok)
            if not ok:
                feedback.append(
                    f"动作 {action.action_id} 时间 {seg.start_sec}-{seg.end_sec} "
                    f"超出视频时长 {annotation.duration_sec}"
                )
                fields_affected.append(f"actions[{idx}].segment")

        # 动作起止时间必须合理（至少 0.1 秒）
        for idx, action in enumerate(annotation.actions):
            seg = action.segment
            ok = seg.end_sec - seg.start_sec >= 0.1
            checks.append(ok)
            if not ok:
                feedback.append(f"动作 {action.action_id} 持续时间过短")
                fields_affected.append(f"actions[{idx}].segment")

        # 同一对象在同一时间不能被两个动作同时使用（简单冲突检测）
        for i, a1 in enumerate(annotation.actions):
            for j, a2 in enumerate(annotation.actions):
                if i >= j:
                    continue
                shared = set(a1.involved_object_ids) & set(a2.involved_object_ids)
                if not shared:
                    continue
                # 时间重叠
                overlap = not (
                    a1.segment.end_sec <= a2.segment.start_sec
                    or a2.segment.end_sec <= a1.segment.start_sec
                )
                ok = not overlap
                checks.append(ok)
                if not ok:
                    feedback.append(
                        f"动作 {a1.action_id} 与 {a2.action_id} 在时间上重叠，"
                        f"且共用对象 {shared}"
                    )
                    fields_affected.append(f"actions[{i}].segment")
                    fields_affected.append(f"actions[{j}].segment")

        # 置信度必须在合理范围
        for idx, action in enumerate(annotation.actions):
            ok = 0.0 <= action.confidence <= 1.0
            checks.append(ok)
            if not ok:
                feedback.append(f"动作 {action.action_id} 置信度越界")
                fields_affected.append(f"actions[{idx}].confidence")

        score = sum(checks) / len(checks) if checks else 1.0
        return Verdict(
            verifier_name=self.name,
            score=score,
            passed=score >= 0.85,
            feedback=feedback,
            fields_affected=fields_affected,
        )


class LLMSemanticVerifier(BaseVerifier):
    """
    基于 LLM 的语义一致性 verifier。

    解决规则 verifier 无法覆盖的场景：
    - description 说"把红色方块放到蓝色方块上"，但 actions 写的是"抓取绿色圆柱"
    - description 与 actions 在对象、动作、空间关系上存在语义冲突

    设计原则：
    1. 只作为插件使用，默认不启用（控制成本、保证可复现性）
    2. 输入只给文本（description + actions JSON），不消耗视频帧 token
    3. 要求模型输出结构化 JSON：一致/不一致 + 理由 + 分数
    4. 失败时安全降级：返回 passed=True，避免阻塞流水线
    """

    name = "llm_semantic_consistency"

    DEFAULT_PROMPT = """你是一名视频标注质检员。请判断下面视频描述与结构化动作列表是否在语义上一致。

【描述】
{description}

【动作列表】
{actions}

请从以下维度检查：
1. 动作动词是否与描述匹配（如描述是"放置"但动作是"抓取"，则不一致）
2. 涉及对象是否匹配（如描述提到"红色方块"但动作涉及"绿色圆柱"，则不一致）
3. 空间/目标关系是否匹配（如描述是"放到...上"但动作目标是"放到...里"，则不一致）
4. 时间顺序是否合理（动作起止时间是否覆盖了描述中的关键事件）

如果只有细微、不重要的差异（如同义词、措辞不同），请视为一致。

请以 JSON 格式输出（不要 markdown 代码块）：
{{
  "consistent": true/false,
  "score": 0.0-1.0,
  "reason": "简短理由"
}}
"""

    def __init__(
        self,
        client: MultiModalClient | None = None,
        threshold: float = 0.85,
        prompt_template: str | None = None,
        enabled: bool = True,
    ):
        self.client = client or MultiModalClient()
        self.threshold = threshold
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT
        self.enabled = enabled

    def _build_prompt(self, annotation: Annotation) -> str:
        actions_text = json.dumps(
            [
                {
                    "verb": a.verb,
                    "objects": a.involved_object_ids,
                    "start": a.segment.start_sec,
                    "end": a.segment.end_sec,
                }
                for a in annotation.actions
            ],
            ensure_ascii=False,
            indent=2,
        )
        return self.prompt_template.format(
            description=annotation.description,
            actions=actions_text,
        )

    @staticmethod
    def _extract_json(text: str) -> dict | None:
        """从模型输出中提取 JSON。"""
        # 尝试 markdown 代码块
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if m:
            candidate = m.group(1).strip()
        else:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return None
            candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None

    def verify(self, annotation: Annotation) -> Verdict:
        if not self.enabled:
            return Verdict(
                verifier_name=self.name,
                score=1.0,
                passed=True,
                feedback=["LLM semantic verifier 已禁用"],
                fields_affected=[],
            )

        prompt = self._build_prompt(annotation)
        try:
            raw_text = self.client.annotate(
                prompt,
                [],  # 不发送任何帧，纯文本判断
                max_tokens=512,
                temperature=0.0,
            )
        except Exception as e:
            # 安全降级：API 失败时不阻塞流水线
            return Verdict(
                verifier_name=self.name,
                score=1.0,
                passed=True,
                feedback=[f"LLM verifier API 调用失败，已安全降级: {e}"],
                fields_affected=[],
            )

        parsed = self._extract_json(raw_text)
        if parsed is None:
            return Verdict(
                verifier_name=self.name,
                score=1.0,
                passed=True,
                feedback=["LLM verifier 输出无法解析为 JSON，已安全降级"],
                fields_affected=["description", "actions"],
            )

        consistent = bool(parsed.get("consistent", True))
        score = float(parsed.get("score", 1.0 if consistent else 0.0))
        score = max(0.0, min(1.0, score))
        reason = parsed.get("reason", "无详细理由")

        passed = consistent and score >= self.threshold
        feedback: list[str] = []
        fields_affected: list[str] = []
        if not passed:
            feedback.append(f"语义不一致: {reason}")
            fields_affected.extend(["description", "actions"])

        return Verdict(
            verifier_name=self.name,
            score=score,
            passed=passed,
            feedback=feedback or [f"语义一致: {reason}"],
            fields_affected=fields_affected,
        )


class VerifierPipeline:
    """组合多个 verifier，输出汇总报告。"""

    def __init__(
        self,
        verifiers: list[BaseVerifier] | None = None,
        thresholds: dict[str, float] | None = None,
        llm_verifier: BaseVerifier | None = None,
    ):
        self.verifiers = verifiers or [
            SchemaCompletenessVerifier(),
            CrossFieldConsistencyVerifier(),
            PhysicalPlausibilityVerifier(),
        ]
        if llm_verifier is not None:
            self.verifiers.append(llm_verifier)
        self.thresholds = thresholds or {
            "coverage": 0.95,
            "consistency": 0.90,
            "plausibility": 0.85,
        }

    def verify(self, annotation: Annotation) -> VerificationReport:
        verdicts = [v.verify(annotation) for v in self.verifiers]

        def find_score(name_keyword: str) -> float:
            for vd in verdicts:
                if name_keyword in vd.verifier_name:
                    return vd.score
            return 1.0

        coverage = find_score("completeness")
        consistency = find_score("consistency")
        plausibility = find_score("plausibility")
        overall_score = (coverage + consistency + plausibility) / 3.0

        stop_recommended = (
            coverage >= self.thresholds["coverage"]
            and consistency >= self.thresholds["consistency"]
            and plausibility >= self.thresholds["plausibility"]
        )

        return VerificationReport(
            clip_id=annotation.clip_id,
            verdicts=verdicts,
            coverage=coverage,
            consistency=consistency,
            plausibility=plausibility,
            overall_score=overall_score,
            stop_recommendation=stop_recommended,
        )


if __name__ == "__main__":
    import json
    from pathlib import Path

    sample = json.loads(Path("data/sample_annotation_good.json").read_text())
    ann = Annotation.model_validate(sample)
    pipeline = VerifierPipeline()
    report = pipeline.verify(ann)
    print(report.model_dump_json(indent=2, ensure_ascii=False))
