# v4.0 - reflector循环
"""
Milestone 4: VMA Loop 编排器

【老师】讲解
------------
Loop Orchestrator 把 Tagger、Verifier、Reflector 串成一个自主循环：

    Tagger -> Verifier -> Reflector -> (stop?) -> Tagger(partial)

设计要点：
1. 状态持久化：每轮结果写入 state，避免丢失进度
2. 停止条件：分数达标 或 达到最大轮次
3. 只把 correction_targets 涉及的字段传给 Tagger，不重复标注全部字段
4. 返回完整 trace，便于 debug 和报表
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from agents.reflector import Reflector
from agents.tagger import Tagger
from agents.stop_arbiter import StopArbiter
from agents.verifier import VerificationReport, VerifierPipeline
from core.schema import Annotation
from core.state import StateManager


class LoopTurn(BaseModel):
    """单轮循环的状态记录。"""

    turn: int
    annotation: Annotation | None
    raw_text: str
    report: VerificationReport
    reflection: dict[str, Any] | None
    stop_decision: dict[str, Any] | None
    error: str | None


class LoopResult(BaseModel):
    """整个循环的最终结果。"""

    clip_id: str
    success: bool
    final_annotation: Annotation | None
    final_report: VerificationReport | None
    turns: list[LoopTurn]
    stopped_by: Literal["quality", "max_turns", "error"] = "quality"
    total_turns: int


class VMALoop:
    """可验证多智能体标注循环。"""

    def __init__(
        self,
        tagger: Tagger | None = None,
        verifier: VerifierPipeline | None = None,
        reflector: Reflector | None = None,
        stop_arbiter: StopArbiter | None = None,
        state_manager: StateManager | None = None,
        max_turns: int = 3,
    ):
        self.tagger = tagger or Tagger()
        self.verifier = verifier or VerifierPipeline()
        self.reflector = reflector or Reflector(max_turns=max_turns)
        self.stop_arbiter = stop_arbiter or StopArbiter(max_turns=max_turns)
        self.state_manager = state_manager
        self.max_turns = max_turns

    def run(self, plan: dict, batch_id: str | None = None) -> LoopResult:
        """
        执行完整循环。

        Args:
            plan: Planner.plan() 的输出

        Returns:
            LoopResult 包含最终标注、每轮状态、停止原因
        """
        clip_id = plan["clip_id"]
        turns: list[LoopTurn] = []
        current_plan = plan

        for turn in range(1, self.max_turns + 1):
            tag_result = self.tagger.tag(current_plan)

            if not tag_result["success"]:
                turns.append(
                    LoopTurn(
                        turn=turn,
                        annotation=None,
                        raw_text=tag_result.get("raw_text", ""),
                        report=VerificationReport(
                            clip_id=clip_id,
                            verdicts=[],
                            coverage=0.0,
                            consistency=0.0,
                            plausibility=0.0,
                            overall_score=0.0,
                            stop_recommendation=True,
                        ),
                        reflection=None,
                        stop_decision=None,
                        error=tag_result["error"],
                    )
                )
                result = LoopResult(
                    clip_id=clip_id,
                    success=False,
                    final_annotation=None,
                    final_report=None,
                    turns=turns,
                    stopped_by="error",
                    total_turns=turn,
                )
                if self.state_manager and batch_id:
                    self.state_manager.save(batch_id, result)
                return result

            annotation = tag_result["annotation"]
            report = self.verifier.verify(annotation)
            stop_decision = self.stop_arbiter.decide(report, turn_count=turn)
            reflection = self.reflector.reflect(report, current_turn=turn)

            turns.append(
                LoopTurn(
                    turn=turn,
                    annotation=annotation,
                    raw_text=tag_result["raw_text"],
                    report=report,
                    reflection=reflection.model_dump(),
                    stop_decision=stop_decision.model_dump(),
                    error=None,
                )
            )

            if stop_decision.stop or not reflection.continue_loop:
                quality_passed = (
                    report.coverage >= self.stop_arbiter.coverage_threshold
                    and report.consistency >= self.stop_arbiter.consistency_threshold
                    and report.plausibility >= self.stop_arbiter.plausibility_threshold
                )
                stopped_by = "quality" if quality_passed else "max_turns"
                result = LoopResult(
                    clip_id=clip_id,
                    success=quality_passed,
                    final_annotation=annotation,
                    final_report=report,
                    turns=turns,
                    stopped_by=stopped_by,
                    total_turns=turn,
                )
                if self.state_manager and batch_id:
                    self.state_manager.save(batch_id, result)
                return result

            # 准备下一轮：只修正指定字段
            fields = [t.field for t in reflection.correction_targets]
            current_plan = {
                **current_plan,
                "partial_fields": fields,
            }

        # 如果循环正常结束（达到 max_turns 但未触发 early return）
        final_turn = turns[-1]
        result = LoopResult(
            clip_id=clip_id,
            success=final_turn.report.stop_recommendation,
            final_annotation=final_turn.annotation,
            final_report=final_turn.report,
            turns=turns,
            stopped_by="max_turns",
            total_turns=len(turns),
        )
        if self.state_manager and batch_id:
            self.state_manager.save(batch_id, result)
        return result


if __name__ == "__main__":
    # 本地测试：使用 mock 数据，不触发真实 API
    from agents.planner import Planner

    planner = Planner()
    # 构造一个最小 plan，使用空帧
    plan = planner.plan_from_frames(
        frames=[],
        clip_id="loop_test_001",
        duration_sec=1.0,
    )
    # 这里仅做接口可用性检查，真实运行需要 API
    print("VMALoop initialized. Use .run(plan) to execute.")
