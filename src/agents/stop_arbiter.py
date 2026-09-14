# v5.0 - stop与状态持久化
"""
Milestone 5: Stop Arbiter

【老师】讲解
------------
Stop Arbiter 是循环的「终止裁判」：
- 根据 verifier 分数和当前轮次决定是否停止
- 把停止原因写清楚，便于后续审计
- 阈值可配置，与 verifier 解耦

停止条件（默认）：
    coverage >= 0.95
    and consistency >= 0.90
    and plausibility >= 0.85
    or turn_count >= max_turns
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from agents.verifier import VerificationReport


class StopDecision(BaseModel):
    """停止决策。"""

    stop: bool
    reason: str
    coverage: float
    consistency: float
    plausibility: float
    turn_count: int
    max_turns: int


class StopArbiter:
    """根据分数与轮次决定停止。"""

    def __init__(
        self,
        coverage_threshold: float = 0.95,
        consistency_threshold: float = 0.90,
        plausibility_threshold: float = 0.85,
        max_turns: int = 3,
    ):
        self.coverage_threshold = coverage_threshold
        self.consistency_threshold = consistency_threshold
        self.plausibility_threshold = plausibility_threshold
        self.max_turns = max_turns

    def decide(self, report: VerificationReport, turn_count: int) -> StopDecision:
        """
        判断是否停止。

        Args:
            report: verifier 汇总报告
            turn_count: 当前已执行轮次
        """
        quality_ok = (
            report.coverage >= self.coverage_threshold
            and report.consistency >= self.consistency_threshold
            and report.plausibility >= self.plausibility_threshold
        )

        if quality_ok:
            reason = "质量阈值达标，停止迭代。"
            stop = True
        elif turn_count >= self.max_turns:
            reason = f"达到最大轮次 {self.max_turns}，停止迭代。"
            stop = True
        else:
            reason = "质量未达标且未达最大轮次，继续迭代。"
            stop = False

        return StopDecision(
            stop=stop,
            reason=reason,
            coverage=report.coverage,
            consistency=report.consistency,
            plausibility=report.plausibility,
            turn_count=turn_count,
            max_turns=self.max_turns,
        )


if __name__ == "__main__":
    from agents.verifier import VerificationReport

    report = VerificationReport(
        clip_id="test",
        verdicts=[],
        coverage=0.96,
        consistency=0.91,
        plausibility=0.86,
        overall_score=0.91,
        stop_recommendation=True,
    )
    arbiter = StopArbiter()
    print(arbiter.decide(report, turn_count=1).model_dump_json(indent=2, ensure_ascii=False))
