# v6.0 - batch与可视化
"""
Milestone 6: Batch Pipeline

【老师】讲解
------------
Batch Pipeline 把单个 clip 的 VMA-Loop 扩展为批量处理：
- 自动发现 data/clips/ 下的视频文件
- 使用 StateManager 跳过已处理 clip（断点续跑）
- 保存每个 clip 的最终标注到 outputs/annotations/
- 保存运行状态到 outputs/states/
- 生成汇总报告到 outputs/reports/

关键原则：
1. 每个 clip 独立处理，失败不影响其他 clip
2. 成本控制：批量处理前检查 clip 数量
3. 报告要定位最需要人工复核的 case
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from agents.planner import Planner
from agents.tagger import Tagger
from core.loop import VMALoop
from core.state import StateManager


SUPPORTED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


class BatchPipeline:
    """批量视频标注流水线。"""

    def __init__(
        self,
        clip_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        loop: VMALoop | None = None,
        planner: Planner | None = None,
        max_clips: int | None = None,
    ):
        if clip_dir is None:
            clip_dir = Path(__file__).parent.parent / "data" / "clips"
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "outputs"

        self.clip_dir = Path(clip_dir)
        self.output_dir = Path(output_dir)
        self.annotations_dir = self.output_dir / "annotations"
        self.states_dir = self.output_dir / "states"
        self.reports_dir = self.output_dir / "reports"
        for d in [self.annotations_dir, self.states_dir, self.reports_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.loop = loop or VMALoop(
            tagger=Tagger(),
            state_manager=StateManager(self.states_dir),
            max_turns=3,
        )
        self.planner = planner or Planner()
        self.max_clips = max_clips

    def discover_clips(self) -> list[Path]:
        """发现所有待处理的视频文件。"""
        clips = sorted(
            p
            for p in self.clip_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_VIDEO_EXTS
        )
        if self.max_clips is not None:
            clips = clips[: self.max_clips]
        return clips

    def run_batch(self, batch_id: str | None = None) -> dict[str, Any]:
        """
        执行批量处理。

        Args:
            batch_id: 批次 ID，默认使用时间戳

        Returns:
            汇总统计 dict
        """
        batch_id = batch_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        clips = self.discover_clips()
        print(f"[Batch {batch_id}] 发现 {len(clips)} 个待处理 clip")

        results_summary = []
        for clip_path in clips:
            clip_id = clip_path.stem
            if self.loop.state_manager and self.loop.state_manager.clip_exists(batch_id, clip_id):
                print(f"  跳过已处理 clip: {clip_id}")
                continue

            print(f"  处理 clip: {clip_id}")
            try:
                plan = self.planner.plan(clip_path, clip_id=clip_id)
                result = self.loop.run(plan, batch_id=batch_id)

                # 保存最终标注
                if result.final_annotation:
                    ann_path = self.annotations_dir / f"{clip_id}.json"
                    with open(ann_path, "w", encoding="utf-8") as f:
                        json.dump(
                            result.final_annotation.model_dump(),
                            f,
                            ensure_ascii=False,
                            indent=2,
                        )

                turn_summary = []
                if result.turns and result.total_turns > 1:
                    for t in result.turns:
                        turn_summary.append(
                            {
                                "turn": t.turn,
                                "coverage": t.report.coverage,
                                "consistency": t.report.consistency,
                                "plausibility": t.report.plausibility,
                                "overall_score": t.report.overall_score,
                                "feedback": [
                                    fb for vd in t.report.verdicts for fb in vd.feedback
                                ],
                                "fields_affected": [
                                    f
                                    for vd in t.report.verdicts
                                    for f in vd.fields_affected
                                ],
                                "reflection_summary": (
                                    t.reflection.get("reflection_summary", "")
                                    if t.reflection
                                    else ""
                                ),
                            }
                        )

                results_summary.append(
                    {
                        "clip_id": clip_id,
                        "success": result.success,
                        "total_turns": result.total_turns,
                        "stopped_by": result.stopped_by,
                        "coverage": result.final_report.coverage if result.final_report else 0.0,
                        "consistency": result.final_report.consistency if result.final_report else 0.0,
                        "plausibility": result.final_report.plausibility if result.final_report else 0.0,
                        "overall_score": result.final_report.overall_score if result.final_report else 0.0,
                        "turns": turn_summary,
                    }
                )
            except Exception as e:
                print(f"  处理 clip {clip_id} 失败: {e}")
                results_summary.append(
                    {
                        "clip_id": clip_id,
                        "success": False,
                        "total_turns": 0,
                        "stopped_by": "error",
                        "coverage": 0.0,
                        "consistency": 0.0,
                        "plausibility": 0.0,
                        "overall_score": 0.0,
                    }
                )

        summary = self._build_summary(batch_id, results_summary)
        report_path = self.reports_dir / f"{batch_id}_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # 同时生成 Markdown 报告
        md_path = self.reports_dir / f"{batch_id}_report.md"
        md_path.write_text(self._render_markdown(summary), encoding="utf-8")

        return summary

    def _build_summary(self, batch_id: str, results: list[dict]) -> dict[str, Any]:
        total = len(results)
        success = sum(1 for r in results if r["success"])
        failed = total - success
        avg_turns = sum(r["total_turns"] for r in results) / total if total else 0.0
        avg_coverage = sum(r["coverage"] for r in results) / total if total else 0.0
        avg_consistency = sum(r["consistency"] for r in results) / total if total else 0.0
        avg_plausibility = sum(r["plausibility"] for r in results) / total if total else 0.0

        # 最需要人工复核的 case：分数低且未成功
        review_cases = sorted(
            results,
            key=lambda r: (r["success"], r["overall_score"]),
        )[:5]

        return {
            "batch_id": batch_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_clips": total,
            "success_count": success,
            "failed_count": failed,
            "success_rate": success / total if total else 0.0,
            "avg_turns": avg_turns,
            "avg_coverage": avg_coverage,
            "avg_consistency": avg_consistency,
            "avg_plausibility": avg_plausibility,
            "review_cases": review_cases,
            "results": results,
        }

    @staticmethod
    def _render_markdown(summary: dict) -> str:
        lines = [
            f"# VMA-Loop Batch Report: {summary['batch_id']}",
            "",
            f"- 生成时间：{summary['generated_at']}",
            f"- 总 clip 数：{summary['total_clips']}",
            f"- 成功：{summary['success_count']}",
            f"- 失败：{summary['failed_count']}",
            f"- 成功率：{summary['success_rate']:.2%}",
            f"- 平均迭代轮数：{summary['avg_turns']:.2f}",
            f"- 平均 coverage：{summary['avg_coverage']:.3f}",
            f"- 平均 consistency：{summary['avg_consistency']:.3f}",
            f"- 平均 plausibility：{summary['avg_plausibility']:.3f}",
            "",
            "## 最需要人工复核的 case",
            "",
        ]
        for case in summary["review_cases"]:
            lines.append(
                f"- **{case['clip_id']}**: success={case['success']}, "
                f"overall={case['overall_score']:.3f}, "
                f"turns={case['total_turns']}, stopped_by={case['stopped_by']}"
            )

        # Reflector before/after 对比
        reflector_cases = [r for r in summary["results"] if r.get("turns") and len(r["turns"]) > 1]
        lines.append("")
        lines.append("## Reflector 多轮修正案例（before / after）")
        lines.append("")
        if not reflector_cases:
            lines.append("本次 batch 没有触发 Reflector 多轮迭代。")
        else:
            lines.append(
                f"共 {len(reflector_cases)} 个 clip 触发 Reflector，以下为每轮 verifier 评分与反馈对比。"
            )
            for r in reflector_cases:
                lines.append("")
                lines.append(f"### {r['clip_id']}")
                lines.append("")
                lines.append("| 轮次 | coverage | consistency | plausibility | overall | 关键反馈 | Reflector 指令 |")
                lines.append("|------|----------|-------------|--------------|---------|----------|----------------|")
                for t in r["turns"]:
                    feedback_short = "; ".join(t["feedback"])[:80] + "..." if len("; ".join(t["feedback"])) > 80 else "; ".join(t["feedback"])
                    reflection_short = t["reflection_summary"][:80] + "..." if len(t["reflection_summary"]) > 80 else t["reflection_summary"]
                    lines.append(
                        f"| {t['turn']} | {t['coverage']:.3f} | {t['consistency']:.3f} | "
                        f"{t['plausibility']:.3f} | {t['overall_score']:.3f} | {feedback_short} | {reflection_short} |"
                    )
                first_turn = r["turns"][0]
                final_turn = r["turns"][-1]
                lines.append("")
                lines.append("**修正效果：**")
                lines.append(
                    f"- 第 1 轮 plausibility={first_turn['plausibility']:.3f}，"
                    f"最终轮 plausibility={final_turn['plausibility']:.3f}"
                )
                lines.append(
                    f"- 第 1 轮反馈：{'; '.join(first_turn['feedback']) or '无'}"
                )
                lines.append(
                    f"- 最终轮反馈：{'; '.join(final_turn['feedback']) or '无'}"
                )

        lines.append("")
        lines.append("## 详细结果")
        lines.append("")
        lines.append("| clip_id | success | coverage | consistency | plausibility | turns | stopped_by |")
        lines.append("|---------|---------|----------|-------------|--------------|-------|------------|")
        for r in summary["results"]:
            lines.append(
                f"| {r['clip_id']} | {r['success']} | {r['coverage']:.3f} | "
                f"{r['consistency']:.3f} | {r['plausibility']:.3f} | "
                f"{r['total_turns']} | {r['stopped_by']} |"
            )
        lines.append("")
        return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VMA-Loop Batch Pipeline")
    parser.add_argument("batch_id", nargs="?", default="test_batch", help="批次 ID")
    parser.add_argument("--max-clips", type=int, default=None, help="最大处理 clip 数")
    args = parser.parse_args()

    pipeline = BatchPipeline(max_clips=args.max_clips)
    summary = pipeline.run_batch(args.batch_id)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
