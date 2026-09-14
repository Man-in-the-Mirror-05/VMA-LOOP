# v5.0 - stop与状态持久化
"""
Milestone 5: State Manager

【老师】讲解
------------
State Manager 解决的是「不要把所有上下文交给 LLM」的问题：
- 每轮循环的状态（annotation、report、reflection）必须落盘
- 支持断点续跑：批量处理中途中断后可恢复
- 使用 JSONL：每行一个独立 JSON，便于追加和逐行读取

设计原则：
1. 写操作是追加，不是覆盖
2. 读操作可指定 batch_id / clip_id 过滤
3. 输出目录由配置决定，默认 outputs/states/
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class StateManager:
    """管理 VMA-Loop 的运行状态。"""

    def __init__(self, output_dir: str | Path | None = None):
        if output_dir is None:
            output_dir = Path(__file__).parent.parent.parent / "outputs" / "states"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _state_file(self, batch_id: str) -> Path:
        return self.output_dir / f"{batch_id}.jsonl"

    def save(self, batch_id: str, loop_result: Any) -> Path:
        """
        保存一个 LoopResult 到 JSONL。

        Args:
            batch_id: 批次 ID
            loop_result: LoopResult 或 dict

        Returns:
            写入的文件路径
        """
        path = self._state_file(batch_id)
        if hasattr(loop_result, "model_dump"):
            record = loop_result.model_dump()
        else:
            record = loop_result

        record["saved_at"] = datetime.now(timezone.utc).isoformat()
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return path

    def load(
        self,
        batch_id: str,
        *,
        clip_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        读取某批次的状态。

        Args:
            batch_id: 批次 ID
            clip_id: 可选，按 clip_id 过滤
            limit: 可选，最多返回条数
        """
        path = self._state_file(batch_id)
        if not path.exists():
            return []

        results: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if clip_id is not None and record.get("clip_id") != clip_id:
                    continue
                results.append(record)
                if limit is not None and len(results) >= limit:
                    break
        return results

    def list_batches(self) -> list[str]:
        """列出所有批次 ID。"""
        return sorted(p.stem for p in self.output_dir.glob("*.jsonl"))

    def clip_exists(self, batch_id: str, clip_id: str) -> bool:
        """检查某 clip 是否已处理过。"""
        records = self.load(batch_id, clip_id=clip_id, limit=1)
        return len(records) > 0


if __name__ == "__main__":
    sm = StateManager()
    print("Output dir:", sm.output_dir)
    print("Batches:", sm.list_batches())
