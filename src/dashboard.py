# v6.0 - batch与可视化
"""
Milestone 6: Streamlit 可视化报表

运行方式：
    conda activate vma-loop
    cd vma_loop
    streamlit run src/dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from core.state import StateManager


OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
REPORTS_DIR = OUTPUTS_DIR / "reports"
STATES_DIR = OUTPUTS_DIR / "states"


def load_latest_report() -> dict | None:
    """加载最新的报告 JSON。"""
    report_files = sorted(REPORTS_DIR.glob("*_report.json"))
    if not report_files:
        return None
    latest = report_files[-1]
    with open(latest, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    st.set_page_config(page_title="VMA-Loop Dashboard", layout="wide")
    st.title("VMA-Loop：可验证多智能体标注引擎")
    st.markdown("查看批量标注结果、verifier 分数分布与需要人工复核的 case。")

    report = load_latest_report()
    if report is None:
        st.warning("未找到报告文件。请先运行 `python src/pipeline.py`。")
        return

    # 顶部指标
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("总 clip 数", report["total_clips"])
    col2.metric("成功率", f"{report['success_rate']:.1%}")
    col3.metric("平均迭代轮数", f"{report['avg_turns']:.2f}")
    col4.metric("平均 overall score", f"{(report['avg_coverage'] + report['avg_consistency'] + report['avg_plausibility']) / 3:.3f}")

    # 分数分布
    st.subheader("Verifier 分数分布")
    df = pd.DataFrame(report["results"])
    if not df.empty:
        st.bar_chart(df[["coverage", "consistency", "plausibility"]].mean())
        st.line_chart(df[["coverage", "consistency", "plausibility", "overall_score"]])

    # 迭代轮数分布
    st.subheader("迭代轮数分布")
    if not df.empty:
        st.bar_chart(df["total_turns"].value_counts().sort_index())

    # 需要复核的 case
    st.subheader("最需要人工复核的 case")
    review_df = pd.DataFrame(report["review_cases"])
    if not review_df.empty:
        st.dataframe(review_df, use_container_width=True)

    # 详细结果表
    st.subheader("详细结果")
    st.dataframe(df, use_container_width=True)

    # 批次列表
    st.subheader("状态文件")
    sm = StateManager(STATES_DIR)
    batches = sm.list_batches()
    if batches:
        selected_batch = st.selectbox("选择批次", batches)
        records = sm.load(selected_batch)
        st.write(f"批次 `{selected_batch}` 共 {len(records)} 条记录")
        if records:
            st.json(records[0], expanded=False)
    else:
        st.info("暂无状态文件。")


if __name__ == "__main__":
    main()
