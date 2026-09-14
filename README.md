# VMA-Loop：可验证多智能体标注引擎

> Verifiable Multi-Agent Annotation Loop for video structured labeling.

## 环境准备

```bash
conda create -n vma-loop python=3.12 -y
conda activate vma-loop
pip install -r requirements.txt
```

## API 配置

复制 `.env.example` 为 `.env` 并填入你的百炼 API key：

```bash
cp .env.example .env
# 编辑 .env
```

当前默认模型为 `qwen-vl-plus`（已验证可用）。

## 运行测试

```bash
PYTHONPATH=src pytest tests/ -v
```

## 一键运行 Batch Pipeline

```bash
# 方案 A：使用高保真机器人操作合成视频（无需下载，立即验证）
PYTHONPATH=src python src/utils/generate_realistic_robot_clips.py --num 20 --out data/clips
PYTHONPATH=src python src/pipeline.py robot_batch --max-clips 20

# 方案 B：使用真实开源数据集（SSv2 需手动下载，详见指南）
# 1) 查看 SSv2 手动下载指南
PYTHONPATH=src python src/utils/download_real_datasets.py --mode guide --dataset ssv2
# 2) 按指南下载并放入 data/manual_clips/ 后复制到 data/clips
PYTHONPATH=src python src/utils/download_real_datasets.py --mode copy --src data/manual_clips --dst data/clips
# 3) 运行真实视频 batch
PYTHONPATH=src python src/pipeline.py real_batch --max-clips 20
```

## 验证结果

| 视频来源 | clip 数 | 成功率 | 平均迭代轮数 | Reflector 触发 | 说明 |
|---------|--------|--------|-------------|---------------|------|
| 早期合成视频（纯色方块） | 20 | 100% | 1.00 | 0 | 场景过于简单，无 Reflector 触发 |
| 高保真机器人操作合成视频 | 20 | 100% | 1.05 | 1 | Reflector 修正对象引用缺失；Markdown 报告新增 before/after |

最新报告：`outputs/reports/robot_batch_report.md`

## 近期改进

- **P0 真实数据接入**：`download_real_datasets.py` 新增 `--dataset ssv2 --mode guide/download`。已验证 HuggingFace `HuggingFaceM4/something_something_v2` 因 legacy loading script 无法在 `datasets>=3` 中自动加载；脚本会打印详细手动下载指南（Qualcomm 官网、19 个分卷拼接、validation.json 使用）。
- **P1 LLM verifier 插件**：`src/agents/verifier.py` 新增 `LLMSemanticVerifier`，通过纯文本 prompt 判断 description 与 actions 是否语义一致。默认不启用，可作为 `VerifierPipeline(llm_verifier=...)` 注入；API 失败时安全降级，不阻塞流水线。
- **P2 报告增强**：`src/pipeline.py` 的 Markdown 报告新增「Reflector 多轮修正案例（before / after）」章节，对 `total_turns > 1` 的 clip 输出每轮 coverage/consistency/plausibility/overall 评分、关键反馈与 Reflector 指令。
- **代码质量**：41 个单元测试覆盖所有核心模块；`.gitignore` 新增 `outputs_*/` 防止备份污染。

## 可视化 Dashboard

```bash
PYTHONPATH=src streamlit run src/dashboard.py
```

## 项目结构

```
vma_loop/
├── src/
│   ├── agents/          # Planner, Tagger, Verifier, Reflector, Stop Arbiter
│   ├── clients/         # MultiModalClient（百炼 OpenAI 兼容接口）
│   ├── core/            # Schema, Loop, State Manager
│   ├── utils/           # 视频预处理与测试数据生成
│   ├── pipeline.py      # 批量流水线
│   └── dashboard.py     # Streamlit 可视化
├── configs/             # YAML 配置
├── prompts/             # 模型 prompt
├── tests/               # 单元测试
├── outputs/             # 输出目录
├── data/                # 样本数据与视频 clip
└── notes/               # 学习笔记
```

## 里程碑

- [x] M1: Schema 与样本数据
- [x] M2: MultiModalClient 与单轮 Tagger
- [x] M3: Verifier 框架
- [x] M4: Reflector 循环
- [x] M5: Stop Arbiter 与状态持久化
- [x] M6: Batch Pipeline 与可视化报表
