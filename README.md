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
# 方案 A：真机机器人操作数据（LeRobot ALOHA，经 hf-mirror.com 自动下载）
# 默认 5 类任务各 4 条，共 20 个 clip；视频与 episodes 元数据自动缓存到 data/_real_cache
PYTHONPATH=src python src/utils/download_real_datasets.py --mode lerobot \
    --repo "lerobot/aloha_static_battery,lerobot/aloha_static_cups_open,lerobot/aloha_static_screw_driver,lerobot/aloha_static_towel,lerobot/aloha_static_ziploc_slide" \
    --max-clips 20 --max-duration 15
PYTHONPATH=src python src/pipeline.py real_batch --max-clips 20

# 方案 B：高保真机器人操作合成视频（无需下载，用于回归测试）
PYTHONPATH=src python src/utils/generate_realistic_robot_clips.py --num 20 --out data/clips
PYTHONPATH=src python src/pipeline.py robot_batch --max-clips 20

# 方案 C：手动下载的数据集（SSv2 需 Qualcomm 官网注册，详见指南）
PYTHONPATH=src python src/utils/download_real_datasets.py --mode guide --dataset ssv2
PYTHONPATH=src python src/utils/download_real_datasets.py --mode copy --src data/manual_clips --dst data/clips
PYTHONPATH=src python src/pipeline.py real_batch --max-clips 20
```

命名区分：LeRobot 的 `aloha_sim_*` 为 MuJoCo 仿真，`aloha_static_*` 与 `aloha_mobile_*` 为真机采集，本项目只取真机数据。

## 验证结果

| 视频来源 | clip 数 | 成功率 | 平均迭代轮数 | 平均 plausibility | Reflector 触发 | 说明 |
|---------|--------|--------|-------------|------------------|---------------|------|
| 早期合成视频（纯色方块） | 20 | 100% | 1.00 | 1.000 | 0 | 场景过于简单，无 Reflector 触发 |
| 高保真机器人操作合成视频 | 20 | 100% | 1.05 | 0.987 | 1 | 1 个 clip 触发 Reflector，consistency 0.400 → 1.0 |
| 真机 ALOHA 机器人操作视频（5 类任务） | 20 | 100% | 1.00 | 0.986 | 0 | 真机数据接入后一轮全部达标，验证闭环尚未被压测 |

最新报告：`outputs/reports/real_batch_report.md`（真机）、`outputs/reports/robot_batch_report.md`（合成）

## 近期改进

- **P0 真实数据接入（已完成）**：`download_real_datasets.py` 重写，新增 `--mode lerobot` 自动下载 LeRobot 真机机器人操作数据集。流程为读 `meta/info.json` 取 fps 与主视角视频键、读 `meta/episodes/*.parquet` 取每条 episode 的起止时间戳、下载对应 mp4 后按帧号精确切片，输出 10fps 独立 clip 与元数据 JSON；支持逗号分隔多仓库配额、缓存复用、旧 clip 自动归档。已跑通 5 类真机任务共 20 个 clip。SSv2 仍需 Qualcomm 官网注册（19 个分卷约 19.4GB），脚本保留手动指南。
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
