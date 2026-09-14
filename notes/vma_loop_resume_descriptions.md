# VMA-Loop 项目简历描述（STAR 法则多版本）

> 基于 `vma_loop` 可验证多智能体视频标注系统编写，所有描述均可在代码中找到对应实现。
> 使用说明：根据目标岗位复制对应版本到简历中，删除其他版本。
> 最近更新：2026-07-18，已纳入 P0 真实数据接入、P1 LLM verifier 插件、P2 Reflector before/after 报告。

---

## 通用 STAR 骨架（所有版本共用的事实基础）

| 维度 | 内容 |
|------|------|
| **S（背景）** | 2026 年个人项目，面向视频结构化标注场景，需将多模态数据生产经验沉淀为可独立运行、可验证、可迭代的 Agent 系统。 |
| **T（任务）** | 构建一个无需人工逐条干预、带 generator-verifier-reflector-stop arbiter 的自主视频标注系统；进一步接入真实开源视频数据、补充规则覆盖不到的语义 verifier、增强报告可解释性。 |
| **A（行动）** | 1）设计 Pydantic 结构化 schema 与三类规则 verifier；2）封装供应商无关 MultiModalClient 接入百炼 API；3）实现 Reflector 定向修正与 Stop Arbiter 自动停止；4）构建 Batch Pipeline 与 Streamlit 可视化报表；5）用 JSONL 做状态持久化支持断点续跑；6）尝试接入 Something-Something V2 真实数据集；7）新增 LLM 语义 verifier 插件；8）在 Markdown 报告中增加 Reflector before/after 对比。 |
| **R（结果）** | 在 20 个高保真机器人操作合成视频 clip 的 batch 上达到 100% 标注成功率，平均迭代 1.05 轮，1 个 clip 触发 Reflector 修正；41 个单元测试全部通过；真实数据自动下载受 legacy loading script / 官网授权限制，已给出完整手动下载路径与失败日志。 |

---

## 版本1：Agent / 大模型算法岗（核心推荐）

**项目标题**：VMA-Loop：可验证多智能体视频结构化标注系统
**岗位关键词匹配**：Agent / Multi-Agent / Loop Engineering / LLM-as-a-Judge / VLM / Verifier / Reflector / 语义一致性

**STAR 描述**：

- **S**：在 2026 年个人项目中，希望把「写脚本调用 VLM 做单次标注」升级为「设计带 verifier 自迭代、可解释、可观测的自主标注系统」，以匹配大模型算法 / Agent 开发实习岗位的能力要求。
- **T**：构建一个可独立运行的多 Agent 视频结构化标注引擎，核心挑战是：云端多模态 API 成本高、模型输出不稳定、标注质量难以量化、规则 verifier 无法覆盖语义等价判断。
- **A**：
  1. 设计 **Generator-Verifier-Reflector-Stop Arbiter** 四层循环：Tagger 生成标注、Verifier 输出 coverage/consistency/plausibility 三维分数、Reflector 定位需修正字段、Stop Arbiter 决定是否停止；
  2. 实现 **三类规则 verifier**（schema completeness / cross-field consistency / physical plausibility），用规则做 anchor 替代纯 LLM-as-a-Judge，降低成本与偏差；
  3. 新增 **LLMSemanticVerifier 插件**：通过纯文本 prompt 判断 description 与 actions 的语义一致性，处理「描述与动作对象/动词/空间关系不一致」等规则无法形式化的问题；默认禁用，API 失败时安全降级；
  4. 封装 **MultiModalClient**：统一 `annotate(video, prompt, schema)` 接口，接入百炼 OpenAI 兼容接口，内置指数退避重试、超时与限流；
  5. 构建 **Batch Pipeline + Streamlit Dashboard**：自动发现 clip、跳过已处理项、异常独立失败；生成 Markdown 报告并新增「Reflector 多轮修正案例（before / after）」章节，展示每轮评分、反馈与 Reflector 指令。
- **R**：20 个高保真机器人操作 clip 的 batch 成功率 100%，平均迭代 1.05 轮，1 个 clip 触发 Reflector 将 consistency 从 0.400 修正到 1.0；41 个单元测试全部通过；系统从单次 prompt 调用升级为可配置、可验证、可观测的 Agentic 标注平台。

**一句话版本（放简历顶部）**：
> 设计并实现 VMA-Loop 可验证多智能体视频标注系统，构建 Generator-Verifier-Reflector-Stop Arbiter 四层循环、三类规则 verifier 与 LLM 语义 verifier 插件，20 个 clip demo 成功率 100%。

---

## 版本2：数据工程 / 数据生产岗

**项目标题**：VMA-Loop：可验证多模态视频数据生产 Pipeline
**岗位关键词匹配**：数据 Pipeline / 结构化标注 / Schema 设计 / Pydantic / 状态持久化 / 质检 / 真实数据接入

**STAR 描述**：

- **S**：视频数据生产是多模态大模型训练的基础环节，传统人工标注成本高、一致性差；同时真实开源视频数据集（如 Something-Something V2）的自动获取受授权与 legacy loading script 限制。
- **T**：构建一个端到端的视频结构化标注数据 Pipeline，要求输出符合 schema、可校验、可追溯；并预留真实数据接入路径，在自动下载不可行时给出明确手动指南。
- **A**：
  1. 设计 **Pydantic Annotation Schema**：定义 clip、action、object、segment、relationship 等实体，区分 required/optional 字段；
  2. 实现 **四类 verifier 质检层**：schema completeness、cross-field consistency、physical plausibility，以及可选的 LLM 语义一致性 verifier；
  3. 更新 `download_real_datasets.py`：新增 `--dataset ssv2 --mode guide/download`，尝试 HuggingFace 自动加载并捕获 `Dataset scripts are no longer supported` 失败，输出 Qualcomm 官网手动下载与分卷拼接指南；
  4. 构建 **JSONL 状态持久化**：每轮标注、verifier 报告、reflector 目标全部落盘，支持断点续跑与历史审计；
  5. 输出 **Markdown 质量报告**：汇总 coverage/consistency/plausibility 分布、迭代轮次分布，自动定位需人工复核的 case，并展示 Reflector before/after 修正过程。
- **R**：20 个 clip 的批量标注全部通过 schema 校验与 verifier 质检，数据可追溯、可恢复，报告可直接用于人工复核；真实数据自动下载边界被明确记录，手动路径清晰可执行。

**一句话版本**：
> 构建可验证视频结构化标注数据 Pipeline，设计 Pydantic schema 与四类 verifier 质检层，实现 JSONL 状态持久化、真实数据接入脚本与自动化质量报告。

---

## 版本3：后端 / 系统工程岗

**项目标题**：VMA-Loop：多模态 API 驱动的可扩展标注后端
**岗位关键词匹配**：后端架构 / OpenAI 兼容接口 / 重试限流 / 配置化 / 可观测 / Streamlit / 插件化

**STAR 描述**：

- **S**：多模态大模型 API 是视频标注的主力计算资源，但存在供应商差异、调用失败、成本不可控、质量判断主观等问题。
- **T**：设计一个可扩展、可切换供应商、高可用的多模态标注后端，支持批量处理、状态恢复、可视化监控，并以插件化方式扩展语义 verifier。
- **A**：
  1. 封装 **MultiModalClient**：统一接口 `annotate(video, prompt, schema)`，底层接入百炼 OpenAI 兼容接口，模型名 / base_url / key 全部走配置；
  2. 实现 **指数退避重试与超时**：最多 3 次重试、最大 30s 退避、60s 超时，避免单点失败拖垮批量任务；
  3. 设计 **插件化 verifier 架构**：`VerifierPipeline` 支持通过 `llm_verifier` 参数注入 `LLMSemanticVerifier`，默认不启用，保证核心流程稳定；
  4. 构建 **YAML 配置化体系**：schema、prompt、API 参数、阈值全部外置，支持测试阶段切换小模型降低成本；
  5. 开发 **Batch Pipeline + StateManager**：自动发现视频 clip、跳过已处理项、异常 clip 独立失败不影响整体；
  6. 输出 **Streamlit Dashboard + Markdown 报告**：实时查看成功率、分数分布、迭代轮次分布，以及 Reflector 修正前后的评分对比。
- **R**：20 个 clip 批量处理成功率 100%，1 个 clip 经 Reflector 修正后质量达标；41 个单元测试全部通过；系统具备高可用、可配置、可观测、可扩展能力。

**一句话版本**：
> 设计多模态 API 驱动的可扩展标注后端，封装供应商无关 Client、实现重试限流与 YAML 配置化，构建插件化 verifier、Batch Pipeline 与可观测报表。

---

## 版本4：多模态 / VLM 岗

**项目标题**：VMA-Loop：基于 VLM 的可验证视频理解 Agent
**岗位关键词匹配**：多模态 / VLM / 视频理解 / Qwen-VL / 结构化输出 / Agent / 语义一致性

**STAR 描述**：

- **S**：视频理解是多模态大模型的重要方向，但 VLM 输出不稳定，存在描述与动作语义不一致、时间/对象引用错误等问题，需要结构化的后处理与验证机制。
- **T**：构建一个基于云端 VLM 的视频结构化理解系统，将模型输出约束到预定义 schema，通过规则 + LLM 多层 verifier 识别错误，并通过多轮迭代提升标注质量。
- **A**：
  1. 接入 **百炼 Qwen-VL-Plus**：通过 OpenAI 兼容接口发送抽帧后的关键帧，而非完整视频，降低 token 成本；
  2. 设计 **视频预处理模块**：等比缩放、FPS 采样、最多 8 帧输入，控制 API 调用量；
  3. 实现 **结构化输出约束**：用 Pydantic schema + 系统 prompt 强制模型输出合法 JSON，避免自由文本；
  4. 构建 **Verifier-Reflector 闭环**：当 VLM 输出缺失字段、时间不合理或语义不一致时，自动定位问题并发起定向修正调用；
  5. 新增 **LLM 语义 verifier**：判断 description 与 actions 在动词、对象、空间关系上是否一致，覆盖规则无法处理的语义等价场景；
  6. 输出 **可视化报表 + Reflector before/after**：展示 VLM 标注结果、verifier 分数与迭代轮次，以及多轮修正前后的关键反馈。
- **R**：20 个高保真合成视频 clip 全部成功标注，1 个 clip 经 Reflector 修正对象引用缺失后通过；验证了 VLM + 规则 verifier + LLM 语义 verifier + 迭代修正的可行性。

**一句话版本**：
> 构建基于 Qwen-VL 的可验证视频理解 Agent，通过抽帧预处理、Pydantic 结构化约束、规则 verifier 与 LLM 语义 verifier 多层质检，提升标注质量。

---

## 版本5：通用精简版（适用于任何技术岗）

如果你不确定投什么岗，用这个最稳的版本：

> **VMA-Loop：可验证多智能体视频结构化标注系统**
>
> - 设计 Generator-Verifier-Reflector-Stop Arbiter 四层循环，实现无需人工逐条干预的自主视频标注；
> - 封装供应商无关 MultiModalClient，接入百炼 OpenAI 兼容接口，内置重试、超时与限流；
> - 实现三类规则 verifier（completeness / consistency / plausibility）与可选 LLM 语义 verifier 插件，量化并扩展标注质量判断；
> - 构建 Batch Pipeline + JSONL 状态持久化 + Streamlit 可视化报表，20 个高保真机器人操作 clip demo 成功率 100%，平均迭代 1.05 轮；
> - 尝试接入 Something-Something V2 真实数据集，记录自动下载限制并输出完整手动下载指南；
> - 增强 Markdown 报告，增加 Reflector 多轮修正 before/after 对比，提升可解释性；
> - 41 个单元测试全部通过，撰写覆盖 Loop Engineering、LLM-as-a-Judge 偏差校准、状态持久化等 9 个主题的学习笔记。

---

## 使用建议

1. **Agent / 大模型算法岗用版本1**，突出四层循环、规则 anchor + LLM verifier 插件、Reflector before/after。
2. **数据工程 / 数据生产岗用版本2**，突出 schema、四类 verifier 质检、真实数据接入、状态持久化。
3. **后端 / 系统工程岗用版本3**，突出 API 封装、重试限流、插件化架构、可观测报表。
4. **多模态 / VLM 岗用版本4**，突出视频预处理、VLM 结构化输出、语义一致性 verifier。
5. **海投 / 不确定用版本5**。

**注意**：所有描述均基于真实可运行代码，未夸大训练模型或实际业务数据规模。真实数据自动下载结果如实记录为失败，手动路径已给出。
