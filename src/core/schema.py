# v1.0 - schema与数据
"""
Milestone 1: 结构化 Schema 定义

【老师】讲解
------------
在 VMA-Loop 中，schema 是整个系统的「宪法」：
- Tagger 的输出必须能被 schema 校验
- Verifier 的检查目标是 schema 的完整性与一致性
- Reflector 的修正指令必须精确到 schema 字段
- Stop Arbiter 的分数要能从 schema 实例中计算出来

因此 schema 设计必须做到：
1. 字段含义清晰，机器和人都能看懂
2. required / optional 明确区分
3. 支持置信度、时间戳、对象引用等扩展
4. 用 Pydantic 做运行时校验，并生成 JSON Schema 约束大模型输出
"""

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, model_validator


class VideoSegment(BaseModel):
    """视频时间片段，用于描述动作或事件的起止。"""

    start_sec: float = Field(..., ge=0.0, description="片段开始时间（秒）")
    end_sec: float = Field(..., ge=0.0, description="片段结束时间（秒）")

    @model_validator(mode="after")
    def check_order(self):
        if self.end_sec < self.start_sec:
            raise ValueError("end_sec 必须大于等于 start_sec")
        return self


class BoundingBox(BaseModel):
    """2D 边界框，坐标为归一化到 [0, 1] 的浮点数。"""

    x1: float = Field(..., ge=0.0, le=1.0)
    y1: float = Field(..., ge=0.0, le=1.0)
    x2: float = Field(..., ge=0.0, le=1.0)
    y2: float = Field(..., ge=0.0, le=1.0)

    @model_validator(mode="after")
    def check_box(self):
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("边界框坐标必须满足 x2>x1 且 y2>y1")
        return self


class PhysicalObject(BaseModel):
    """视频中出现的物理对象。"""

    object_id: str = Field(..., description="对象唯一标识，例如 obj_001")
    name: str = Field(..., description="对象名称，例如 '杯子' / 'knife'")
    category: Optional[str] = Field(None, description="对象类别")
    bbox: Optional[BoundingBox] = Field(None, description="可选的边界框")
    attributes: Optional[dict[str, Any]] = Field(
        default_factory=dict, description="附加属性，如颜色、状态等"
    )


class ActionItem(BaseModel):
    """一个动作/事件条目。"""

    action_id: str = Field(..., description="动作唯一标识")
    verb: str = Field(..., description="谓语动词，例如 'pour' / 'cut'")
    noun: Optional[str] = Field(None, description="动作作用对象")
    subject: Optional[str] = Field(None, description="动作执行主体")
    segment: VideoSegment = Field(..., description="动作发生的时间区间")
    involved_object_ids: list[str] = Field(
        default_factory=list,
        description="本动作涉及的对象 ID 列表，需与 objects 中的 object_id 对应",
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="模型对该动作的置信度")


class Annotation(BaseModel):
    """单个视频 clip 的完整结构化标注。这是 Tagger 的目标输出格式。"""

    clip_id: str = Field(..., min_length=1, description="视频片段唯一 ID")
    duration_sec: float = Field(..., ge=0.0, description="视频总时长（秒）")
    description: str = Field(..., description="对整个片段的自然语言描述")
    scene_context: Optional[str] = Field(None, description="场景上下文，如厨房、户外")
    objects: list[PhysicalObject] = Field(default_factory=list, description="检测到的对象")
    actions: list[ActionItem] = Field(default_factory=list, description="检测到的动作序列")
    relationships: Optional[list[dict[str, Any]]] = Field(
        default_factory=list, description="对象/动作之间的关系，如空间关系、因果关系"
    )
    overall_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="整段标注的总体置信度"
    )

    # 注：跨字段一致性（如对象引用、动作时间是否超出视频时长）
    # 由 CrossFieldConsistencyVerifier / PhysicalPlausibilityVerifier 负责，
    # 不在 Pydantic 层强制拦截，以便循环中保留中间状态供 verifier 诊断。

    @model_validator(mode="after")
    def check_action_within_duration(self):
        # 仅做软警告：如果时间明显异常，允许存储但记录
        return self

    @model_validator(mode="after")
    def check_object_references(self):
        # 软验证：不抛异常，让 verifier 处理语义一致性
        return self


class AnnotationBatch(BaseModel):
    """一批 clip 的标注结果。"""

    batch_id: str = Field(..., description="批次 ID")
    annotations: list[Annotation] = Field(default_factory=list)


# JSON Schema 导出函数，用于写进多模态模型 prompt 的 response_format 或 system message
def get_annotation_json_schema() -> dict[str, Any]:
    return Annotation.model_json_schema()


if __name__ == "__main__":
    import json

    print(json.dumps(get_annotation_json_schema(), indent=2, ensure_ascii=False))
