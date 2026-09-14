"""
真实开源视频数据集下载脚本。

当前环境限制说明：
- HuggingFace Hub 无法直接访问（已尝试 hf-mirror.com）
- UCF101 完整包 6.5GB，不适合本机直接下载
- Something-Something V2 / Epic-Kitchens-100 / Ego4D 需要官网注册

本脚本提供三种获取真实数据的路径：
1. 用户手动下载 SSv2 / Epic-Kitchens / UCF101 子集后，用本脚本解压/重采样为 MP4
2. 提供阿里云盘/百度网盘分享链接时，用本脚本整理为统一格式
3. 作为 fallback，调用 generate_realistic_robot_clips.py 生成高保真合成视频

推荐的真实数据集（按与机器人操作相关度排序）：
- Something-Something V2：人手-物体交互短视频，最接近机器人操作预训练数据
- Epic-Kitchens-100：第一人称厨房操作视频
- Ego4D：大规模第一人称视频，包含大量操作行为
- UCF101：通用动作识别数据集，部分类别涉及物体操作
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import urlparse


SUPPORTED_SOURCES = {
    "something_something_v2": {
        "display_name": "Something-Something V2 (SSv2)",
        "url": "https://20bn.com/datasets/something-something/v2",
        "note": "需注册后下载，validation 集约 1.6GB",
        "recommended": True,
    },
    "epic_kitchens_100": {
        "display_name": "Epic-Kitchens-100",
        "url": "https://epic-kitchens.github.io/2021",
        "note": "需注册后下载，clip 包约 80GB（建议只下载子集）",
        "recommended": False,
    },
    "ego4d": {
        "display_name": "Ego4D",
        "url": "https://ego4d-data.org/",
        "note": "需注册并申请，数据量极大",
        "recommended": False,
    },
    "ucf101": {
        "display_name": "UCF101",
        "url": "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar",
        "note": "6.5GB，可直接下载但较大；推荐只选 20 个动作类",
        "recommended": False,
    },
}

SSV2_MANUAL_GUIDE = """
================================================================================
Something-Something V2 (SSv2) 手动下载指南
================================================================================
SSv2 当前无法通过公开镜像或 HuggingFace datasets 库自动下载完整视频。
HuggingFace 上的 something_something_v2 数据集仅包含 metadata/loading script，
脚本会明确要求 manual download；完整 19.4 GB 视频需按以下步骤获取。

1. 注册并登录 Qualcomm 官方分发页面：
   https://developer.qualcomm.com/software/ai-datasets/something-something
   （原 20bn 数据已迁移至 Qualcomm）

2. 下载视频分卷（约 19 个 .rar 文件，总大小约 19.4 GB）：
   something-something-v2-videos.part01.rar
   something-something-v2-videos.part02.rar
   ...
   something-something-v2-videos.part19.rar

3. 下载标签文件：
   something-something-v2-labels.zip

4. 在本地拼接/解压（示例命令，需安装 7z 或 WinRAR）：
   # Linux/macOS (p7zip)
   7z x something-something-v2-videos.part01.rar
   # Windows (7z 或 WinRAR)
   7z x something-something-v2-videos.part01.rar

   解压后得到大量 WebM 短视频，即为 SSv2 原始视频。

5. 取 validation 子集（约 24,777 条）：
   解压 labels 后，读取 validation.json，每条记录格式：
   {"id": "12345", "template": "Dropping [something] into [something].", "label": "...", "placeholders": [...]}
   按 id 到 video 文件夹找到对应 {id}.webm。

6. 复制 20 个 validation 视频到 data/manual_clips/，再运行：
   PYTHONPATH=src python src/utils/download_real_datasets.py --mode copy \\
       --src data/manual_clips --dst data/clips
   PYTHONPATH=src python src/pipeline.py real_batch --max-clips 20

7. 本脚本曾尝试的自动下载入口（预期失败，结果见日志）：
   - HuggingFace datasets (legacy loading script) -> Not supported in datasets>=3
   - 直接 HTTP 拉取 -> 需登录/授权，无公开直链
================================================================================
"""


def print_dataset_guide(dataset: str | None = None):
    """打印数据集获取指南。"""
    if dataset == "something_something_v2" or dataset == "ssv2":
        print(SSV2_MANUAL_GUIDE)
        return

    print("=" * 60)
    print("真实开源机器人/操作视频数据集获取指南")
    print("=" * 60)
    for name, info in SUPPORTED_SOURCES.items():
        marker = "★ 推荐" if info["recommended"] else ""
        print(f"\n{info['display_name']} ({name}): {marker}")
        print(f"  官网/下载: {info['url']}")
        print(f"  说明: {info['note']}")
    print("\n操作步骤：")
    print("1. 手动下载上述任一数据集的 MP4/WebM 文件")
    print("2. 将视频放入 data/manual_clips/ 目录")
    print("3. 运行: PYTHONPATH=src python src/utils/download_real_datasets.py --mode copy")
    print("4. 运行: PYTHONPATH=src python src/pipeline.py real_batch --max-clips 20")
    print("=" * 60)


def _ensure_hf_endpoint():
    """若环境未设置 HF endpoint，尝试使用 hf-mirror.com。"""
    if not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print(f"[INFO] 已设置 HF_ENDPOINT={os.environ['HF_ENDPOINT']}")


def _report_failure(dataset: str, reason: str, log_path: Path | None = None):
    """统一记录自动下载失败信息。"""
    msg = (
        f"[FAIL] 自动下载 {dataset} 失败。\n"
        f"       原因: {reason}\n"
        f"       请使用手动下载指南: --mode guide --dataset {dataset}"
    )
    print(msg)
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def _try_download_ssv2_hf(out_dir: Path, max_clips: int = 20) -> list[Path]:
    """
    尝试通过 HuggingFace datasets 加载 SSv2 validation 子集。
    预期会失败：该数据集使用 legacy loading script 且要求 manual download。
    """
    _ensure_hf_endpoint()
    print("[SSv2] 尝试 HuggingFace datasets 自动加载...")
    try:
        # 延迟导入，避免 datasets 不在环境中时脚本完全不可用
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(f"未安装 datasets 库: {e}")

    ds_name = "HuggingFaceM4/something_something_v2"
    print(f"[SSv2] load_dataset('{ds_name}', split='validation', streaming=True)")
    try:
        ds = load_dataset(ds_name, split="validation", streaming=True)
    except Exception as e:
        raise RuntimeError(f"HuggingFace datasets 加载失败: {e}")

    # 如果加载成功，尝试保存视频（理论上不会走到这里）
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for i, sample in enumerate(ds):
        if i >= max_clips:
            break
        clip_id = sample.get("id", f"ssv2_{i:05d}")
        video = sample.get("video")
        if video is None:
            continue
        dst = out_dir / f"real_clip_{i+1:03d}.mp4"
        # streaming 返回的可能是 bytes/path，这里做最小兼容
        if isinstance(video, bytes):
            dst.write_bytes(video)
        elif isinstance(video, (str, Path)):
            shutil.copy2(video, dst)
        else:
            continue
        saved.append(dst)
        print(f"[SSv2] saved {dst}")
    return saved


def _try_download_ssv2_http(out_dir: Path, max_clips: int = 20) -> list[Path]:
    """
    尝试通过公开 HTTP 直链下载 SSv2 视频。
    预期会失败：SSv2 无公开直链，需登录授权。
    """
    print("[SSv2] 尝试公开 HTTP 直链下载...")
    raise RuntimeError("SSv2 无公开 HTTP 直链；需注册 Qualcomm 官网后手动下载。")


def download_ssv2(
    out_dir: str | Path = "data/clips",
    max_clips: int = 20,
    method: str = "hf",
) -> list[Path]:
    """
    尝试自动下载 SSv2 validation 子集。
    若失败，打印详细手动下载指南并记录失败原因。
    """
    out_dir = Path(out_dir)
    log_path = out_dir.parent / "logs" / "ssv2_download.log"

    try:
        if method == "hf":
            return _try_download_ssv2_hf(out_dir, max_clips)
        elif method == "http":
            return _try_download_ssv2_http(out_dir, max_clips)
        else:
            raise ValueError(f"未知下载方法: {method}")
    except Exception as e:
        _report_failure("something_something_v2", str(e), log_path)
        raise


def copy_manual_clips(src_dir: Path, dst_dir: Path, max_clips: int = 20) -> list[Path]:
    """复制用户手动放置的真实视频到统一目录。"""
    dst_dir.mkdir(parents=True, exist_ok=True)
    # 清空目标目录
    for p in dst_dir.iterdir():
        if p.is_file():
            p.unlink()

    src_dir = Path(src_dir)
    supported_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    clips = sorted(
        p for p in src_dir.iterdir()
        if p.is_file() and p.suffix.lower() in supported_exts
    )[:max_clips]

    copied = []
    for i, clip in enumerate(clips, 1):
        new_name = f"real_clip_{i:03d}{clip.suffix}"
        dst = dst_dir / new_name
        shutil.copy2(clip, dst)
        copied.append(dst)

    return copied


def run_synthetic_fallback(num: int = 20):
    """当没有真实数据时，生成高保真合成视频。"""
    import sys

    sys.path.insert(0, "src/utils")
    from generate_realistic_robot_clips import generate_robot_clip

    out_dir = Path("data/clips")
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = []
    for i in range(num):
        clip_path = out_dir / f"robot_clip_{i+1:03d}.mp4"
        info = generate_robot_clip(clip_path)
        info["path"] = str(clip_path)
        info["clip_id"] = clip_path.stem
        metadata.append(info)
        print(f"Generated: {clip_path} | action={info['action']} | shape={info['shape']}")

    meta_path = out_dir / "robot_clips_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Synthetic fallback generated {num} clips in {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="真实视频数据集准备工具")
    parser.add_argument(
        "--mode",
        choices=["guide", "copy", "synthetic", "download"],
        default="guide",
        help=(
            "guide: 打印获取指南; copy: 复制手动下载的视频; "
            "synthetic: 生成高保真合成视频; download: 尝试自动下载（可能失败）"
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="something_something_v2",
        choices=list(SUPPORTED_SOURCES.keys()) + ["ssv2"],
        help="目标数据集（仅 download/guide 模式使用）",
    )
    parser.add_argument("--src", type=str, default="data/manual_clips", help="手动视频源目录")
    parser.add_argument("--dst", type=str, default="data/clips", help="目标目录")
    parser.add_argument("--max-clips", type=int, default=20, help="最大 clip 数")
    parser.add_argument(
        "--method",
        type=str,
        default="hf",
        choices=["hf", "http"],
        help="download 模式下的下载方法",
    )
    args = parser.parse_args()

    if args.mode == "guide":
        print_dataset_guide(args.dataset)
    elif args.mode == "copy":
        copied = copy_manual_clips(Path(args.src), Path(args.dst), args.max_clips)
        print(f"Copied {len(copied)} clips to {args.dst}")
    elif args.mode == "synthetic":
        run_synthetic_fallback(args.max_clips)
    elif args.mode == "download":
        if args.dataset in ("something_something_v2", "ssv2"):
            try:
                download_ssv2(args.dst, args.max_clips, method=args.method)
            except RuntimeError as e:
                print(f"\n{e}")
                sys.exit(1)
        else:
            print(f"[WARN] 暂不支持自动下载 {args.dataset}，请使用 --mode guide --dataset {args.dataset}")


if __name__ == "__main__":
    main()
