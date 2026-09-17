"""
真实开源视频数据集下载脚本。

自动下载源（已验证可用）：
- LeRobot 真实机器人操作数据集（HuggingFace，经 hf-mirror.com 镜像访问）
  默认 lerobot/aloha_static_battery：ALOHA 真机双臂遥操作，任务为
  "Place the battery into the slot of the remote controller."
  同类可选：aloha_static_coffee / aloha_static_candy / aloha_static_tape /
            aloha_static_ziploc_slide / aloha_static_towel 等。
  注意区分命名：aloha_sim_* 为 MuJoCo 仿真，aloha_static_* 与 aloha_mobile_*
  为真机采集，本项目只取真机数据。

仍需手动下载的源（自动路径已确认不可行）：
- Something-Something V2：需 Qualcomm 官网注册，19 个分卷约 19.4GB
- Epic-Kitchens-100 / Ego4D：需注册申请，数据量极大

LeRobot v3 数据集结构：
  meta/info.json                           fps、视频键、路径模板
  meta/episodes/chunk-*/file-*.parquet     每条 episode 的起止时间戳与任务描述
  videos/{video_key}/chunk-*/file-*.mp4    多个 episode 拼接在同一 mp4 中

因此取真实片段的做法是：先读 episodes parquet 拿到每条 episode 在视频中的
[from_timestamp, to_timestamp)，再按帧号切出独立 clip。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import yaml

HF_MIRROR = "https://hf-mirror.com"
DEFAULT_REPO = "lerobot/aloha_static_battery"

# ---------------------------------------------------------------------------
# 通用下载工具
# ---------------------------------------------------------------------------


def ensure_hf_endpoint() -> str:
    """统一 HF 端点，默认走 hf-mirror.com。"""
    endpoint = os.environ.get("HF_ENDPOINT") or HF_MIRROR
    os.environ["HF_ENDPOINT"] = endpoint
    return endpoint.rstrip("/")


def resolve_url(repo: str, path: str, repo_type: str = "datasets") -> str:
    endpoint = ensure_hf_endpoint()
    return f"{endpoint}/{repo_type}/{repo}/resolve/main/{path}"


def api_url(repo: str, path: str = "", repo_type: str = "datasets") -> str:
    endpoint = ensure_hf_endpoint()
    suffix = f"/{path}" if path else ""
    return f"{endpoint}/api/{repo_type}/{repo}{suffix}"


def download_file(url: str, dst: Path, retries: int = 3, timeout: int = 120) -> Path:
    """流式下载，带重试，已存在且非空则跳过。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        print(f"[SKIP] 已存在 {dst.name} ({dst.stat().st_size} bytes)")
        return dst

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "vma-loop/1.0"})
            tmp = dst.with_suffix(dst.suffix + ".part")
            with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as f:
                total = int(resp.headers.get("Content-Length") or 0)
                got = 0
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    got += len(chunk)
                    if total and got % (10 * 1024 * 1024) < 1024 * 1024:
                        print(f"    {dst.name}: {got/1048576:.0f}/{total/1048576:.0f} MB")
            tmp.replace(dst)
            print(f"[OK] {dst.name} ({dst.stat().st_size} bytes)")
            return dst
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            last_err = e
            print(f"[RETRY {attempt}/{retries}] {dst.name}: {e}")
    raise RuntimeError(f"下载失败 {url}: {last_err}")


def http_get_json(url: str, timeout: int = 60):
    req = urllib.request.Request(url, headers={"User-Agent": "vma-loop/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8-sig"))


# ---------------------------------------------------------------------------
# LeRobot 真机数据集
# ---------------------------------------------------------------------------


def _pick_video_key(features: dict) -> str:
    """选择作为标注输入的主视角视频键。"""
    video_keys = [k for k, v in features.items() if v.get("dtype") == "video"]
    if not video_keys:
        raise RuntimeError("数据集中没有任何 video 特征")
    for preferred in ("observation.images.cam_high", "observation.images.top", "observation.image"):
        if preferred in video_keys:
            return preferred
    return sorted(video_keys)[0]


def _list_episode_parquets(repo: str) -> list[str]:
    """列出 meta/episodes 下的所有 parquet。"""
    try:
        entries = http_get_json(api_url(repo, "tree/main/meta/episodes?recursive=true"))
        paths = [e["path"] for e in entries if e.get("type") == "file" and e["path"].endswith(".parquet")]
        if paths:
            return sorted(paths)
    except Exception as e:
        print(f"[WARN] 列举 episodes 失败，回退到默认路径: {e}")
    return ["meta/episodes/chunk-000/file-000.parquet"]


def _read_episodes(repo: str, cache_dir: Path) -> tuple[list[dict], dict]:
    """下载并解析 episodes 元数据，返回 (episodes, info)。"""
    try:
        import pyarrow.parquet as pq
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(f"需要 pyarrow 读取 episodes 元数据: {e}")

    info = http_get_json(resolve_url(repo, "meta/info.json"))
    video_key = _pick_video_key(info["features"])
    fps = float(info["features"][video_key]["video_info"]["video.fps"])

    episodes: list[dict] = []
    for rel in _list_episode_parquets(repo):
        # 缓存名必须带仓库前缀：不同仓库的 episodes 文件同名（都是 file-000.parquet）
        local = cache_dir / f"{repo.split('/')[-1]}-{Path(rel).name}"
        download_file(resolve_url(repo, rel), local)
        table = pq.read_table(local)
        cols = [
            "episode_index",
            f"videos/{video_key}/chunk_index",
            f"videos/{video_key}/file_index",
            f"videos/{video_key}/from_timestamp",
            f"videos/{video_key}/to_timestamp",
            "tasks",
            "length",
        ]
        present = [c for c in cols if c in table.column_names]
        for row in table.select(present).to_pylist():
            tasks = row.get("tasks") or []
            episodes.append(
                {
                    "episode_index": int(row["episode_index"]),
                    "chunk_index": int(row[f"videos/{video_key}/chunk_index"]),
                    "file_index": int(row[f"videos/{video_key}/file_index"]),
                    "from_sec": float(row[f"videos/{video_key}/from_timestamp"]),
                    "to_sec": float(row[f"videos/{video_key}/to_timestamp"]),
                    "length": int(row.get("length") or 0),
                    "task": tasks[0] if tasks else "",
                }
            )
    episodes.sort(key=lambda e: e["episode_index"])
    meta = {"repo": repo, "video_key": video_key, "fps": fps, "total_episodes": len(episodes)}
    return episodes, meta


def _cut_clip(
    src: Path,
    dst: Path,
    start_frame: int,
    end_frame: int,
    out_fps: float,
    src_fps: float,
) -> float:
    """按帧区间切出 clip，按 out_fps 抽帧，返回实际时长（秒）。"""
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {src}")
    if start_frame > 0:
        # 直接定位到起始帧，避免从头解码整个视频
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))

    writer = None
    frame_idx = start_frame
    kept = 0
    step = max(1, int(round(src_fps / out_fps)))
    try:
        while frame_idx < end_frame:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx >= start_frame and (frame_idx - start_frame) % step == 0:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(dst), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h)
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"无法创建输出视频: {dst}")
                writer.write(frame)
                kept += 1
            frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if kept == 0:
        raise RuntimeError(f"未写出任何帧: {dst}")
    return kept / out_fps


def fetch_lerobot_clips(
    repos: list[str] | str = DEFAULT_REPO,
    out_dir: str | Path = "data/clips",
    max_clips: int = 20,
    max_duration: float = 15.0,
    out_fps: float = 10.0,
) -> list[dict]:
    """从多个 LeRobot 真机数据集各切若干 episode，组成多任务 clip 集。"""
    import math

    if isinstance(repos, str):
        repos = [r.strip() for r in repos.split(",") if r.strip()]
    out_dir = Path(out_dir)
    root = out_dir.parent
    cache_dir = root / "_real_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    per_repo = max(1, math.ceil(max_clips / len(repos)))
    records: list[dict] = []

    for repo in repos:
        if len(records) >= max_clips:
            break
        print(f"\n[LeRobot] 数据源 {repo}（本仓库配额 {per_repo} 条）")
        try:
            episodes, meta = _read_episodes(repo, cache_dir)
        except Exception as e:
            print(f"[WARN] 跳过 {repo}: {e}")
            continue
        print(
            f"[LeRobot] {meta['total_episodes']} 条 episode，"
            f"主视角 {meta['video_key']}，源 {meta['fps']} fps"
        )

        video_cache: dict[int, Path] = {}
        taken = 0
        for ep in episodes:
            if len(records) >= max_clips or taken >= per_repo:
                break
            start_sec, end_sec = ep["from_sec"], ep["to_sec"]
            if end_sec - start_sec > max_duration:
                end_sec = start_sec + max_duration

            file_index = ep["file_index"]
            if file_index not in video_cache:
                rel = (
                    f"videos/{meta['video_key']}/chunk-{ep['chunk_index']:03d}/"
                    f"file-{file_index:03d}.mp4"
                )
                video_cache[file_index] = download_file(
                    resolve_url(repo, rel), cache_dir / f"{repo.split('/')[-1]}-{Path(rel).name}",
                    timeout=900,
                )
            src_video = video_cache[file_index]

            clip_id = f"real_clip_{len(records) + 1:03d}"
            dst = out_dir / f"{clip_id}.mp4"
            src_fps = meta["fps"]
            start_frame = int(round(start_sec * src_fps))
            end_frame = int(round(end_sec * src_fps))
            duration = _cut_clip(src_video, dst, start_frame, end_frame, out_fps, src_fps)
            print(
                f"  {clip_id} <- {repo.split('/')[-1]} ep{ep['episode_index']:03d} "
                f"[{start_sec:.1f},{end_sec:.1f})s  {duration:.1f}s  {ep['task'][:44]}"
            )
            records.append(
                {
                    "clip_id": clip_id,
                    "source": repo,
                    "episode_index": ep["episode_index"],
                    "task": ep["task"],
                    "video_file": src_video.name,
                    "from_sec": round(start_sec, 3),
                    "to_sec": round(end_sec, 3),
                    "duration_sec": round(duration, 3),
                    "fps": out_fps,
                    "path": str(dst).replace("\\", "/"),
                }
            )
            taken += 1

    meta_path = out_dir / "real_clips_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"[LeRobot] 写出 {len(records)} 个真实 clip，元数据 {meta_path}")

    tasks: dict[str, int] = {}
    for r in records:
        tasks[r["task"]] = tasks.get(r["task"], 0) + 1
    for t, n in sorted(tasks.items(), key=lambda kv: -kv[1]):
        print(f"    任务分布: {n:>3}  {t}")
    return records


def archive_existing_clips(clip_dir: str | Path, backup_dir: str | Path) -> int:
    """把已有 clip 移到备份目录，避免与真实数据混跑。"""
    clip_dir, backup_dir = Path(clip_dir), Path(backup_dir)
    if not clip_dir.exists():
        return 0
    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    moved = 0
    for p in clip_dir.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            backup_dir.mkdir(parents=True, exist_ok=True)
            target = backup_dir / p.name
            if target.exists():
                target.unlink()
            shutil.move(str(p), str(target))
            moved += 1
    if moved:
        print(f"[ARCHIVE] {moved} 个旧 clip 已移动到 {backup_dir}")
    return moved


# ---------------------------------------------------------------------------
# 历史记录：SSv2 手动路径（自动下载已确认不可行）
# ---------------------------------------------------------------------------

SSV2_MANUAL_GUIDE = """
================================================================================
Something-Something V2 (SSv2) 手动下载指南
================================================================================
SSv2 无法通过公开镜像或 HuggingFace datasets 库自动下载完整视频。
HuggingFace 上的 something_something_v2 数据集仅包含 metadata/loading script，
脚本会明确要求 manual download；完整 19.4 GB 视频需按以下步骤获取。

1. 注册并登录 Qualcomm 官方分发页面：
   https://developer.qualcomm.com/software/ai-datasets/something-something

2. 下载视频分卷（约 19 个 .rar 文件，总大小约 19.4 GB）：
   something-something-v2-videos.part01.rar ... part19.rar

3. 下载标签文件：something-something-v2-labels.zip

4. 本地拼接/解压（需 7z 或 WinRAR）：
   7z x something-something-v2-videos.part01.rar

5. 取 validation 子集，按 id 找到对应 {id}.webm。

6. 复制到 data/manual_clips/，再运行：
   python src/utils/download_real_datasets.py --mode copy --src data/manual_clips
   python src/pipeline.py real_batch --max-clips 20

如果需要真机机器人操作视频，直接用 LeRobot 自动路径即可：
   python src/utils/download_real_datasets.py --mode lerobot \\
       --repo lerobot/aloha_static_battery --max-clips 20
================================================================================
"""

MANUAL_SOURCES = {
    "something_something_v2": {
        "display_name": "Something-Something V2 (SSv2)",
        "url": "https://20bn.com/datasets/something-something/v2",
        "note": "需 Qualcomm 官网注册，19 个分卷约 19.4GB",
    },
    "epic_kitchens_100": {
        "display_name": "Epic-Kitchens-100",
        "url": "https://epic-kitchens.github.io/2021",
        "note": "需注册，clip 包约 80GB",
    },
    "ego4d": {
        "display_name": "Ego4D",
        "url": "https://ego4d-data.org/",
        "note": "需注册并申请，数据量极大",
    },
}


def print_dataset_guide(dataset: str | None = None):
    if dataset in ("something_something_v2", "ssv2"):
        print(SSV2_MANUAL_GUIDE)
        return
    print("=" * 64)
    print("真实视频数据获取指南")
    print("=" * 64)
    print("\n【自动路径】LeRobot 真机机器人操作数据集")
    print(f"  默认数据源: {DEFAULT_REPO}")
    print("  运行: python src/utils/download_real_datasets.py --mode lerobot --max-clips 20")
    print("  可选真机任务: aloha_static_coffee / candy / tape / towel / ziploc_slide 等")
    print("\n【手动路径】需注册的数据集")
    for name, info in MANUAL_SOURCES.items():
        print(f"\n  {info['display_name']} ({name})")
        print(f"    下载: {info['url']}")
        print(f"    说明: {info['note']}")
    print("\n步骤：放入 data/manual_clips/ 后运行 --mode copy，再运行 src/pipeline.py")
    print("=" * 64)


def copy_manual_clips(src_dir: Path, dst_dir: Path, max_clips: int = 20) -> list[Path]:
    """复制手动放置的真实视频到统一目录。"""
    dst_dir.mkdir(parents=True, exist_ok=True)
    supported = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    clips = sorted(
        p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in supported
    )[:max_clips]
    copied = []
    for i, clip in enumerate(clips, 1):
        dst = dst_dir / f"real_clip_{i:03d}{clip.suffix}"
        shutil.copy2(clip, dst)
        copied.append(dst)
    return copied


def run_synthetic_fallback(num: int = 20):
    """没有真实数据时的兜底：生成高保真合成视频（仅用于回归测试）。"""
    sys.path.insert(0, str(Path(__file__).parent))
    from generate_realistic_robot_clips import generate_robot_clip

    out_dir = Path("data/clips")
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = []
    for i in range(num):
        clip_path = out_dir / f"robot_clip_{i + 1:03d}.mp4"
        info = generate_robot_clip(clip_path)
        info["path"] = str(clip_path)
        info["clip_id"] = clip_path.stem
        metadata.append(info)
        print(f"Generated: {clip_path} | action={info['action']} | shape={info['shape']}")
    with open(out_dir / "robot_clips_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def load_data_config(path: str | Path = "configs/data_config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="真实视频数据集准备工具")
    parser.add_argument(
        "--mode",
        choices=["guide", "copy", "synthetic", "lerobot"],
        default="guide",
        help="guide: 获取指南; lerobot: 自动下载真机数据; copy: 复制手动数据; synthetic: 合成兜底",
    )
    parser.add_argument("--dataset", type=str, default="lerobot", help="guide 模式下的数据集名")
    parser.add_argument(
        "--repo",
        type=str,
        default=DEFAULT_REPO,
        help="LeRobot 数据集仓库，可用逗号分隔多个以组成多任务 clip 集",
    )
    parser.add_argument("--src", type=str, default="data/manual_clips", help="手动视频源目录")
    parser.add_argument("--dst", type=str, default="data/clips", help="目标目录")
    parser.add_argument("--max-clips", type=int, default=20, help="最大 clip 数")
    parser.add_argument("--max-duration", type=float, default=15.0, help="单个 clip 最长秒数")
    parser.add_argument("--out-fps", type=float, default=10.0, help="输出 clip 帧率")
    parser.add_argument(
        "--keep-existing", action="store_true", help="保留 clips 目录中已有的视频"
    )
    args = parser.parse_args()

    if args.mode == "guide":
        print_dataset_guide(args.dataset)
    elif args.mode == "copy":
        copied = copy_manual_clips(Path(args.src), Path(args.dst), args.max_clips)
        print(f"Copied {len(copied)} clips to {args.dst}")
    elif args.mode == "synthetic":
        run_synthetic_fallback(args.max_clips)
    elif args.mode == "lerobot":
        if not args.keep_existing:
            archive_existing_clips(args.dst, Path(args.dst).parent / "clips_synthetic_backup")
        fetch_lerobot_clips(
            repos=args.repo,
            out_dir=args.dst,
            max_clips=args.max_clips,
            max_duration=args.max_duration,
            out_fps=args.out_fps,
        )


if __name__ == "__main__":
    main()
