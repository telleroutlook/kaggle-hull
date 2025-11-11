#!/usr/bin/env python3
"""打包 Kaggle 部署归档并输出 checksum."""

from __future__ import annotations

import argparse
import hashlib
import os
import zipfile
from pathlib import Path
from typing import Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Kaggle deployment archive.")
    parser.add_argument(
        "--include-tests",
        action="store_true",
        help="包含 working/tests 目录（默认跳过以减小归档体积）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("input/kaggle_hull_solver.zip"),
        help="输出压缩包路径",
    )
    return parser.parse_args()


def build_manifest(include_tests: bool) -> List[str]:
    manifest = [
        "working/main.py",
        "working/main_fixed.py",
        "working/inference_server.py",
        "working/warnings_handler.py",  # 添加缺少的警告处理模块
        "working/__init__.py",
        "working/config.ini",
        "working/lib/",
        "working/artifacts/",
        "requirements.txt",
        "README.md",
        "IFLOW.md",
        "KAGGLE_DEPLOYMENT.md",
        "kaggle_simple_cell_fixed.py",
        "create_kaggle_archive.py",
    ]
    if include_tests:
        manifest.append("working/tests/")
    return manifest


def _iter_files(base: str) -> Iterable[tuple[str, str]]:
    if os.path.isdir(base):
        for root, _, files in os.walk(base):
            if "__pycache__" in root:
                continue
            for file in files:
                if file.endswith((".pyc", ".pyo")):
                    continue
                file_path = os.path.join(root, file)
                yield file_path, os.path.relpath(file_path, ".")
    elif os.path.exists(base):
        yield base, base


def write_checksum(path: Path) -> Path:
    sha256 = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            if not chunk:
                break
            sha256.update(chunk)
    checksum = sha256.hexdigest()
    checksum_path = path.with_suffix(path.suffix + ".sha256")
    checksum_path.write_text(f"{checksum}  {path.name}\n", encoding="utf-8")
    print(f"🔐 SHA256: {checksum}")
    print(f"📝 Checksum saved to {checksum_path}")
    return checksum_path


def create_kaggle_archive(include_tests: bool = False, output: Path | None = None):
    archive_path = output or Path("input/kaggle_hull_solver.zip")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    files_to_include = build_manifest(include_tests)

    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        added_files = set()
        for item in files_to_include:
            found_any = False
            for file_path, arcname in _iter_files(item):
                if arcname in added_files:
                    continue
                zipf.write(file_path, arcname)
                added_files.add(arcname)
                found_any = True
                print(f"Added: {arcname}")
            if not found_any:
                print(f"Warning: {item} not found, skipping")

    size_mb = archive_path.stat().st_size / (1024 * 1024)
    print(f"\n✅ Created Kaggle deployment archive: {archive_path} ({size_mb:.2f} MB)")
    print(f"📁 Total files added: {len(added_files)}")
    write_checksum(archive_path)


if __name__ == "__main__":
    cli_args = parse_args()
    create_kaggle_archive(include_tests=cli_args.include_tests, output=cli_args.output)
