"""Configuration for dicom-mcp, driven by environment variables.

Env vars:
  DICOM_ROOTS         path-sep separated list of directories the server may read.
                      Defaults to <hermes-home>/dicom-inbox and ./dicom-data.
  DICOM_WORKDIR       where extracted archives / anonymized copies go.
                      Default <hermes-home>/dicom-mcp-work.
  DICOM_RESCAN        "1" forces a fresh filesystem scan on the next call.
  DICOM_MAX_FILES     max files indexed per scan (default 20000).
  DICOM_MAX_IMAGE_PX  max image width/height returned (default 1024).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path


def _hermes_home() -> Path:
    h = os.environ.get("HERMES_HOME")
    if h:
        return Path(h)
    return Path.home() / "AppData" / "Local" / "hermes"


def _default_roots() -> list[Path]:
    cands = [
        _hermes_home() / "dicom-inbox",
        Path.cwd() / "dicom-data",
    ]
    return [c for c in cands if c.is_dir()] or [cands[0]]


@dataclass
class Config:
    roots: list[Path] = field(default_factory=_default_roots)
    workdir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("DICOM_WORKDIR") or (_hermes_home() / "dicom-mcp-work")
        )
    )
    max_files: int = int(os.environ.get("DICOM_MAX_FILES", "20000"))
    max_image_px: int = int(os.environ.get("DICOM_MAX_IMAGE_PX", "1024"))

    @classmethod
    def from_env(cls) -> "Config":
        roots_env = os.environ.get("DICOM_ROOTS")
        roots = (
            [Path(p) for p in roots_env.split(os.pathsep) if p.strip()]
            if roots_env
            else _default_roots()
        )
        roots = [p for p in roots if p.exists()]
        cfg = cls(roots=roots or _default_roots())
        cfg.workdir.mkdir(parents=True, exist_ok=True)
        return cfg

    def log(self, msg: str) -> None:
        # stdout is the MCP channel — logs go to stderr only.
        print(f"[dicom-mcp] {msg}", file=sys.stderr, flush=True)
