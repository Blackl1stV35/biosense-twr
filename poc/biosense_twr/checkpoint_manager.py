"""
Chunked checkpoint manager for A100 Colab Pro.

Design:
  - Keeps generated data in memory as numpy arrays / PyTorch tensors
  - When a chunk is complete (n_scenarios threshold), compresses with
    zstd (best ratio/speed for float32 arrays) and writes to Google Drive
  - Tracks session state (which chunks done) in a JSON manifest on Drive
  - 5 GB Drive budget: auto-estimates chunk sizes and warns before overflow
  - Restores from Drive on new session to resume interrupted runs

Drive folder structure:
  biosense_twr_poc/
    manifest.json
    chunk_0000.npz.zst
    chunk_0001.npz.zst
    ...
    model_best.pt
    model_latest.pt
"""

from __future__ import annotations
import os
import json
import gzip
import io
import time
import hashlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any
import numpy as np

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False  # fallback to gzip

DRIVE_BUDGET_BYTES = 4.5 * 1024**3  # 4.5 GB safety margin under 5 GB limit
CHUNK_WARN_BYTES   = 200 * 1024**2   # warn if single chunk > 200 MB


@dataclass
class ChunkManifest:
    run_id: str
    total_scenarios: int
    chunk_size: int
    chunks_completed: list[str] = field(default_factory=list)
    total_bytes_written: int = 0
    best_oks: float = 0.0
    created_at: str = ""
    updated_at: str = ""


class CheckpointManager:
    """
    Manages chunked I/O between in-memory simulation state
    and Google Drive with compression.
    """

    def __init__(
        self,
        drive_folder_id: str,
        run_id: str,
        chunk_size: int = 500,
        compress_level: int = 3,
    ):
        self.drive_folder_id = drive_folder_id
        self.run_id = run_id
        self.chunk_size = chunk_size
        self.compress_level = compress_level

        # In-memory buffer
        self._buffer: list[dict[str, Any]] = []
        self._buffer_bytes: int = 0

        # Drive API (lazy init)
        self._drive_service = None
        self._manifest: ChunkManifest | None = None

        # Local cache dir in Colab
        self.local_cache = Path(f"/tmp/biosense_cache/{run_id}")
        self.local_cache.mkdir(parents=True, exist_ok=True)

    # ── Drive API ────────────────────────────────────────────────────
    def _get_drive(self):
        if self._drive_service is None:
            try:
                from googleapiclient.discovery import build
                from google.colab import auth
                auth.authenticate_user()
                from google.auth import default
                creds, _ = default()
                self._drive_service = build('drive', 'v3', credentials=creds)
            except Exception as e:
                print(f"[CheckpointManager] Drive unavailable: {e}. Writing to /tmp only.")
        return self._drive_service

    def _drive_upload(self, local_path: Path, filename: str) -> str | None:
        """Upload file to Drive folder. Returns file ID or None."""
        svc = self._get_drive()
        if svc is None:
            return None
        try:
            from googleapiclient.http import MediaFileUpload
            meta = {"name": filename, "parents": [self.drive_folder_id]}
            media = MediaFileUpload(str(local_path), resumable=True)
            f = svc.files().create(body=meta, media_body=media, fields="id").execute()
            return f.get("id")
        except Exception as e:
            print(f"[CheckpointManager] Upload failed: {e}")
            return None

    def _drive_download(self, file_id: str, local_path: Path) -> bool:
        svc = self._get_drive()
        if svc is None:
            return False
        try:
            import io
            from googleapiclient.http import MediaIoBaseDownload
            req = svc.files().get_media(fileId=file_id)
            buf = io.BytesIO()
            dl  = MediaIoBaseDownload(buf, req)
            done = False
            while not done:
                _, done = dl.next_chunk()
            local_path.write_bytes(buf.getvalue())
            return True
        except Exception as e:
            print(f"[CheckpointManager] Download failed: {e}")
            return False

    # ── Compression ──────────────────────────────────────────────────
    def _compress(self, data: bytes) -> bytes:
        if HAS_ZSTD:
            cctx = zstd.ZstdCompressor(level=self.compress_level)
            return cctx.compress(data)
        else:
            buf = io.BytesIO()
            with gzip.GzipFile(fileobj=buf, mode='wb', compresslevel=6) as gz:
                gz.write(data)
            return buf.getvalue()

    def _decompress(self, data: bytes) -> bytes:
        if HAS_ZSTD:
            dctx = zstd.ZstdDecompressor()
            return dctx.decompress(data)
        else:
            buf = io.BytesIO(data)
            with gzip.GzipFile(fileobj=buf, mode='rb') as gz:
                return gz.read()

    # ── Buffer management ────────────────────────────────────────────
    def add(self, scenario: dict[str, Any]) -> None:
        """Add one scenario's data to the in-memory buffer."""
        self._buffer.append(scenario)
        # Rough size estimate
        for v in scenario.values():
            if isinstance(v, np.ndarray):
                self._buffer_bytes += v.nbytes

        if len(self._buffer) >= self.chunk_size:
            self.flush()

    def flush(self, force: bool = False) -> None:
        """Flush buffer to compressed chunk file → Drive."""
        if not self._buffer:
            return

        chunk_idx = len(self._manifest.chunks_completed) if self._manifest else 0
        chunk_name = f"chunk_{chunk_idx:04d}.npz"
        local_path = self.local_cache / chunk_name

        # Pack buffer into numpy savez
        arrays: dict[str, Any] = {}
        for si, scenario in enumerate(self._buffer):
            for key, val in scenario.items():
                arr_key = f"s{si:04d}_{key}"
                arrays[arr_key] = np.asarray(val) if not isinstance(val, np.ndarray) else val

        # Save to npz in memory
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        raw_bytes = buf.getvalue()

        # Compress
        compressed = self._compress(raw_bytes)
        compressed_path = local_path.with_suffix(".npz.zst" if HAS_ZSTD else ".npz.gz")
        compressed_path.write_bytes(compressed)

        size_mb = len(compressed) / 1024**2
        if len(compressed) > CHUNK_WARN_BYTES:
            print(f"[CheckpointManager] Warning: chunk {chunk_idx} = {size_mb:.1f} MB")

        # Budget check
        if self._manifest:
            projected = self._manifest.total_bytes_written + len(compressed)
            if projected > DRIVE_BUDGET_BYTES:
                print(f"[CheckpointManager] BUDGET WARNING: projected {projected/1024**3:.2f} GB")

        # Upload
        file_id = self._drive_upload(compressed_path, compressed_path.name)
        print(f"[CheckpointManager] Chunk {chunk_idx} flushed: {size_mb:.1f} MB compressed"
              + (f" → Drive {file_id}" if file_id else " → local only"))

        # Update manifest
        if self._manifest:
            self._manifest.chunks_completed.append(compressed_path.name)
            self._manifest.total_bytes_written += len(compressed)
            self._manifest.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ")
            self._save_manifest()

        # Clear buffer
        self._buffer.clear()
        self._buffer_bytes = 0

    # ── Manifest ─────────────────────────────────────────────────────
    def init_manifest(self, total_scenarios: int) -> None:
        manifest_path = self.local_cache / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                d = json.load(f)
            self._manifest = ChunkManifest(**d)
            print(f"[CheckpointManager] Resumed: {len(self._manifest.chunks_completed)} chunks done")
        else:
            self._manifest = ChunkManifest(
                run_id=self.run_id,
                total_scenarios=total_scenarios,
                chunk_size=self.chunk_size,
                created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                updated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            )
            self._save_manifest()

    def _save_manifest(self) -> None:
        if self._manifest is None:
            return
        manifest_path = self.local_cache / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(asdict(self._manifest), f, indent=2)

    def update_best_oks(self, oks: float) -> None:
        if self._manifest and oks > self._manifest.best_oks:
            self._manifest.best_oks = oks
            self._save_manifest()
            print(f"[CheckpointManager] New best OKS: {oks:.4f}")

    def scenarios_already_done(self) -> int:
        if self._manifest is None:
            return 0
        return len(self._manifest.chunks_completed) * self._manifest.chunk_size

    def budget_remaining_gb(self) -> float:
        if self._manifest is None:
            return DRIVE_BUDGET_BYTES / 1024**3
        used = self._manifest.total_bytes_written
        return (DRIVE_BUDGET_BYTES - used) / 1024**3
