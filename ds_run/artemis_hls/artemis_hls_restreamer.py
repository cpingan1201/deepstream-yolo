#!/usr/bin/env python3
"""
Restream Hikvision iSecure Center temporary preview RTSP URLs.

Two downstream modes are supported:

1. rtsp: publish every camera to a stable MediaMTX RTSP path. This is the
   preferred mode for DeepStream because rtspsrc can recover from brief
   publisher refreshes better than hlsdemux handles long-running live HLS.
2. hls: keep the previous fixed HLS URL behavior with active/standby slots.

The upstream preview URL is temporary, so both modes refresh the upstream URL
before it expires and keep the downstream address stable.
"""

import argparse
import csv
import functools
import hashlib
import hmac
import http.server
import json
import os
import re
import shutil
import signal
import ssl
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib import request


ARTEMIS_PATH = "/artemis"
CAMERA_LIST_API = ARTEMIS_PATH + "/api/resource/v1/cameras"
PREVIEW_URL_API = ARTEMIS_PATH + "/api/video/v2/cameras/previewURLs"

# Default runtime configuration for this machine.
DEFAULT_KEY_FILE = "key.txt"
DEFAULT_ARTEMIS_HOST = "111.4.138.199:10443"
DEFAULT_CAMERA_JSON = "./cameras.json"
DEFAULT_OUTPUT_DIR = "/mnt/data/hls"
DEFAULT_HTTP_HOST = "0.0.0.0"
DEFAULT_HTTP_PORT = 18080
DEFAULT_PUBLIC_HOST = "111.4.138.205"
DEFAULT_FFMPEG = "/usr/bin/ffmpeg"
DEFAULT_STREAM_TYPE = 1
DEFAULT_OUTPUT_MODE = "rtsp"
DEFAULT_RTSP_PUBLISH_HOST = "127.0.0.1"
DEFAULT_RTSP_PORT = 8554
DEFAULT_RTSP_STARTUP_TIMEOUT = 8
DEFAULT_PREVIEW_TTL = 300
DEFAULT_REFRESH_AHEAD = 30
DEFAULT_REFRESH_JITTER = 90
DEFAULT_WARMUP_TIMEOUT = 20
DEFAULT_WARMUP_MIN_SEGMENTS = 2
DEFAULT_RETRY_5XX = [30, 60, 120, 300]
DEFAULT_RETRY_TIMEOUT = [10, 20, 30, 60]
DEFAULT_RETRY_GENERIC = [5, 10, 20, 30]
DEFAULT_RETRY_STANDBY_5XX = [60, 120, 300, 600]
DEFAULT_RETRY_STANDBY_TIMEOUT = [15, 30, 60, 120]
DEFAULT_RETRY_AUTH = [300, 600, 900, 1800]
DEFAULT_PREVIEW_CONCURRENCY = 2
DEFAULT_WARMUP_CONCURRENCY = 8
DEFAULT_START_STAGGER = 0.5
DEFAULT_FFMPEG_START_GAP = 0.3

LOG_LOCK = threading.Lock()
MAIN_LOG_PATH: Optional[Path] = None
SLOT_REGISTRY_LOCK = threading.Lock()
SLOT_REGISTRY: Dict[int, "SlotProcess"] = {}
FFMPEG_START_LOCK = threading.Lock()
LAST_FFMPEG_START_AT = 0.0


def setup_main_logger(log_path: Path) -> None:
    global MAIN_LOG_PATH
    log_path.parent.mkdir(parents=True, exist_ok=True)
    MAIN_LOG_PATH = log_path
    log(f"[LOG] main diagnostic log: {log_path}")


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"{timestamp} {message}"
    with LOG_LOCK:
        print(line, flush=True)
        if MAIN_LOG_PATH is not None:
            try:
                with MAIN_LOG_PATH.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except OSError:
                pass


def register_slot_process(slot: "SlotProcess") -> None:
    with SLOT_REGISTRY_LOCK:
        SLOT_REGISTRY[slot.process.pid] = slot


def unregister_slot_process(slot: Optional["SlotProcess"]) -> None:
    if slot is None:
        return
    with SLOT_REGISTRY_LOCK:
        SLOT_REGISTRY.pop(slot.process.pid, None)


def start_process_reaper(stop_event: threading.Event, interval: float = 2.0) -> threading.Thread:
    def worker() -> None:
        while not stop_event.is_set():
            with SLOT_REGISTRY_LOCK:
                slots = list(SLOT_REGISTRY.values())
            for slot in slots:
                proc = slot.process
                if proc.poll() is not None:
                    try:
                        proc.wait(timeout=0)
                    except Exception:
                        pass
            stop_event.wait(interval)

    thread = threading.Thread(target=worker, name="ffmpeg-reaper", daemon=True)
    thread.start()
    return thread


def read_key_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^(ak|appKey|key|sk|appSecret|secret)\s*[:=]\s*(.+)$", line, re.I)
        if not match:
            continue
        key = match.group(1).lower()
        value = match.group(2).strip()
        if key in ("ak", "appkey", "key"):
            values["ak"] = value
        elif key in ("sk", "appsecret", "secret"):
            values["sk"] = value
    if not values.get("ak") or not values.get("sk"):
        raise ValueError(f"{path} must contain ak=... and sk=...")
    return values


class ArtemisClient:
    def __init__(self, host: str, app_key: str, app_secret: str, timeout: int = 20, insecure_tls: bool = True):
        self.host = host.strip().replace("https://", "").replace("http://", "").rstrip("/")
        self.base_url = "https://" + self.host
        self.app_key = app_key
        self.app_secret = app_secret
        self.timeout = timeout
        self.ssl_context = ssl._create_unverified_context() if insecure_tls else None

    def post(self, path: str, body_obj: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(body_obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        method = "POST"
        accept = "*/*"
        content_type = "application/json"
        timestamp = str(int(time.time() * 1000))
        nonce = str(uuid.uuid4())
        signature_headers = "x-ca-key,x-ca-nonce,x-ca-timestamp"
        string_to_sign = (
            f"{method}\n"
            f"{accept}\n"
            f"{content_type}\n"
            f"x-ca-key:{self.app_key}\n"
            f"x-ca-nonce:{nonce}\n"
            f"x-ca-timestamp:{timestamp}\n"
            f"{path}"
        )
        signature = hmac.new(
            self.app_secret.encode("utf-8"),
            string_to_sign.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        signature_b64 = __import__("base64").b64encode(signature).decode("ascii")

        req = request.Request(
            self.base_url + path,
            data=body,
            method=method,
            headers={
                "Accept": accept,
                "Content-Type": content_type,
                "X-Ca-Key": self.app_key,
                "X-Ca-Nonce": nonce,
                "X-Ca-Timestamp": timestamp,
                "X-Ca-Signature-Headers": signature_headers,
                "X-Ca-Signature": signature_b64,
            },
        )
        with request.urlopen(req, timeout=self.timeout, context=self.ssl_context) as resp:
            payload = resp.read().decode("utf-8")
        return json.loads(payload)

    def get_cameras(self, page_size: int = 1000) -> List[Dict[str, Any]]:
        first = self.post(CAMERA_LIST_API, {"pageNo": 1, "pageSize": page_size})
        self._assert_success(first, "camera list page 1")
        total = int(first.get("data", {}).get("total", 0))
        cameras = list(first.get("data", {}).get("list", []) or [])
        total_pages = (total + page_size - 1) // page_size
        for page in range(2, total_pages + 1):
            result = self.post(CAMERA_LIST_API, {"pageNo": page, "pageSize": page_size})
            self._assert_success(result, f"camera list page {page}")
            cameras.extend(result.get("data", {}).get("list", []) or [])
        return cameras

    def get_preview_url(
        self,
        camera_index_code: str,
        stream_type: int,
        protocol: str = "rtsp",
        transmode: int = 1,
        streamform: str = "rtp",
    ) -> str:
        result = self.post(
            PREVIEW_URL_API,
            {
                "cameraIndexCode": camera_index_code,
                "streamType": stream_type,
                "protocol": protocol,
                "transmode": transmode,
                "expand": "transcode=0",
                "streamform": streamform,
            },
        )
        self._assert_success(result, f"preview URL for {camera_index_code}")
        url = (result.get("data") or {}).get("url")
        if not url:
            raise RuntimeError(f"preview URL missing for {camera_index_code}: {result}")
        return str(url)

    @staticmethod
    def _assert_success(result: Dict[str, Any], label: str) -> None:
        if str(result.get("code")) != "0":
            raise RuntimeError(f"{label} failed: {result.get('code')} {result.get('msg')}")


@dataclass
class CameraStream:
    index: int
    camera_index_code: str
    camera_name: str
    gb_index_code: str = ""
    channel_no: str = ""
    region_index_code: str = ""


def clean_hls_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.glob("*"):
        if child.is_file() and child.suffix.lower() in (".m3u8", ".ts", ".tmp"):
            try:
                child.unlink()
            except OSError:
                pass


def read_playlist(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def count_segments_in_playlist(path: Path) -> int:
    text = read_playlist(path)
    return sum(1 for line in text.splitlines() if line.strip().endswith(".ts"))


def get_latest_media_mtime(path: Path) -> float:
    playlist = path / "index.m3u8"
    mtimes = []
    if playlist.exists():
        mtimes.append(playlist.stat().st_mtime)
    for child in path.glob("*.ts"):
        try:
            mtimes.append(child.stat().st_mtime)
        except OSError:
            pass
    return max(mtimes, default=0.0)


def copy_playlist_atomic(src: Path, dst: Path, slot_name: str) -> None:
    content = read_playlist(src)
    if not content:
        raise RuntimeError(f"playlist not ready: {src}")

    slot_prefix = f"{slot_name}/"
    lines = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.endswith(".ts") and not stripped.startswith("#"):
            lines.append(slot_prefix + stripped)
        else:
            lines.append(line)
    temp = dst.with_suffix(".tmp")
    temp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temp, dst)


class SlotProcess:
    def __init__(self, slot_name: str, slot_dir: Path, process: subprocess.Popen, log_path: Path, log_file):
        self.slot_name = slot_name
        self.slot_dir = slot_dir
        self.process = process
        self.log_path = log_path
        self.log_file = log_file
        self.started_at = time.monotonic()
        self.preview_url = ""
        self.refresh_deadline = 0.0


class HlsWorker(threading.Thread):
    def __init__(
        self,
        client: ArtemisClient,
        camera: CameraStream,
        output_dir: Path,
        ffmpeg_bin: str,
        stream_type: int,
        hls_time: int,
        hls_list_size: int,
        restart_delay: int,
        transcode: bool,
        log_dir: Path,
        preview_ttl: int,
        refresh_ahead: int,
        refresh_jitter: int,
        warmup_timeout: int,
        warmup_min_segments: int,
        preview_semaphore: threading.Semaphore,
        warmup_semaphore: threading.Semaphore,
        ffmpeg_start_gap: float,
    ):
        super().__init__(name=f"hls-{camera.camera_index_code}", daemon=True)
        self.client = client
        self.camera = camera
        self.output_dir = output_dir
        self.ffmpeg_bin = ffmpeg_bin
        self.stream_type = stream_type
        self.hls_time = hls_time
        self.hls_list_size = hls_list_size
        self.restart_delay = restart_delay
        self.transcode = transcode
        self.log_dir = log_dir
        self.preview_ttl = max(0, preview_ttl)
        self.refresh_ahead = max(0, refresh_ahead)
        self.refresh_jitter = max(0, refresh_jitter)
        self.warmup_timeout = max(5, warmup_timeout)
        self.warmup_min_segments = max(1, warmup_min_segments)
        self.preview_semaphore = preview_semaphore
        self.warmup_semaphore = warmup_semaphore
        self.ffmpeg_start_gap = max(0.0, ffmpeg_start_gap)
        self.stop_event = threading.Event()
        self.stream_dir = self.output_dir / self.camera.camera_index_code
        self.public_playlist = self.stream_dir / "index.m3u8"
        self.active_slot_name = "slot_a"
        self.active_slot: Optional[SlotProcess] = None
        self.standby_slot: Optional[SlotProcess] = None
        self.ever_ready = False
        self.bootstrap_failures = 0
        self.active_rebuild_failures = 0
        self.standby_failures = 0
        self.next_active_retry_at = 0.0
        self.next_standby_retry_at = 0.0

    def stop(self) -> None:
        self.stop_event.set()
        self._stop_slot(self.active_slot)
        self._stop_slot(self.standby_slot)
        self.active_slot = None
        self.standby_slot = None

    def _slot_dir(self, slot_name: str) -> Path:
        return self.stream_dir / slot_name

    def _slot_playlist(self, slot_name: str) -> Path:
        return self._slot_dir(slot_name) / "index.m3u8"

    def _inactive_slot_name(self) -> str:
        return "slot_b" if self.active_slot_name == "slot_a" else "slot_a"

    def _read_failure_text(self, slot: Optional[SlotProcess], exc: Optional[BaseException] = None) -> str:
        parts: List[str] = []
        if exc is not None:
            parts.append(str(exc))
        tail = self._read_slot_log_tail(slot, max_lines=20)
        if tail:
            parts.append(tail)
        return "\n".join(part for part in parts if part).strip()

    def _classify_failure(self, failure_text: str) -> str:
        text = (failure_text or "").lower()
        if (re.search(r"\b5\d\d\b", text) or "5xx" in text) and (
            "server error" in text or "internal server error" in text or "method describe failed" in text
        ):
            return "server_5xx"
        if any(token in text for token in ["401", "403", "404", "unauthorized", "forbidden", "not found"]):
            return "auth_or_notfound"
        if any(
            token in text
            for token in ["timed out", "timeout", "connection refused", "temporarily unavailable", "i/o error"]
        ):
            return "timeout"
        return "generic"

    def _get_retry_jitter(self, context: str, failures: int, base_delay: int) -> int:
        if base_delay <= 1:
            return 0
        max_jitter = min(15, max(1, base_delay // 5))
        seed = f"{self.camera.camera_index_code}:{context}:{failures}:{base_delay}"
        digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) % (max_jitter + 1)

    def _pick_retry_delay(self, context: str, failure_type: str, failures: int) -> int:
        if context == "standby":
            schedules = {
                "server_5xx": DEFAULT_RETRY_STANDBY_5XX,
                "timeout": DEFAULT_RETRY_STANDBY_TIMEOUT,
                "auth_or_notfound": DEFAULT_RETRY_AUTH,
                "generic": [10, 20, 40, 60],
            }
        else:
            schedules = {
                "server_5xx": DEFAULT_RETRY_5XX,
                "timeout": DEFAULT_RETRY_TIMEOUT,
                "auth_or_notfound": DEFAULT_RETRY_AUTH,
                "generic": [
                    max(1, self.restart_delay),
                    max(10, self.restart_delay * 2),
                    20,
                    30,
                ],
            }
        schedule = schedules.get(failure_type, schedules["generic"])
        base_delay = schedule[min(max(failures - 1, 0), len(schedule) - 1)]
        return base_delay + self._get_retry_jitter(context, failures, base_delay)

    def _schedule_retry(
        self,
        context: str,
        slot: Optional[SlotProcess] = None,
        exc: Optional[BaseException] = None,
    ) -> int:
        failure_text = self._read_failure_text(slot, exc)
        failure_type = self._classify_failure(failure_text)
        if context == "bootstrap":
            self.bootstrap_failures += 1
            failures = self.bootstrap_failures
            delay = self._pick_retry_delay(context, failure_type, failures)
            self.next_active_retry_at = time.monotonic() + delay
        elif context == "active_rebuild":
            self.active_rebuild_failures += 1
            failures = self.active_rebuild_failures
            delay = self._pick_retry_delay(context, failure_type, failures)
            self.next_active_retry_at = time.monotonic() + delay
        else:
            self.standby_failures += 1
            failures = self.standby_failures
            delay = self._pick_retry_delay(context, failure_type, failures)
            self.next_standby_retry_at = time.monotonic() + delay
        log(
            f"[RETRY BACKOFF] {self.camera.camera_name}: context={context}, type={failure_type}, "
            f"failures={failures}, retry_in={delay}s"
        )
        return delay

    def _reset_active_retry_state(self) -> None:
        self.bootstrap_failures = 0
        self.active_rebuild_failures = 0
        self.next_active_retry_at = 0.0

    def _reset_standby_retry_state(self) -> None:
        self.standby_failures = 0
        self.next_standby_retry_at = 0.0

    def _wait_for_retry_window(self) -> bool:
        remaining = self.next_active_retry_at - time.monotonic()
        if remaining <= 0:
            return False
        self.stop_event.wait(min(remaining, 1.0))
        return True

    def _get_refresh_jitter_seconds(self) -> int:
        if self.refresh_jitter <= 0:
            return 0
        max_jitter = min(self.refresh_jitter, max(0, self.preview_ttl - self.refresh_ahead - 1))
        if max_jitter <= 0:
            return 0
        digest = hashlib.sha1(self.camera.camera_index_code.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) % (max_jitter + 1)

    def _get_refresh_after_seconds(self) -> int:
        jitter = self._get_refresh_jitter_seconds()
        return max(1, self.preview_ttl - self.refresh_ahead - jitter)

    def _get_preview_url_limited(self, reason: str) -> str:
        log(f"[PREVIEW WAIT] {self.camera.camera_name}: reason={reason}")
        with self.preview_semaphore:
            log(f"[PREVIEW REQUEST] {self.camera.camera_name}: reason={reason}")
            return self.client.get_preview_url(
                self.camera.camera_index_code,
                stream_type=self.stream_type,
                protocol="rtsp",
                transmode=1,
                streamform="rtp",
            )

    def _build_ffmpeg_cmd(self, preview_url: str, slot_dir: Path) -> List[str]:
        segment_pattern = str(slot_dir / "%06d.ts")
        playlist_path = str(slot_dir / "index.m3u8")
        cmd = [
            self.ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-nostdin",
            "-rtsp_transport",
            "tcp",
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-i",
            preview_url,
        ]
        if self.transcode:
            cmd.extend(
                [
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-tune",
                    "zerolatency",
                    "-pix_fmt",
                    "yuv420p",
                ]
            )
        else:
            cmd.extend(["-c", "copy"])
        cmd.extend(
            [
                "-f",
                "hls",
                "-hls_time",
                str(self.hls_time),
                "-hls_list_size",
                str(self.hls_list_size),
                "-hls_flags",
                "delete_segments+omit_endlist+independent_segments",
                "-hls_segment_filename",
                segment_pattern,
                playlist_path,
            ]
        )
        return cmd

    def _spawn_slot(self, slot_name: str, preview_url: str) -> SlotProcess:
        global LAST_FFMPEG_START_AT
        slot_dir = self._slot_dir(slot_name)
        clean_hls_dir(slot_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / f"{self.camera.camera_index_code}_{slot_name}.log"
        cmd = self._build_ffmpeg_cmd(preview_url, slot_dir)
        log_file = log_path.open("wb")
        with FFMPEG_START_LOCK:
            elapsed = time.monotonic() - LAST_FFMPEG_START_AT
            wait_seconds = self.ffmpeg_start_gap - elapsed
            if wait_seconds > 0:
                self.stop_event.wait(wait_seconds)
            log(f"[FFMPEG START] {self.camera.camera_name} {slot_name}: start_gap={self.ffmpeg_start_gap:.2f}s")
            process = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
            LAST_FFMPEG_START_AT = time.monotonic()
        slot = SlotProcess(
            slot_name=slot_name,
            slot_dir=slot_dir,
            process=process,
            log_path=log_path,
            log_file=log_file,
        )
        register_slot_process(slot)
        slot.preview_url = preview_url
        slot.refresh_deadline = time.monotonic() + self._get_refresh_after_seconds()
        log(
            f"[START] {self.camera.index}. {self.camera.camera_name} -> /hls/{self.camera.camera_index_code}/index.m3u8 "
            f"({slot_name}, refresh in {max(1, int(slot.refresh_deadline - time.monotonic()))}s)"
        )
        log(
            f"[SLOT START] {self.camera.camera_name} {slot_name}: "
            f"log={log_path.name}, preview_tail=...{preview_url[-48:]}"
        )
        return slot

    def _read_slot_log_tail(self, slot: Optional[SlotProcess], max_lines: int = 12) -> str:
        if slot is None:
            return ""
        try:
            lines = slot.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return ""
        tail = lines[-max_lines:]
        return "\n".join(tail).strip()

    def _log_slot_failure_tail(self, slot: Optional[SlotProcess], prefix: str) -> None:
        tail = self._read_slot_log_tail(slot)
        if tail:
            log(f"{prefix} ffmpeg tail:\n{tail}")

    def _stop_slot(self, slot: Optional[SlotProcess]) -> None:
        if slot is None:
            return
        proc = slot.process
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    pass
        else:
            try:
                proc.wait(timeout=0)
            except Exception:
                pass
        try:
            slot.log_file.flush()
        except Exception:
            pass
        try:
            slot.log_file.close()
        except Exception:
            pass
        unregister_slot_process(slot)

    def _slot_is_healthy(self, slot_name: str) -> bool:
        playlist = self._slot_playlist(slot_name)
        if not playlist.exists():
            return False
        segment_count = count_segments_in_playlist(playlist)
        if segment_count < self.warmup_min_segments:
            return False
        latest_mtime = max(
            [playlist.stat().st_mtime] + [child.stat().st_mtime for child in self._slot_dir(slot_name).glob("*.ts")],
            default=0.0,
        )
        return latest_mtime > 0 and (time.time() - latest_mtime) < max(10, self.hls_time * self.warmup_min_segments + 4)

    def _describe_slot_health(self, slot_name: str) -> str:
        slot_dir = self._slot_dir(slot_name)
        playlist = self._slot_playlist(slot_name)
        playlist_exists = playlist.exists()
        segment_count = count_segments_in_playlist(playlist) if playlist_exists else 0
        latest_mtime = get_latest_media_mtime(slot_dir)
        media_age = (time.time() - latest_mtime) if latest_mtime > 0 else None
        healthy = self._slot_is_healthy(slot_name)
        media_age_text = f"{media_age:.1f}s" if media_age is not None else "n/a"
        return (
            f"healthy={healthy}, playlist={playlist_exists}, "
            f"segments={segment_count}, media_age={media_age_text}"
        )

    def _promote_slot(self, slot: SlotProcess) -> None:
        copy_playlist_atomic(self._slot_playlist(slot.slot_name), self.public_playlist, slot.slot_name)
        old_active = self.active_slot
        old_slot_name = self.active_slot_name
        self.active_slot = slot
        self.active_slot_name = slot.slot_name
        self.standby_slot = None
        self._reset_standby_retry_state()
        log(
            f"[SWITCH] {self.camera.camera_name} {old_slot_name} -> {slot.slot_name}, "
            f"public=/hls/{self.camera.camera_index_code}/index.m3u8"
        )
        log(
            f"[SWITCH DETAIL] {self.camera.camera_name}: "
            f"old={old_slot_name}({self._describe_slot_health(old_slot_name)}), "
            f"new={slot.slot_name}({self._describe_slot_health(slot.slot_name)})"
        )
        self._stop_slot(old_active)

    def _bootstrap_active_slot(self) -> None:
        log(f"[BOOTSTRAP] {self.camera.camera_name} requesting initial preview URL")
        preview_url = self._get_preview_url_limited("bootstrap")
        with self.warmup_semaphore:
            active = self._spawn_slot(self.active_slot_name, preview_url)
            self.active_slot = active
            if not self._wait_for_slot_ready(active):
                raise RuntimeError(f"initial slot warmup failed for {self.camera.camera_name}")
        copy_playlist_atomic(self._slot_playlist(active.slot_name), self.public_playlist, active.slot_name)
        self.ever_ready = True
        self._reset_active_retry_state()
        self._reset_standby_retry_state()
        log(f"[READY] {self.camera.camera_name} public HLS ready ({self._describe_slot_health(active.slot_name)})")

    def _wait_for_slot_ready(self, slot: SlotProcess) -> bool:
        deadline = time.monotonic() + self.warmup_timeout
        last_log_second = -1
        while not self.stop_event.is_set() and time.monotonic() < deadline:
            if slot.process.poll() is not None:
                log(
                    f"[SLOT EXIT] {self.camera.camera_name} {slot.slot_name} exited during warmup, "
                    f"health={self._describe_slot_health(slot.slot_name)}"
                )
                self._log_slot_failure_tail(slot, f"[SLOT EXIT] {self.camera.camera_name} {slot.slot_name}")
                return False
            if self._slot_is_healthy(slot.slot_name):
                log(f"[SLOT READY] {self.camera.camera_name} {slot.slot_name}: {self._describe_slot_health(slot.slot_name)}")
                return True
            elapsed = int(self.warmup_timeout - max(0, deadline - time.monotonic()))
            if elapsed != last_log_second:
                last_log_second = elapsed
                active_state = "none"
                if self.active_slot is not None:
                    active_state = self._describe_slot_health(self.active_slot.slot_name)
                log(
                    f"[SLOT WAIT] {self.camera.camera_name} {slot.slot_name}: "
                    f"{self._describe_slot_health(slot.slot_name)} | active={active_state}"
                )
            time.sleep(1)
        healthy = self._slot_is_healthy(slot.slot_name)
        if not healthy:
            log(f"[SLOT TIMEOUT] {self.camera.camera_name} {slot.slot_name}: {self._describe_slot_health(slot.slot_name)}")
            self._log_slot_failure_tail(slot, f"[SLOT TIMEOUT] {self.camera.camera_name} {slot.slot_name}")
        return healthy

    def _ensure_active_playlist_fresh(self) -> None:
        if self.active_slot and self._slot_is_healthy(self.active_slot.slot_name):
            try:
                copy_playlist_atomic(self._slot_playlist(self.active_slot.slot_name), self.public_playlist, self.active_slot.slot_name)
            except Exception:
                pass

    def _start_standby_if_needed(self) -> None:
        if self.active_slot is None or self.standby_slot is not None:
            return
        if self.preview_ttl <= 0:
            return
        if time.monotonic() < self.next_standby_retry_at:
            return
        if time.monotonic() < self.active_slot.refresh_deadline:
            return

        standby_name = self._inactive_slot_name()
        log(
            f"[ROTATE REQUEST] {self.camera.camera_name}: active={self.active_slot.slot_name} "
            f"health={self._describe_slot_health(self.active_slot.slot_name)}"
        )
        try:
            preview_url = self._get_preview_url_limited("standby")
        except Exception as exc:
            log(f"[ROTATE ERROR] {self.camera.camera_name}: standby preview request failed: {exc}")
            self._schedule_retry("standby", exc=exc)
            return
        log(
            f"[ROTATE] {self.camera.camera_name} starting standby slot {standby_name}, "
            f"preview_tail=...{preview_url[-48:]}"
        )
        with self.warmup_semaphore:
            self.standby_slot = self._spawn_slot(standby_name, preview_url)

    def _promote_standby_if_ready(self) -> None:
        if self.standby_slot is None:
            return
        standby = self.standby_slot
        if standby.process.poll() is not None:
            log(f"[STANDBY EXIT] {self.camera.camera_name} {standby.slot_name} exited before switch")
            self._log_slot_failure_tail(standby, f"[STANDBY EXIT] {self.camera.camera_name} {standby.slot_name}")
            self._schedule_retry("standby", standby)
            self._stop_slot(standby)
            self.standby_slot = None
            return

        if self.active_slot and self.active_slot.process.poll() is not None:
            log(
                f"[ACTIVE DROPPED] {self.camera.camera_name} active {self.active_slot.slot_name} died while "
                f"standby {standby.slot_name} was warming up"
            )

        if self._wait_for_slot_ready(standby):
            self._promote_slot(standby)
        else:
            log(
                f"[SWITCH ABORT] {self.camera.camera_name} standby {standby.slot_name} not ready, "
                f"active={self.active_slot.slot_name if self.active_slot else 'none'}"
            )
            self._log_slot_failure_tail(standby, f"[SWITCH ABORT] {self.camera.camera_name} {standby.slot_name}")
            self._schedule_retry("standby", standby)
            self._stop_slot(standby)
            self.standby_slot = None

    def run(self) -> None:
        self.stream_dir.mkdir(parents=True, exist_ok=True)
        clean_hls_dir(self._slot_dir("slot_a"))
        clean_hls_dir(self._slot_dir("slot_b"))
        while not self.stop_event.is_set():
            try:
                if self.active_slot is None:
                    if self._wait_for_retry_window():
                        continue
                    retry_context = "bootstrap" if not self.ever_ready else "active_rebuild"
                    try:
                        self._bootstrap_active_slot()
                    except Exception as exc:
                        log(f"[ERROR] {self.camera.camera_name}: bootstrap failed: {exc}")
                        self._log_slot_failure_tail(self.active_slot, f"[BOOTSTRAP FAIL] {self.camera.camera_name}")
                        self._schedule_retry(retry_context, self.active_slot, exc)
                        self._stop_slot(self.active_slot)
                        self.active_slot = None
                        continue

                if self.active_slot and self.active_slot.process.poll() is not None:
                    log(f"[ACTIVE EXIT] {self.camera.camera_name} active slot exited, rebuilding")
                    self._log_slot_failure_tail(self.active_slot, f"[ACTIVE EXIT] {self.camera.camera_name}")
                    self._stop_slot(self.active_slot)
                    self.active_slot = None
                    continue

                self._ensure_active_playlist_fresh()
                self._start_standby_if_needed()
                self._promote_standby_if_ready()
            except Exception as exc:
                log(f"[ERROR] {self.camera.camera_name}: {exc}")
            if self.stop_event.wait(1):
                break

        self._stop_slot(self.standby_slot)
        self._stop_slot(self.active_slot)
        self.standby_slot = None
        self.active_slot = None


class RtspRelayWorker(threading.Thread):
    """Publish one temporary Hikvision preview URL to one stable MediaMTX path."""

    def __init__(
        self,
        client: ArtemisClient,
        camera: CameraStream,
        ffmpeg_bin: str,
        stream_type: int,
        restart_delay: int,
        transcode: bool,
        log_dir: Path,
        preview_ttl: int,
        refresh_ahead: int,
        refresh_jitter: int,
        startup_timeout: int,
        preview_semaphore: threading.Semaphore,
        rtsp_publish_host: str,
        rtsp_port: int,
        ffmpeg_start_gap: float,
    ):
        super().__init__(name=f"rtsp-{camera.camera_index_code}", daemon=True)
        self.client = client
        self.camera = camera
        self.ffmpeg_bin = ffmpeg_bin
        self.stream_type = stream_type
        self.restart_delay = max(1, restart_delay)
        self.transcode = transcode
        self.log_dir = log_dir
        self.preview_ttl = max(0, preview_ttl)
        self.refresh_ahead = max(0, refresh_ahead)
        self.refresh_jitter = max(0, refresh_jitter)
        self.startup_timeout = max(2, startup_timeout)
        self.preview_semaphore = preview_semaphore
        self.rtsp_publish_host = rtsp_publish_host.strip() or "127.0.0.1"
        self.rtsp_port = int(rtsp_port)
        self.ffmpeg_start_gap = max(0.0, ffmpeg_start_gap)
        self.stop_event = threading.Event()
        self.process_slot: Optional[SlotProcess] = None
        self.failures = 0
        self.next_retry_at = 0.0
        self.refresh_deadline = 0.0

    def stop(self) -> None:
        self.stop_event.set()
        self._stop_process()

    def _rtsp_publish_url(self) -> str:
        return f"rtsp://{self.rtsp_publish_host}:{self.rtsp_port}/{self.camera.camera_index_code}"

    def _get_refresh_jitter_seconds(self) -> int:
        if self.refresh_jitter <= 0:
            return 0
        max_jitter = min(self.refresh_jitter, max(0, self.preview_ttl - self.refresh_ahead - 1))
        if max_jitter <= 0:
            return 0
        digest = hashlib.sha1(self.camera.camera_index_code.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) % (max_jitter + 1)

    def _get_refresh_after_seconds(self) -> int:
        if self.preview_ttl <= 0:
            return 0
        jitter = self._get_refresh_jitter_seconds()
        return max(10, self.preview_ttl - self.refresh_ahead - jitter)

    def _get_preview_url_limited(self, reason: str) -> str:
        log(f"[RTSP PREVIEW WAIT] {self.camera.camera_name}: reason={reason}")
        with self.preview_semaphore:
            log(f"[RTSP PREVIEW REQUEST] {self.camera.camera_name}: reason={reason}")
            return self.client.get_preview_url(
                self.camera.camera_index_code,
                stream_type=self.stream_type,
                protocol="rtsp",
                transmode=1,
                streamform="rtp",
            )

    def _build_ffmpeg_cmd(self, preview_url: str) -> List[str]:
        cmd = [
            self.ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-nostdin",
            "-rtsp_transport",
            "tcp",
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-i",
            preview_url,
            "-an",
        ]
        if self.transcode:
            cmd.extend(
                [
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-tune",
                    "zerolatency",
                    "-pix_fmt",
                    "yuv420p",
                ]
            )
        else:
            cmd.extend(["-c:v", "copy"])
        cmd.extend(["-f", "rtsp", "-rtsp_transport", "tcp", self._rtsp_publish_url()])
        return cmd

    def _spawn_process(self, preview_url: str, reason: str) -> SlotProcess:
        global LAST_FFMPEG_START_AT
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / f"{self.camera.camera_index_code}_rtsp.log"
        cmd = self._build_ffmpeg_cmd(preview_url)
        log_file = log_path.open("wb")
        with FFMPEG_START_LOCK:
            elapsed = time.monotonic() - LAST_FFMPEG_START_AT
            wait_seconds = self.ffmpeg_start_gap - elapsed
            if wait_seconds > 0:
                self.stop_event.wait(wait_seconds)
            log(
                f"[RTSP FFMPEG START] {self.camera.camera_name}: reason={reason}, "
                f"publish={self._rtsp_publish_url()}"
            )
            process = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
            LAST_FFMPEG_START_AT = time.monotonic()
        slot = SlotProcess("rtsp", self.log_dir, process, log_path, log_file)
        slot.preview_url = preview_url
        slot.refresh_deadline = time.monotonic() + self._get_refresh_after_seconds()
        register_slot_process(slot)
        return slot

    def _read_process_tail(self, max_lines: int = 16) -> str:
        slot = self.process_slot
        if slot is None:
            return ""
        try:
            lines = slot.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return ""
        return "\n".join(lines[-max_lines:]).strip()

    def _classify_failure(self, text: str) -> str:
        text = (text or "").lower()
        if (re.search(r"\b5\d\d\b", text) or "5xx" in text) and (
            "server error" in text or "internal server error" in text or "method describe failed" in text
        ):
            return "server_5xx"
        if any(token in text for token in ["401", "403", "404", "unauthorized", "forbidden", "not found"]):
            return "auth_or_notfound"
        if any(
            token in text
            for token in ["timed out", "timeout", "connection refused", "temporarily unavailable", "i/o error"]
        ):
            return "timeout"
        return "generic"

    def _retry_delay(self, failure_type: str) -> int:
        schedules = {
            "server_5xx": DEFAULT_RETRY_5XX,
            "timeout": DEFAULT_RETRY_TIMEOUT,
            "auth_or_notfound": DEFAULT_RETRY_AUTH,
            "generic": DEFAULT_RETRY_GENERIC,
        }
        schedule = schedules.get(failure_type, DEFAULT_RETRY_GENERIC)
        base_delay = schedule[min(max(self.failures - 1, 0), len(schedule) - 1)]
        max_jitter = min(15, max(1, base_delay // 5))
        seed = f"{self.camera.camera_index_code}:rtsp:{self.failures}:{base_delay}"
        digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()
        return base_delay + (int(digest[:8], 16) % (max_jitter + 1))

    def _schedule_retry(self, reason: str, exc: Optional[BaseException] = None) -> None:
        tail = self._read_process_tail()
        failure_text = "\n".join(part for part in [str(exc) if exc else "", tail] if part).strip()
        failure_type = self._classify_failure(failure_text)
        self.failures += 1
        delay = self._retry_delay(failure_type)
        self.next_retry_at = time.monotonic() + delay
        if tail:
            log(f"[RTSP FAIL TAIL] {self.camera.camera_name}:\n{tail}")
        log(
            f"[RTSP RETRY BACKOFF] {self.camera.camera_name}: reason={reason}, "
            f"type={failure_type}, failures={self.failures}, retry_in={delay}s"
        )

    def _stop_slot_object(self, slot: SlotProcess) -> None:
        proc = slot.process
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    pass
        else:
            try:
                proc.wait(timeout=0)
            except Exception:
                pass
        try:
            slot.log_file.flush()
        except Exception:
            pass
        try:
            slot.log_file.close()
        except Exception:
            pass
        unregister_slot_process(slot)

    def _stop_process(self) -> None:
        slot = self.process_slot
        if slot is None:
            return
        self._stop_slot_object(slot)
        self.process_slot = None

    def _start_or_refresh(self, reason: str) -> None:
        old_slot = self.process_slot
        try:
            preview_url = self._get_preview_url_limited(reason)
        except Exception as exc:
            if old_slot is not None and old_slot.process.poll() is None:
                self._schedule_retry(reason, exc)
                log(f"[RTSP REFRESH KEEP OLD] {self.camera.camera_name}: preview request failed, old publisher kept")
                return
            raise
        new_slot = self._spawn_process(preview_url, reason)
        self.process_slot = new_slot
        deadline = time.monotonic() + self.startup_timeout
        startup_error = ""
        while not self.stop_event.is_set() and time.monotonic() < deadline:
            if new_slot.process.poll() is not None:
                startup_error = f"ffmpeg exited during RTSP startup for {self.camera.camera_name}"
                break
            self.stop_event.wait(0.5)
        if self.stop_event.is_set():
            return
        if startup_error:
            tail = self._read_process_tail()
            self._stop_slot_object(new_slot)
            self.process_slot = old_slot if old_slot and old_slot.process.poll() is None else None
            if self.process_slot is not None:
                self._schedule_retry(reason, RuntimeError(startup_error + ("\n" + tail if tail else "")))
                log(f"[RTSP REFRESH KEEP OLD] {self.camera.camera_name}: new publisher failed, old publisher kept")
                return
            raise RuntimeError(startup_error + ("\n" + tail if tail else ""))
        if old_slot is not None:
            self._stop_slot_object(old_slot)
        self.failures = 0
        self.next_retry_at = 0.0
        self.refresh_deadline = new_slot.refresh_deadline
        log(
            f"[RTSP READY] {self.camera.index}. {self.camera.camera_name} -> "
            f"{self._rtsp_publish_url()} (refresh in {max(0, int(self.refresh_deadline - time.monotonic()))}s)"
        )

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                now = time.monotonic()
                if now < self.next_retry_at:
                    self.stop_event.wait(min(1.0, self.next_retry_at - now))
                    continue

                if self.process_slot is None:
                    self._start_or_refresh("bootstrap")
                elif self.process_slot.process.poll() is not None:
                    log(f"[RTSP EXIT] {self.camera.camera_name}: publisher exited")
                    self._schedule_retry("publisher_exit")
                    self._stop_process()
                elif self.preview_ttl > 0 and now >= self.refresh_deadline:
                    log(f"[RTSP REFRESH] {self.camera.camera_name}: refreshing temporary preview URL")
                    self._start_or_refresh("refresh")
            except Exception as exc:
                log(f"[RTSP ERROR] {self.camera.camera_name}: {exc}")
                self._schedule_retry("exception", exc)
                self._stop_process()
            if self.stop_event.wait(1):
                break
        self._stop_process()


class PrefixHlsHandler(http.server.SimpleHTTPRequestHandler):
    url_prefix = "/hls"

    def translate_path(self, path: str) -> str:
        if path.startswith(self.url_prefix + "/"):
            path = path[len(self.url_prefix):]
        return super().translate_path(path)

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, fmt: str, *args: Any) -> None:
        log("[HTTP] " + fmt % args)


def start_http_server(output_dir: Path, host: str, port: int) -> http.server.ThreadingHTTPServer:
    handler = functools.partial(PrefixHlsHandler, directory=str(output_dir))
    server = http.server.ThreadingHTTPServer((host, port), handler)
    thread = threading.Thread(target=server.serve_forever, name="hls-http", daemon=True)
    thread.start()
    return server


def camera_to_stream(index: int, camera: Dict[str, Any]) -> CameraStream:
    return CameraStream(
        index=int(camera.get("index") or index),
        camera_index_code=str(camera.get("cameraIndexCode") or ""),
        camera_name=str(camera.get("cameraName") or ""),
        gb_index_code=str(camera.get("gbIndexCode") or ""),
        channel_no=str(camera.get("channelNo") or ""),
        region_index_code=str(camera.get("regionIndexCode") or ""),
    )


def load_cameras_from_json(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, list):
        cameras = payload
    elif isinstance(payload, dict):
        cameras = payload.get("cameras")
        if cameras is None:
            cameras = (payload.get("data") or {}).get("list")
    else:
        cameras = None

    if not isinstance(cameras, list):
        raise ValueError(f"{path} must be a camera list or contain cameras/data.list")
    return [camera for camera in cameras if isinstance(camera, dict)]


def write_stream_index(cameras: Iterable[CameraStream], output_dir: Path, public_base_url: str) -> None:
    rows = []
    for camera in cameras:
        rows.append(
            {
                "index": camera.index,
                "cameraName": camera.camera_name,
                "cameraIndexCode": camera.camera_index_code,
                "gbIndexCode": camera.gb_index_code,
                "channelNo": camera.channel_no,
                "hlsUrl": f"{public_base_url.rstrip('/')}/hls/{camera.camera_index_code}/index.m3u8",
            }
        )
    (output_dir / "streams.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "streams.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "cameraName", "cameraIndexCode", "gbIndexCode", "channelNo", "hlsUrl"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_rtsp_stream_index(cameras: Iterable[CameraStream], output_dir: Path, public_rtsp_base_url: str) -> None:
    rows = []
    for camera in cameras:
        rows.append(
            {
                "index": camera.index,
                "cameraName": camera.camera_name,
                "cameraIndexCode": camera.camera_index_code,
                "gbIndexCode": camera.gb_index_code,
                "channelNo": camera.channel_no,
                "rtspUrl": f"{public_rtsp_base_url.rstrip('/')}/{camera.camera_index_code}",
            }
        )
    (output_dir / "streams_rtsp.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "streams_rtsp.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "cameraName", "cameraIndexCode", "gbIndexCode", "channelNo", "rtspUrl"],
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restream iSecure Center cameras to fixed local RTSP/HLS URLs.")
    parser.add_argument("--key-file", default=DEFAULT_KEY_FILE, help="File containing ak=... and sk=...")
    parser.add_argument("--host", default=DEFAULT_ARTEMIS_HOST, help="Artemis host:port")
    parser.add_argument("--camera-json", default=DEFAULT_CAMERA_JSON, help="Camera list JSON. If missing, fetch from OpenAPI.")
    parser.add_argument("--output-mode", default=DEFAULT_OUTPUT_MODE, choices=["rtsp", "hls"], help="Downstream output mode")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for HLS files")
    parser.add_argument("--http-host", default=DEFAULT_HTTP_HOST, help="HTTP bind host")
    parser.add_argument("--http-port", type=int, default=DEFAULT_HTTP_PORT, help="HTTP bind port")
    parser.add_argument("--public-host", default=DEFAULT_PUBLIC_HOST, help="Public host/IP used in generated stream index")
    parser.add_argument("--rtsp-publish-host", default=DEFAULT_RTSP_PUBLISH_HOST, help="MediaMTX host for ffmpeg publishing")
    parser.add_argument("--rtsp-port", type=int, default=DEFAULT_RTSP_PORT, help="MediaMTX RTSP port")
    parser.add_argument("--rtsp-startup-timeout", type=int, default=DEFAULT_RTSP_STARTUP_TIMEOUT, help="Seconds a new RTSP publisher must stay alive before it is considered ready")
    parser.add_argument("--ffmpeg", default=DEFAULT_FFMPEG, help="ffmpeg executable path")
    parser.add_argument("--all", action="store_true", help="Start all cameras")
    parser.add_argument("--camera-index-code", action="append", default=[], help="Start only the specified cameraIndexCode; can repeat")
    parser.add_argument("--name-contains", default="", help="Start cameras whose name contains this text")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of started cameras, useful for testing")
    parser.add_argument("--stream-type", type=int, default=DEFAULT_STREAM_TYPE, choices=[0, 1, 2], help="0 main stream, 1 sub stream, 2 third stream")
    parser.add_argument("--hls-time", type=int, default=2, help="HLS segment duration seconds")
    parser.add_argument("--hls-list-size", type=int, default=60, help="HLS playlist segment count")
    parser.add_argument("--restart-delay", type=int, default=5, help="Delay before rebuilding a failed camera stream")
    parser.add_argument("--preview-ttl", type=int, default=DEFAULT_PREVIEW_TTL, help="Upstream preview URL lifetime seconds")
    parser.add_argument("--refresh-ahead", type=int, default=DEFAULT_REFRESH_AHEAD, help="Refresh the upstream preview URL this many seconds before TTL")
    parser.add_argument("--refresh-jitter", type=int, default=DEFAULT_REFRESH_JITTER, help="Spread refresh start times to avoid switching all cameras together")
    parser.add_argument("--warmup-timeout", type=int, default=DEFAULT_WARMUP_TIMEOUT, help="Seconds to wait for standby HLS to produce a healthy playlist")
    parser.add_argument("--warmup-min-segments", type=int, default=DEFAULT_WARMUP_MIN_SEGMENTS, help="Minimum segment count required before promoting standby slot")
    parser.add_argument("--preview-concurrency", type=int, default=DEFAULT_PREVIEW_CONCURRENCY, help="Maximum concurrent Artemis preview URL requests")
    parser.add_argument("--warmup-concurrency", type=int, default=DEFAULT_WARMUP_CONCURRENCY, help="Maximum concurrent ffmpeg slot warmups")
    parser.add_argument("--start-stagger", type=float, default=DEFAULT_START_STAGGER, help="Seconds to wait between starting camera workers")
    parser.add_argument("--ffmpeg-start-gap", type=float, default=DEFAULT_FFMPEG_START_GAP, help="Minimum seconds between ffmpeg process starts")
    parser.add_argument("--transcode", action="store_true", help="Transcode to H264 for better browser HLS compatibility; costs CPU")
    parser.add_argument("--list-only", action="store_true", help="Only fetch cameras and write stream index; do not start ffmpeg")
    return parser.parse_args()


def select_cameras(args: argparse.Namespace, all_cameras: List[Dict[str, Any]]) -> List[CameraStream]:
    selected: List[CameraStream] = []
    wanted_codes = set(args.camera_index_code or [])
    name_filter = args.name_contains.strip()
    select_all = args.all or (not wanted_codes and not name_filter)

    for idx, camera in enumerate(all_cameras, start=1):
        stream = camera_to_stream(idx, camera)
        if not stream.camera_index_code:
            continue
        if select_all:
            selected.append(stream)
        elif wanted_codes and stream.camera_index_code in wanted_codes:
            selected.append(stream)
        elif name_filter and name_filter in stream.camera_name:
            selected.append(stream)

    if args.limit > 0:
        selected = selected[: args.limit]
    return selected


def main() -> int:
    args = parse_args()
    stop_event = threading.Event()
    keys = read_key_file(Path(args.key_file))
    ffmpeg_bin = shutil.which(args.ffmpeg) or (args.ffmpeg if Path(args.ffmpeg).exists() else "")
    if not ffmpeg_bin and not args.list_only:
        raise RuntimeError("ffmpeg not found. Install ffmpeg or pass --ffmpeg path.")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "_logs"
    setup_main_logger(log_dir / "restreamer.log")
    _reaper_thread = start_process_reaper(stop_event)

    client = ArtemisClient(args.host, keys["ak"], keys["sk"])
    camera_json = Path(args.camera_json)
    if camera_json.exists():
        all_cameras = load_cameras_from_json(camera_json)
        camera_source = str(camera_json)
    else:
        all_cameras = client.get_cameras()
        camera_source = "OpenAPI"
    selected = select_cameras(args, all_cameras)

    if not selected:
        log("No cameras selected. Check camera-json, --camera-index-code, or --name-contains.")
        log(f"Loaded {len(all_cameras)} cameras from {camera_source}.")
        return 2

    public_host = args.public_host.strip() or "127.0.0.1"
    public_base_url = f"http://{public_host}:{args.http_port}"
    public_rtsp_base_url = f"rtsp://{public_host}:{args.rtsp_port}"
    write_stream_index(selected, output_dir, public_base_url)
    write_rtsp_stream_index(selected, output_dir, public_rtsp_base_url)
    log(f"Loaded {len(all_cameras)} cameras from {camera_source}; selected {len(selected)}.")
    log(f"Stream index written: {output_dir / 'streams.csv'}")
    log(f"RTSP stream index written: {output_dir / 'streams_rtsp.csv'}")

    if args.list_only:
        return 0

    preview_semaphore = threading.Semaphore(max(1, args.preview_concurrency))
    warmup_semaphore = threading.Semaphore(max(1, args.warmup_concurrency))
    start_stagger = max(0.0, args.start_stagger)
    ffmpeg_start_gap = max(0.0, args.ffmpeg_start_gap)
    log(
        f"Rate limits: preview_concurrency={max(1, args.preview_concurrency)}, "
        f"warmup_concurrency={max(1, args.warmup_concurrency)}, "
        f"start_stagger={start_stagger:.2f}s, ffmpeg_start_gap={ffmpeg_start_gap:.2f}s"
    )

    server: Optional[http.server.ThreadingHTTPServer] = None
    if args.output_mode == "hls":
        server = start_http_server(output_dir, args.http_host, args.http_port)
        log(f"HLS server: http://{public_host}:{args.http_port}/hls/<cameraIndexCode>/index.m3u8")
        workers = [
            HlsWorker(
                client=client,
                camera=camera,
                output_dir=output_dir,
                ffmpeg_bin=ffmpeg_bin,
                stream_type=args.stream_type,
                hls_time=args.hls_time,
                hls_list_size=args.hls_list_size,
                restart_delay=args.restart_delay,
                transcode=args.transcode,
                log_dir=log_dir,
                preview_ttl=args.preview_ttl,
                refresh_ahead=args.refresh_ahead,
                refresh_jitter=args.refresh_jitter,
                warmup_timeout=args.warmup_timeout,
                warmup_min_segments=args.warmup_min_segments,
                preview_semaphore=preview_semaphore,
                warmup_semaphore=warmup_semaphore,
                ffmpeg_start_gap=ffmpeg_start_gap,
            )
            for camera in selected
        ]
    else:
        log(
            f"RTSP relay mode: publish to rtsp://{args.rtsp_publish_host}:{args.rtsp_port}/<cameraIndexCode>, "
            f"public={public_rtsp_base_url}/<cameraIndexCode>"
        )
        log("MediaMTX must be running first, and overridePublisher should be enabled for smooth refresh.")
        workers = [
            RtspRelayWorker(
                client=client,
                camera=camera,
                ffmpeg_bin=ffmpeg_bin,
                stream_type=args.stream_type,
                restart_delay=args.restart_delay,
                transcode=args.transcode,
                log_dir=log_dir,
                preview_ttl=args.preview_ttl,
                refresh_ahead=args.refresh_ahead,
                refresh_jitter=args.refresh_jitter,
                startup_timeout=args.rtsp_startup_timeout,
                preview_semaphore=preview_semaphore,
                rtsp_publish_host=args.rtsp_publish_host,
                rtsp_port=args.rtsp_port,
                ffmpeg_start_gap=ffmpeg_start_gap,
            )
            for camera in selected
        ]

    def handle_stop(signum: int, frame: Any) -> None:
        log("Stopping...")
        stop_event.set()
        for worker in workers:
            worker.stop()
        if server is not None:
            server.shutdown()

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    for worker in workers:
        worker.start()
        time.sleep(start_stagger)

    while not stop_event.is_set():
        time.sleep(1)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
