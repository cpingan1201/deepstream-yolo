#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepStream 多路视频流处理系统 v6.8 - 多摄像头版本
支持业务逻辑：检测结果处理、图片保存、上传
"""

import os
import re
import sys
import time
import json
import ssl
import uuid
import hmac
import hashlib
import base64
from datetime import datetime, timezone, timedelta
import threading
import resource
from urllib import request

# 突破Linux默认的1024文件描述符限制，支持110+路流
resource.setrlimit(resource.RLIMIT_NOFILE, (65535, 65535))

# 禁用GStreamer警告输出
os.environ['G_MESSAGES_DEBUG'] = 'none'
os.environ['GST_DEBUG'] = '0'

# 过滤nvv4l2decoder的Color primaries警告
class StderrFilter:
    def write(self, text):
        if 'Color primaries' not in text:
            sys.__stderr__.write(text)
    def flush(self):
        sys.__stderr__.flush()
sys.stderr = StderrFilter()

# 1. 先导入 GObject 和 GStreamer 基础库
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst

# 2. 【核心】必须在这里初始化 GStreamer！
Gst.init(None)

# 3. 硬件绑定库全局占位符（严格延迟到子进程绑定 GPU 后导入）
pyds = None
cv2 = None
np = None

import queue
import glob as glob_module
import requests

# Prometheus 指标模块
try:
    import prometheus_metrics as pm
    METRICS_ENABLED = True
except ImportError:
    METRICS_ENABLED = False
    print("[WARN] prometheus_metrics not found, metrics disabled")

# 北京时区
BEIJING_TZ = timezone(timedelta(hours=8))

ARTEMIS_HOST = os.getenv("ARTEMIS_HOST", "111.4.138.199:10443")
ARTEMIS_APP_KEY = os.getenv("ARTEMIS_APP_KEY", "29473298")
ARTEMIS_APP_SECRET = os.getenv("ARTEMIS_APP_SECRET", "G4vqvYG7KLMQ0KZnT4xZ")
ARTEMIS_PREVIEW_URL_API = "/artemis/api/video/v2/cameras/previewURLs"
ARTEMIS_STREAM_TYPE = int(os.getenv("ARTEMIS_STREAM_TYPE", "1"))
ARTEMIS_PREVIEW_CONCURRENCY = int(os.getenv("ARTEMIS_PREVIEW_CONCURRENCY", "2"))
ARTEMIS_PREVIEW_RETRIES = int(os.getenv("ARTEMIS_PREVIEW_RETRIES", "3"))
ARTEMIS_PREVIEW_RETRY_DELAY = float(os.getenv("ARTEMIS_PREVIEW_RETRY_DELAY", "2.0"))
ARTEMIS_PREVIEW_TIMEOUT = int(os.getenv("ARTEMIS_PREVIEW_TIMEOUT", "20"))

# 尝试加载姿态估计模型（暂时禁用，因CUDA问题）
def load_pose_model():
    print("[INFO] Pose model loading disabled (CUDA compatibility issue)")
    return False

# 输出目录
OUTPUT_DIR = "output"
DETECTION_DIR = f"{OUTPUT_DIR}/detections"
DATASET_DIR = "/workspace/dataset"

# 服务器上传配置
UPLOAD_CONFIG = {
    "enabled": False,
    "base_url": "http://umod.me:6969/api/",
    "file_upload_url": "http://umod.me:6969/api/common/upload",
    "notification_url": "http://umod.me:6969/api/notification/task/emergency",
    "app_id": "be5aab82-2fda-4f8f-a3a9-f09bf8c04909",
    "app_secret": "46b85751-44e2-40b5-b4f7-c926327656a4",
    "hls_base_url": "http://umod.me:6969/hls"
}

# OCR模型路径
OCR_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'EasyOCR-1.7.2', 'easyocr')

# 标签名称映射
LABEL_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "fire",
    5: "smoke",
    6: "animal",
    7: "cigarette",
    8: "stick"
}

# 检测阈值配置
CONFIDENCE_THRESHOLD = 0.25

# 每类检测的冷却时间（秒）
DETECTION_COOLDOWN = {
    "person": 0,      # 人员持续检测
    "car": 60,        # 车辆60秒
    "motorcycle": 60, # 摩托车60秒
    "fire": 10,      # 火警10秒
    "smoke": 10,     # 烟雾10秒
    "cigarette": 30, # 香烟30秒
    "animal": 30,    # 动物30秒
    "stick": 30,     # 棍棒30秒
}
PERSON_ALERT_STATE_TIMEOUT = float(os.getenv("PERSON_ALERT_STATE_TIMEOUT", "60"))

# VLM需要分析的检测类型
VLM_REQUIRED_TYPES = [
    'fire',
    'smoke',
    'cigarette',
    'dangerous_item',
    'power_failure',
    'floor_stuck',
    'object',
    'fall',
    'fight',
    'uncivilized_pet',
    'motor_vehicle_violations',
    'non_motor_violations',
]

# 队列与超时控制，避免高峰期无限堆积
IMAGE_SAVER_QUEUE_MAXSIZE = int(os.getenv("IMAGE_SAVER_QUEUE_MAXSIZE", "100"))
FRAME_PROCESSOR_QUEUE_MAXSIZE = int(os.getenv("FRAME_PROCESSOR_QUEUE_MAXSIZE", "5"))
VLM_PENDING_QUEUE_MAXSIZE = int(os.getenv("VLM_PENDING_QUEUE_MAXSIZE", "256"))
UPLOAD_QUEUE_MAXSIZE = int(os.getenv("UPLOAD_QUEUE_MAXSIZE", "512"))
VLM_MAX_WORKERS = int(os.getenv("VLM_MAX_WORKERS", "2"))
VLM_HTTP_TIMEOUT = int(os.getenv("VLM_HTTP_TIMEOUT", "60"))
VLM_RESULT_WAIT_TIMEOUT = int(os.getenv("VLM_RESULT_WAIT_TIMEOUT", "120"))
VLM_RESULT_POLL_INTERVAL = float(os.getenv("VLM_RESULT_POLL_INTERVAL", "0.25"))
ELEVATOR_OCR_INTERVAL = float(os.getenv("ELEVATOR_OCR_INTERVAL", "1.0"))
ELEVATOR_FRAME_MAX_AGE = float(os.getenv("ELEVATOR_FRAME_MAX_AGE", "2.0"))
ELEVATOR_QUEUE_DROP_LOG_INTERVAL = int(os.getenv("ELEVATOR_QUEUE_DROP_LOG_INTERVAL", "20"))
WATCHDOG_SOURCE_TIMEOUT = float(os.getenv("WATCHDOG_SOURCE_TIMEOUT", "60"))
WATCHDOG_PIPELINE_STALL_TIMEOUT = float(os.getenv("WATCHDOG_PIPELINE_STALL_TIMEOUT", "360"))
PROCESS_STALL_WARN_THRESHOLD = float(os.getenv("PROCESS_STALL_WARN_THRESHOLD", "15"))
WATCHDOG_STATS_INTERVAL_TICKS = int(os.getenv("WATCHDOG_STATS_INTERVAL_TICKS", "4"))
SOURCE_TIMEOUT_DETAIL_TOPN = int(os.getenv("SOURCE_TIMEOUT_DETAIL_TOPN", "5"))
REVIVE_VALIDATE_TIMEOUT = float(os.getenv("REVIVE_VALIDATE_TIMEOUT", "60"))
RTSP_REVIVE_VALIDATE_TIMEOUT = float(os.getenv("RTSP_REVIVE_VALIDATE_TIMEOUT", "15"))
REVIVE_SUCCESS_REQUIRED_STREAK = int(os.getenv("REVIVE_SUCCESS_REQUIRED_STREAK", "2"))
REVIVE_SUCCESS_IDLE_SECONDS = float(os.getenv("REVIVE_SUCCESS_IDLE_SECONDS", "3"))
REVIVE_WORKER_COUNT = int(os.getenv("REVIVE_WORKER_COUNT", "2"))
SOURCE_INITIAL_CONNECT_TIMEOUT = float(os.getenv("SOURCE_INITIAL_CONNECT_TIMEOUT", "180"))
SOURCE_START_STAGGER = float(os.getenv("SOURCE_START_STAGGER", "0.2"))
WATCHDOG_REVIVE_BATCH_LIMIT = int(os.getenv("WATCHDOG_REVIVE_BATCH_LIMIT", "8"))
RTSP_LINK_LOG = os.getenv("RTSP_LINK_LOG", "0").lower() in ("1", "true", "yes", "on")
RTSP_PAD_SKIP_LOG = os.getenv("RTSP_PAD_SKIP_LOG", "0").lower() in ("1", "true", "yes", "on")
BUS_RESOURCE_ERROR_LOG_INTERVAL = float(os.getenv("BUS_RESOURCE_ERROR_LOG_INTERVAL", "30"))
BUS_DATA_STREAM_WARN_LOG_INTERVAL = float(os.getenv("BUS_DATA_STREAM_WARN_LOG_INTERVAL", "30"))
BUS_RESOURCE_REQUEUE_LOG_INTERVAL = float(os.getenv("BUS_RESOURCE_REQUEUE_LOG_INTERVAL", "60"))
MANAGER_RESTART_COOLDOWN = float(os.getenv("MANAGER_RESTART_COOLDOWN", "15"))
MANAGER_KILL_RESTART_COOLDOWN = float(os.getenv("MANAGER_KILL_RESTART_COOLDOWN", "20"))
WORKER_MAX_UPTIME_SECONDS = float(os.getenv("WORKER_MAX_UPTIME_SECONDS", str(12 * 60 * 60)))
WORKER_ROTATE_EXIT_CODE = int(os.getenv("WORKER_ROTATE_EXIT_CODE", "88"))
EXPECTED_SOURCE_FPS = float(os.getenv("EXPECTED_SOURCE_FPS", "25"))
DECODE_LOW_FPS_RATIO = float(os.getenv("DECODE_LOW_FPS_RATIO", "0.8"))
DECODE_STATS_TOPN = int(os.getenv("DECODE_STATS_TOPN", "8"))
FRAME_DELAY_WARN_SECONDS = float(os.getenv("FRAME_DELAY_WARN_SECONDS", "1.0"))
PTS_PROGRESS_WARN_RATIO = float(os.getenv("PTS_PROGRESS_WARN_RATIO", "0.9"))
SOURCE_QUEUE_TIME_WARN_SECONDS = float(os.getenv("SOURCE_QUEUE_TIME_WARN_SECONDS", "0.5"))
FIRE_SMOKE_DURATION_THRESHOLD = float(os.getenv("FIRE_SMOKE_DURATION_THRESHOLD", "3.0"))
FIRE_SMOKE_ALERT_COOLDOWN = float(os.getenv("FIRE_SMOKE_ALERT_COOLDOWN", "30"))
FIRE_SMOKE_GAP_RESET_TIMEOUT = float(os.getenv("FIRE_SMOKE_GAP_RESET_TIMEOUT", "8.0"))
FIRE_SMOKE_CLEANUP_TIMEOUT = float(os.getenv("FIRE_SMOKE_CLEANUP_TIMEOUT", "60"))
VEHICLE_VIOLATION_DURATION_THRESHOLD = float(os.getenv("VEHICLE_VIOLATION_DURATION_THRESHOLD", "30.0"))
FIGHT_MIN_BBOX_HEIGHT = float(os.getenv("FIGHT_MIN_BBOX_HEIGHT", "60"))
FIGHT_MIN_BBOX_AREA = float(os.getenv("FIGHT_MIN_BBOX_AREA", "2500"))
FIGHT_CLOSE_DISTANCE_RATIO = float(os.getenv("FIGHT_CLOSE_DISTANCE_RATIO", "0.85"))
FIGHT_FAR_DISTANCE_RATIO = float(os.getenv("FIGHT_FAR_DISTANCE_RATIO", "1.15"))
FIGHT_MOTION_RATIO = float(os.getenv("FIGHT_MOTION_RATIO", "0.18"))
FIGHT_MIN_MOTION_PIXELS = float(os.getenv("FIGHT_MIN_MOTION_PIXELS", "12"))
FIGHT_IOU_THRESHOLD = float(os.getenv("FIGHT_IOU_THRESHOLD", "0.03"))
FIGHT_STRONG_IOU_THRESHOLD = float(os.getenv("FIGHT_STRONG_IOU_THRESHOLD", "0.10"))
FIGHT_EDGE_GAP_RATIO = float(os.getenv("FIGHT_EDGE_GAP_RATIO", "0.18"))
FIGHT_STRONG_EDGE_GAP_RATIO = float(os.getenv("FIGHT_STRONG_EDGE_GAP_RATIO", "0.08"))
FIGHT_CONTACT_DURATION = float(os.getenv("FIGHT_CONTACT_DURATION", "0.8"))
FIGHT_MUTUAL_MOTION_DURATION = float(os.getenv("FIGHT_MUTUAL_MOTION_DURATION", "0.45"))
FIGHT_PAIR_COOLDOWN = float(os.getenv("FIGHT_PAIR_COOLDOWN", "90"))
FIGHT_HISTORY_SIZE = int(os.getenv("FIGHT_HISTORY_SIZE", "16"))
FIGHT_HISTORY_MIN_SAMPLES = int(os.getenv("FIGHT_HISTORY_MIN_SAMPLES", "6"))
FIGHT_SWING_WINDOW = int(os.getenv("FIGHT_SWING_WINDOW", "8"))
FIGHT_SWING_DELTA_RATIO = float(os.getenv("FIGHT_SWING_DELTA_RATIO", "0.08"))
FIGHT_SWING_REQUIRED = int(os.getenv("FIGHT_SWING_REQUIRED", "2"))
FIGHT_STATE_MAX_STEP_SECONDS = float(os.getenv("FIGHT_STATE_MAX_STEP_SECONDS", "0.20"))
FIGHT_STATE_DECAY = float(os.getenv("FIGHT_STATE_DECAY", "1.5"))

# 这些类别后续可能需要截图或做图像级分析，因此在 probe 内同步抓帧
FRAME_SNAPSHOT_REQUIRED_CLASS_IDS = {0, 1, 2, 3, 4, 5, 7, 8}


class ArtemisPreviewClient:
    def __init__(self, host=ARTEMIS_HOST, app_key=ARTEMIS_APP_KEY, app_secret=ARTEMIS_APP_SECRET, timeout=ARTEMIS_PREVIEW_TIMEOUT):
        self.host = host.strip().replace("https://", "").replace("http://", "").rstrip("/")
        self.base_url = "https://" + self.host
        self.timeout = timeout
        self.ssl_context = ssl._create_unverified_context()
        self.app_key = app_key
        self.app_secret = app_secret
        self.lock = threading.Lock()

    def _post(self, path, body_obj):
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
                "X-Ca-Signature": base64.b64encode(signature).decode("ascii"),
            },
        )
        with request.urlopen(req, timeout=self.timeout, context=self.ssl_context) as resp:
            payload = resp.read().decode("utf-8")
        return json.loads(payload)

    def get_preview_url(self, camera_index_code):
        result = self._post(
            ARTEMIS_PREVIEW_URL_API,
            {
                "cameraIndexCode": camera_index_code,
                "streamType": ARTEMIS_STREAM_TYPE,
                "protocol": "rtsp",
                "transmode": 1,
                "expand": "transcode=0",
                "streamform": "rtp",
            },
        )
        if str(result.get("code")) != "0":
            raise RuntimeError(f"{result.get('code')} {result.get('msg')}")
        url = (result.get("data") or {}).get("url")
        if not url:
            raise RuntimeError(f"preview URL missing: {result}")
        return str(url)

    def get_preview_url_with_retry(self, camera_index_code, camera_name=""):
        last_error = None
        for attempt in range(1, ARTEMIS_PREVIEW_RETRIES + 1):
            try:
                with self.lock:
                    return self.get_preview_url(camera_index_code)
            except Exception as exc:
                last_error = exc
                print(f"[Artemis] {camera_name or camera_index_code} 获取openUrl失败 {attempt}/{ARTEMIS_PREVIEW_RETRIES}: {exc}", flush=True)
                if attempt < ARTEMIS_PREVIEW_RETRIES:
                    time.sleep(ARTEMIS_PREVIEW_RETRY_DELAY * attempt)
        raise RuntimeError(last_error)


def normalize_camera_configs(config):
    raw_cameras = config.get("cameras", {})
    if isinstance(raw_cameras, list):
        items = [(f"camera_{idx:03d}", cam) for idx, cam in enumerate(raw_cameras, start=1)]
    elif isinstance(raw_cameras, dict):
        items = list(raw_cameras.items())
    else:
        raise ValueError("config.cameras must be a list or dict")

    normalized = {}
    for idx, (cam_key, cam_data) in enumerate(items, start=1):
        if not isinstance(cam_data, dict):
            continue
        code = str(cam_data.get("cameraIndexCode") or cam_data.get("name") or "").strip()
        if not code:
            continue
        camera_name = str(
            cam_data.get("name")
            or cam_data.get("display_name")
            or cam_data.get("cameraName")
            or cam_key
        ).strip()
        cfg = dict(cam_data)
        cfg["index"] = int(cfg.get("index") or idx)
        cfg["cameraIndexCode"] = code
        cfg["name"] = camera_name
        cfg.pop("display_name", None)
        cfg.pop("cameraName", None)
        cfg.pop("encoding", None)
        normalized[f"camera_{idx:03d}"] = cfg
    return normalized


def build_cameras_data_from_config(camera_configs, preview_client=None):
    cameras_data = []
    for cam_key, cam_data in camera_configs.items():
        code = cam_data.get("cameraIndexCode")
        camera_name = cam_data.get("name", cam_key)
        if not code:
            print(f"[Artemis] 跳过 {camera_name}: 缺少 cameraIndexCode", flush=True)
            continue
        cameras_data.append({
            "url": f"rtsp://127.0.0.1:1/{code}",
            "name": camera_name,
            "cameraIndexCode": code,
            "camera_key": code,
        })
        if len(cameras_data) % 50 == 0:
            print(f"[Artemis] loaded {len(cameras_data)} cameraIndexCode entries", flush=True)
    return cameras_data

# ==============================================================================
# 检测器类 (简化版)
# ==============================================================================

class FireSmokeDetector:
    """火焰烟雾检测器 - 完整版"""

    def __init__(
        self,
        duration_threshold=FIRE_SMOKE_DURATION_THRESHOLD,
        alert_cooldown=FIRE_SMOKE_ALERT_COOLDOWN,
        gap_reset_timeout=FIRE_SMOKE_GAP_RESET_TIMEOUT,
        cleanup_timeout=FIRE_SMOKE_CLEANUP_TIMEOUT,
    ):
        self.duration_threshold = duration_threshold
        self.alert_cooldown = alert_cooldown
        self.gap_reset_timeout = gap_reset_timeout
        self.cleanup_timeout = cleanup_timeout
        self.states = {}

    def process(self, label, cam, conf, display_cam=None):
        """处理火焰/烟雾检测"""
        current_time = time.time()
        key = f"{cam}_{label}"
        log_cam = display_cam or cam
        state = self.states.get(key)

        if state is None or (current_time - state['last_seen']) > self.gap_reset_timeout:
            self.states[key] = {
                'start_time': current_time,
                'last_seen': current_time,
                'last_alert_time': 0.0,
            }
            return False

        state['last_seen'] = current_time
        duration = current_time - state['start_time']
        if duration < self.duration_threshold:
            return False
        if current_time - state['last_alert_time'] < self.alert_cooldown:
            return False

        state['last_alert_time'] = current_time
        print(f"[{label.upper()}] {log_cam}: 检测持续 {duration:.1f} 秒，触发告警")
        return True

        return False

    def cleanup(self):
        """清理超时记录"""
        current_time = time.time()
        keys = list(self.states.keys())
        for key in keys:
            if current_time - self.states[key]['last_seen'] > self.cleanup_timeout:
                del self.states[key]


class FallVideoBuffer:
    """Fall detection buffer - 摔倒检测缓冲区"""

    def __init__(self, tid, start_f, fall_angle_threshold=50.0, min_frames_required=5, baseline_angle=0.0,
                 angle_change_threshold=15.0, fallen_duration_required=1.5, height_drop_ratio=0.2):
        self.tid = tid
        self.max_len = 60
        self.history = []
        self.confirmed = False
        self.last_time = time.time()
        self.fall_angle_threshold = fall_angle_threshold
        self.min_frames_required = min_frames_required
        self.baseline_angle = baseline_angle
        self.state = 'standing'
        self.fall_start_time = None
        self.standing_angle_avg = None
        self.angle_history = []
        self.angle_change_threshold = angle_change_threshold
        self.fallen_duration_required = fallen_duration_required
        self.height_drop_ratio = height_drop_ratio

    def update(self, pose, frame, fnum):
        """更新缓冲区"""
        self.last_time = time.time()
        angle = self._calculate_spine_angle(pose)
        height = self._calculate_body_height(pose)

        self.history.append({
            'pose': pose,
            'angle': angle,
            'frame': frame.copy(),
            'height': height,
            'time': time.time()
        })
        if len(self.history) > self.max_len:
            self.history.pop(0)

        if angle is not None:
            self.angle_history.append({'angle': angle, 'time': time.time()})
            if len(self.angle_history) > 30:
                self.angle_history.pop(0)

            if angle < 30 and self.state == 'standing':
                if self.standing_angle_avg is None:
                    self.standing_angle_avg = angle
                else:
                    self.standing_angle_avg = self.standing_angle_avg * 0.9 + angle * 0.1

    def _calculate_spine_angle(self, kpts):
        """计算脊柱角度"""
        if kpts[5][0]==0 or kpts[6][0]==0 or kpts[11][0]==0 or kpts[12][0]==0:
            return None
        mid_sh = (kpts[5] + kpts[6]) / 2
        mid_hip = (kpts[11] + kpts[12]) / 2
        dx, dy = mid_hip[0] - mid_sh[0], mid_hip[1] - mid_sh[1]
        if dy == 0 and dx == 0:
            return None
        angle = math.degrees(math.atan2(abs(dx), abs(dy)))
        adj_angle = abs(angle - self.baseline_angle)
        return 180 - adj_angle if adj_angle > 90 else adj_angle

    def _calculate_body_height(self, kpts):
        """计算身体高度"""
        head_y = None
        foot_y = None
        for idx in [0, 1, 2, 5, 6]:
            if kpts[idx][1] > 0:
                if head_y is None or kpts[idx][1] < head_y:
                    head_y = kpts[idx][1]
        for idx in [15, 16, 13, 14]:
            if kpts[idx][1] > 0:
                if foot_y is None or kpts[idx][1] > foot_y:
                    foot_y = kpts[idx][1]
        if head_y is not None and foot_y is not None:
            return foot_y - head_y
        return None

    def _calculate_angle_change_rate(self):
        """计算角度变化率"""
        if len(self.angle_history) < 3:
            return 0
        now = time.time()
        recent = [h for h in self.angle_history if now - h['time'] < 1.0]
        if len(recent) < 2:
            return 0
        angle_diff = recent[-1]['angle'] - recent[0]['angle']
        time_diff = recent[-1]['time'] - recent[0]['time']
        if time_diff > 0:
            return angle_diff / time_diff
        return 0

    def _check_geometry_fall(self, pose):
        """几何检查：肩膀必须在臀部下方"""
        shoulder_y = (pose[5][1] + pose[6][1]) / 2
        hip_y = (pose[11][1] + pose[12][1]) / 2
        return shoulder_y > hip_y

    def _check_height_drop_strict(self):
        """严格版高度下降检测"""
        if len(self.history) < 20:
            return True

        recent_heights = [h['height'] for h in self.history[-5:] if h['height'] is not None]
        older_heights = [h['height'] for h in self.history[-30:-5] if h['height'] is not None]

        if not recent_heights or not older_heights:
            return True

        avg_recent = sum(recent_heights) / len(recent_heights)
        avg_older = sum(older_heights) / len(older_heights)

        return avg_older > 0 and (avg_older - avg_recent) / avg_older > 0.25

    def check_fall(self, w, h):
        """检查是否摔倒"""
        if len(self.history) < 15:
            return False

        curr = self.history[-1]
        if curr['angle'] is None:
            return False

        pose = curr['pose']
        valid = pose[(pose[:,0]>0) & (pose[:,1]>0)]
        if len(valid) < 6:
            return False

        # 检查关键点完整性
        has_shoulders = (pose[5][0] > 0 and pose[5][1] > 0) and (pose[6][0] > 0 and pose[6][1] > 0)
        has_hips = (pose[11][0] > 0 and pose[11][1] > 0) and (pose[12][0] > 0 and pose[12][1] > 0)
        has_left_leg = (pose[13][0] > 0 and pose[13][1] > 0) or (pose[15][0] > 0 and pose[15][1] > 0)
        has_right_leg = (pose[14][0] > 0 and pose[14][1] > 0) or (pose[16][0] > 0 and pose[16][1] > 0)
        has_legs = has_left_leg or has_right_leg

        if not (has_shoulders and has_hips and has_legs):
            return False

        # 边界检查
        min_x, max_x = valid[:,0].min(), valid[:,0].max()
        min_y, max_y = valid[:,1].min(), valid[:,1].max()
        mx, my = w * 0.03, h * 0.03
        if min_x < mx or max_x > w-mx or min_y < my or max_y > h-my:
            return False

        now = time.time()
        curr_angle = curr['angle']
        angle_change_rate = self._calculate_angle_change_rate()

        # 宽松版摔倒检测
        if curr_angle <= 60:
            self.state = 'standing'
            self.fall_start_time = None
            return False

        if not self._check_height_drop_strict():
            self.state = 'standing'
            return False

        if angle_change_rate < 10 or angle_change_rate > 60:
            self.state = 'standing'
            return False

        if not self._check_geometry_fall(pose):
            self.state = 'standing'
            return False

        # 状态机
        if self.state == 'standing':
            self.state = 'falling'
            self.fall_start_time = now
            print(f"[摔倒检测] ID:{self.tid} 检测到潜在摔倒信号")

        elif self.state == 'falling':
            duration = now - self.fall_start_time
            if duration >= 1.0:
                self.state = 'fallen'
                print(f"[摔倒检测] ID:{self.tid} 确认摔倒！持续{duration:.1f}秒")
                return True
            if curr_angle < 50:
                self.state = 'standing'
                self.fall_start_time = None

        elif self.state == 'fallen':
            pass

        return False

    def get_sequence(self):
        """获取视频序列"""
        return [(d['frame'], f"fall_{self.tid}_{i:03d}.jpg") for i, d in enumerate(self.history[-30:])]


class FightBuffer:
    """Fight detection buffer - 打架检测缓冲区"""

    def __init__(self, tid, min_frames_required=5, interaction_threshold=40.0):
        self.tid = tid
        self.max_len = 60
        self.history = []
        self.confirmed = False
        self.last_time = time.time()
        self.state = 'normal'
        self.interaction_start_time = None
        self.min_frames_required = min_frames_required
        self.interaction_threshold = interaction_threshold
        self.prev_poses = {}

    def update(self, pose, other_poses, frame, fnum):
        """更新状态"""
        self.last_time = time.time()

        self.history.append({
            'pose': pose,
            'other_poses': other_poses,
            'frame': frame.copy(),
            'time': time.time()
        })
        if len(self.history) > self.max_len:
            self.history.pop(0)

        self.prev_poses['self'] = pose
        if other_poses:
            self.prev_poses['other'] = other_poses[0] if len(other_poses) > 0 else None

    def _calculate_distance(self, pose1, pose2):
        """计算两人中心点距离"""
        def get_center(pose):
            x_coords = [p[0] for p in pose if p[0] > 0]
            y_coords = [p[1] for p in pose if p[1] > 0]
            if not x_coords or not y_coords:
                return None
            return (sum(x_coords)/len(x_coords), sum(y_coords)/len(y_coords))

        c1 = get_center(pose1)
        c2 = get_center(pose2)
        if c1 is None or c2 is None:
            return float('inf')

        dx, dy = c1[0]-c2[0], c1[1]-c2[1]
        return (dx**2 + dy**2)**0.5

    def _calc_motion(self, pose, prev_pose):
        """计算上半身运动量"""
        if prev_pose is None:
            return 0
        upper_body_indices = [5, 6, 7, 8, 11, 12]
        motion = sum(
            abs(pose[i][0] - prev_pose[i][0]) + abs(pose[i][1] - prev_pose[i][1])
            for i in upper_body_indices
            if pose[i][0] > 0 and prev_pose[i][0] > 0
        )
        return motion / len([i for i in upper_body_indices if pose[i][0] > 0 and prev_pose[i][0] > 0])

    def _detect_mutual_high_motion(self):
        """检测双方高运动"""
        if len(self.history) < 8:
            return 0

        recent = self.history[-8:]
        mutual_high_motion_frames = 0

        for i, f in enumerate(recent):
            if i == 0:
                continue

            pose = f['pose']
            other_poses = f.get('other_poses', [])

            for op in other_poses:
                motion_self = self._calc_motion(pose, self.history[i-1]['pose'] if i > 0 else None)
                motion_other = self._calc_motion(op, self.history[i-1].get('other_poses', [None])[0] if i > 0 else None)

                if motion_self > 50 and motion_other > 50:
                    mutual_high_motion_frames += 1
                    break

        return mutual_high_motion_frames

    def _detect_close_proximity_strict(self):
        """近距离检测"""
        if len(self.history) < 8:
            return False

        recent = self.history[-8:]
        close_frames = 0

        for f in recent:
            pose = f['pose']
            other_poses = f.get('other_poses', [])

            for op in other_poses:
                dist = self._calculate_distance(pose, op)
                if dist < 50:  # 宽松：50像素
                    close_frames += 1
                    break

        return close_frames >= 4

    def check_fight(self):
        """检查是否打架"""
        if len(self.history) < 8:
            return False

        # 条件1：近距离接触
        close_proximity = self._detect_close_proximity_strict()

        # 条件2：双方高运动
        mutual_motion = self._detect_mutual_high_motion()

        # 状态机
        if self.state == 'normal':
            if close_proximity and mutual_motion >= 2:
                self.state = 'interacting'
                self.interaction_start_time = time.time()
                print(f"[打架检测] ID:{self.tid} 检测到疑似打架")

        elif self.state == 'interacting':
            duration = time.time() - self.interaction_start_time
            if duration >= 1.5:  # 1.5秒确认
                print(f"[打架检测] ID:{self.tid} 确认打架！持续{duration:.1f}秒")
                return True
            if not close_proximity:
                self.state = 'normal'
                self.interaction_start_time = None

        return False


class DangerousItemBuffer:
    """危险物品检测缓冲区"""

    def __init__(self, tid, item_type=None, duration_threshold=0.5):
        self.tid = tid
        self.item_type = item_type
        self.duration_threshold = duration_threshold
        self.first_seen = None
        self.last_seen = None

    def update(self, item_detected, item_bbox, person_pose, frame, fnum):
        """更新缓冲区"""
        current_time = time.time()

        if item_detected:
            if self.first_seen is None:
                self.first_seen = current_time
            self.last_seen = current_time

            # 检查持续时间
            if self.last_seen - self.first_seen >= self.duration_threshold:
                return True  # 检测到危险物品

        return False

    def check_dangerous_item(self):
        """检查危险物品"""
        if self.first_seen and self.last_seen:
            return (self.last_seen - self.first_seen) >= self.duration_threshold
        return False


# ==============================================================================
# OCR 识别模块
# ==============================================================================

class OCRProcessor:
    """OCR识别处理器"""

    def __init__(self, model_path=OCR_MODEL_PATH):
        self.reader = None
        self.model_path = model_path

    def init(self):
        """初始化OCR"""
        if self.reader is not None:
            return True

        try:
            # 设置模型路径
            os.environ['EASYOCR_MODULE_PATH'] = self.model_path
            import easyocr
            self.reader = easyocr.Reader(['en'], gpu=True)
            print("[OCR] EasyOCR 初始化成功 (GPU版本)")
            return True
        except Exception as e:
            print(f"[OCR ERROR] EasyOCR 初始化失败: {e}")
            return False

    def recognize(self, image):
        """识别文字"""
        if self.reader is None:
            if not self.init():
                return None, 0.0

        try:
            results = self.reader.readtext(image)
            if not results:
                return None, 0.0

            best_result = None
            best_confidence = 0.0

            for (bbox, text, confidence) in results:
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_result = text

            return best_result, best_confidence
        except Exception as e:
            print(f"[OCR ERROR] {e}")
            return None, 0.0


# ==============================================================================
# 上传模块
# ==============================================================================

class Uploader:
    """检测结果上传器"""

    def __init__(self, config=UPLOAD_CONFIG, camera_configs=None):
        self.config = config
        self.camera_configs = camera_configs or {}
        self.enabled = config.get("enabled", False)

        if self.enabled:
            self.auth_header = {
                "X-App-Id": config.get("app_id", ""),
                "X-App-Secret": config.get("app_secret", "")
            }
            print(f"[UPLOAD] 上传功能已启用: {config.get('base_url', '')}")

    def upload_file(self, file_path, folder_type):
        """上传文件"""
        if not self.enabled or not os.path.exists(file_path):
            return False

        try:
            import requests
            url = self.config.get("file_upload_url", "")

            with open(file_path, 'rb') as f:
                files = {'file': (os.path.basename(file_path), f, 'image/jpeg')}
                response = requests.post(
                    url,
                    files=files,
                    headers=self.auth_header,
                    timeout=30
                )

            if response.status_code == 200:
                print(f"[UPLOAD SUCCESS] {file_path}")
                return True
            else:
                print(f"[UPLOAD FAILED] {response.status_code}")
                return False
        except Exception as e:
            print(f"[UPLOAD ERROR] {e}")
            return False

    def send_notification(self, camera_name, detection_type, file_path="", vlm_result=""):
        """发送告警通知"""
        if not self.enabled:
            return

        try:
            import requests
            url = self.config.get("notification_url", "")

            # 获取摄像头IP
            camera_ip = ""
            for cam_key, cam_config in self.camera_configs.items():
                if cam_config.get("name") == camera_name:
                    if cam_config.get("rtsp_url") and "@" in cam_config.get("rtsp_url", ""):
                        camera_ip = cam_config.get("rtsp_url", "").split("@")[1].split(":")[0]
                    else:
                        camera_ip = cam_config.get("cameraIndexCode", "")
                    break

            data = {
                "camera_name": camera_name,
                "camera_ip": camera_ip,
                "alert_type": detection_type,
                "description": vlm_result,
                "file_path": file_path
            }

            response = requests.post(
                url,
                json=data,
                headers=self.auth_header,
                timeout=10
            )

            if response.status_code == 200:
                print(f"[NOTIFY SUCCESS] {camera_name}: {detection_type}")
            else:
                print(f"[NOTIFY FAILED] {response.status_code}")
        except Exception as e:
            print(f"[NOTIFY ERROR] {e}")
        """检查危险物品"""
        if self.first_seen and self.last_seen:
            return (self.last_seen - self.first_seen) >= self.duration_threshold
        return False


# 告警图片保存目录
ALARM_IMG_DIR = os.path.join(OUTPUT_DIR, "alarms")
DANGEROUS_ITEM_DIR = os.path.join(OUTPUT_DIR, "dangerous_item_detections")
FIRE_DIR = os.path.join(OUTPUT_DIR, "fire_detections")
SMOKE_DIR = os.path.join(OUTPUT_DIR, "smoke_detections")
CIGARETTE_DIR = os.path.join(OUTPUT_DIR, "cigarette_detections")
MOTOR_DIR = os.path.join(OUTPUT_DIR, "motor_vehicle_violations")
NON_MOTOR_DIR = os.path.join(OUTPUT_DIR, "non_motor_violations")
FALL_DIR = os.path.join(OUTPUT_DIR, "fall_detections")
FIGHT_DIR = os.path.join(OUTPUT_DIR, "fight_detections")
PET_DIR = os.path.join(OUTPUT_DIR, "pet_detections")
ELEVATOR_POWER_DIR = os.path.join(OUTPUT_DIR, "elevator_power_failures")
ELEVATOR_FLOOR_DIR = os.path.join(OUTPUT_DIR, "elevator_floor_stuck")
OBJECT_DIR = os.path.join(OUTPUT_DIR, "object_detections")
os.makedirs(ALARM_IMG_DIR, exist_ok=True)
os.makedirs(DANGEROUS_ITEM_DIR, exist_ok=True)
os.makedirs(FIRE_DIR, exist_ok=True)
os.makedirs(SMOKE_DIR, exist_ok=True)
os.makedirs(CIGARETTE_DIR, exist_ok=True)
os.makedirs(MOTOR_DIR, exist_ok=True)
os.makedirs(NON_MOTOR_DIR, exist_ok=True)
os.makedirs(FALL_DIR, exist_ok=True)
os.makedirs(FIGHT_DIR, exist_ok=True)
os.makedirs(PET_DIR, exist_ok=True)
os.makedirs(ELEVATOR_POWER_DIR, exist_ok=True)
os.makedirs(ELEVATOR_FLOOR_DIR, exist_ok=True)
os.makedirs(OBJECT_DIR, exist_ok=True)


class ElevatorStatusManager:
    """电梯状态管理器 - 基于OCR检测楼层号判断停电和故障"""
    def __init__(self):
        self.camera_status = {}

    def init_camera(self, camera_name, display_area):
        if camera_name not in self.camera_status:
            self.camera_status[camera_name] = {
                'display_area': display_area,
                'last_floor_number': None,
                'people_count': 0,
                'success_count': 0,
                'fail_count': 0,
                'power_failure_reported': False,
                'floor_stuck_reported': False
            }

    def update_person_count(self, camera_name, count):
        if camera_name not in self.camera_status:
            return
        status = self.camera_status[camera_name]
        prev_count = status.get('people_count', 0)
        status['people_count'] = count

        if prev_count == 0 and count > 0:
            status['success_count'] = 0
            status['fail_count'] = 0
            status['power_failure_reported'] = False
        elif prev_count > 0 and count == 0:
            status['last_floor_number'] = None
            status['success_count'] = 0
            status['fail_count'] = 0
            status['power_failure_reported'] = False
            status['floor_stuck_reported'] = False

    def update_floor_detection(self, camera_name, floor_number, confidence, current_time):
        if camera_name not in self.camera_status:
            return None

        status = self.camera_status[camera_name]

        if status['people_count'] == 0:
            if status['last_floor_number'] is not None or status['success_count'] > 0 or status['fail_count'] > 0:
                status['last_floor_number'] = None
                status['success_count'] = 0
                status['fail_count'] = 0
                status['floor_stuck_reported'] = False
                status['power_failure_reported'] = False
            return None

        if floor_number is not None and confidence > 0.1:
            status['fail_count'] = 0

            if status['last_floor_number'] != floor_number:
                status['last_floor_number'] = floor_number
                status['success_count'] = 1
                status['floor_stuck_reported'] = False
                status['power_failure_reported'] = False
            else:
                status['success_count'] += 1
                if status['success_count'] > 15 and not status['floor_stuck_reported']:
                    status['floor_stuck_reported'] = True
                    return 'floor_stuck'

            return None

        status['success_count'] = 0
        status['fail_count'] += 1
        status['last_floor_number'] = None
        status['floor_stuck_reported'] = False

        # 电梯停电检测已禁用
        # if status['fail_count'] > 15 and not status['power_failure_reported']:
        #     status['power_failure_reported'] = True
        #     return 'power_failure'

        return None


class AsyncElevatorDetector:
    """异步电梯检测器 - OCR和YOLO-World检测在独立线程执行，不阻塞GStreamer"""
    def __init__(self, elevator_status, camera_configs, image_saver=None):
        self.elevator_status = elevator_status
        self.camera_configs = camera_configs
        self.image_saver = image_saver  # ImageSaver引用，用于直接保存图片
        self.ocr_queue = queue.Queue(maxsize=50)  # OCR检测队列
        self.obj_queue = queue.Queue(maxsize=50)   # YOLO-World检测队列
        self.result_queue = queue.Queue(maxsize=100)  # 结果队列（仅用于统计）
        self.running = True
        self.ocr_dropped_count = 0
        self.obj_dropped_count = 0
        self.ocr_ready = False
        self.yolo_world_ready = False
        self.ocr_reader = None
        self._yolo_world = None
        self._target_classes = {'chair', 'stool', 'bottle', 'box', 'bag', 'backpack', 'cup', 'umbrella', 'knife', 'scissors'}

        # 确保HOME指向有模型的位置
        os.environ['HOME'] = '/root'

        # 先在主线程初始化模型
        self.init_ocr()
        self.init_yolo_world()

        # 启动工作线程
        self.ocr_thread = threading.Thread(target=self._ocr_worker, daemon=True)
        self.obj_thread = threading.Thread(target=self._obj_worker, daemon=True)
        self.result_thread = threading.Thread(target=self._result_worker, daemon=True)
        self.ocr_thread.start()
        self.obj_thread.start()
        self.result_thread.start()
        print("[AsyncElevatorDetector] 异步电梯检测器已启动")

    def update_queue_metrics(self):
        if METRICS_ENABLED:
            pm.QUEUE_SIZE.labels(queue_name='elevator_ocr').set(self.ocr_queue.qsize())
            pm.QUEUE_SIZE.labels(queue_name='elevator_obj').set(self.obj_queue.qsize())

    def _record_queue_drop(self, queue_name, dropped_count, camera_name):
        if METRICS_ENABLED:
            pm.QUEUE_DROPPED.labels(queue_name=queue_name).inc()
        if dropped_count == 1 or dropped_count % ELEVATOR_QUEUE_DROP_LOG_INTERVAL == 0:
            print(f"[{queue_name}] 队列已满，累计丢弃 {dropped_count} 帧: {camera_name}")

    def init_ocr(self):
        """初始化OCR"""
        if self.ocr_ready:
            return True
        try:
            import easyocr
            # 使用绝对路径 /workspace/EasyOCR-1.7.2/easyocr
            os.environ['EASYOCR_MODULE_PATH'] = '/workspace/EasyOCR-1.7.2/easyocr'
            self.ocr_reader = easyocr.Reader(['en'], gpu=True)
            self.ocr_ready = True
            print("[AsyncElevatorDetector] EasyOCR GPU初始化成功")
            return True
        except Exception as e:
            print(f"[AsyncElevatorDetector] EasyOCR初始化失败: {e}")
            return False

    def init_yolo_world(self):
        """初始化YOLO-World"""
        if self.yolo_world_ready:
            return True
        try:
            from ultralytics import YOLO
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolov8s-worldv2.pt')
            if not os.path.exists(model_path):
                model_path = 'yolov8s-worldv2.pt'
            self._yolo_world = YOLO(model_path)
            # 目标类别（用于结果过滤）
            self._target_classes = {'chair', 'stool', 'bottle', 'box', 'bag', 'backpack', 'cup', 'umbrella', 'knife', 'scissors'}
            self.yolo_world_ready = True
            print(f"[AsyncElevatorDetector] YOLO-World GPU初始化成功")
            return True
        except Exception as e:
            print(f"[AsyncElevatorDetector] YOLO-World初始化失败: {e}")
            return False

    def enqueue_ocr(self, frame_bgr, camera_name, person_count, camera_key=""):
        """将帧放入OCR检测队列"""
        try:
            self.ocr_queue.put_nowait((frame_bgr, camera_name, person_count, camera_key))
            self.update_queue_metrics()
            return True
        except queue.Full:
            self.ocr_dropped_count += 1
            self._record_queue_drop('elevator_ocr', self.ocr_dropped_count, camera_name)
            self.update_queue_metrics()
            return False

    def enqueue_obj(self, frame_bgr, camera_name, camera_key=""):
        """将帧放入YOLO-World检测队列"""
        try:
            self.obj_queue.put_nowait((frame_bgr, camera_name, camera_key))
            self.update_queue_metrics()
            return True
        except queue.Full:
            self.obj_dropped_count += 1
            self._record_queue_drop('elevator_obj', self.obj_dropped_count, camera_name)
            self.update_queue_metrics()
            return False

    def _ocr_worker(self):
        """OCR检测工作线程"""
        if not self.init_ocr():
            print("[AsyncElevatorDetector] OCR初始化失败，跳过OCR检测")
            return

        while self.running:
            try:
                item = self.ocr_queue.get(timeout=1.0)
                if len(item) == 4:
                    frame_bgr, camera_name, person_count, camera_key = item
                else:
                    frame_bgr, camera_name, person_count = item
                    camera_key = ""
                self.update_queue_metrics()
                self.elevator_status.update_person_count(camera_name, person_count)

                display_area = self.elevator_status.camera_status.get(camera_name, {}).get('display_area')
                if display_area:
                    x1, y1, x2, y2 = display_area['x1'], display_area['y1'], display_area['x2'], display_area['y2']
                    roi = frame_bgr[y1:y2, x1:x2]
                    if roi.size > 0:
                        results = self.ocr_reader.readtext(roi)
                        text = ' '.join([r[1] for r in results if r[2] > 0.3])
                        floor_number = self._extract_floor_number(text)
                        conf = max([r[2] for r in results]) if results else 0

                        violation_type = self.elevator_status.update_floor_detection(
                            camera_name, floor_number, conf, time.time())

                        # 告警时直接保存图片（不经过队列，避免竞争）
                        if violation_type and person_count > 0:
                            status = self.elevator_status.camera_status.get(camera_name, {})
                            fail_count = status.get('fail_count', 0)
                            success_count = status.get('success_count', 0)
                            if violation_type == 'power_failure':
                                print(f"[停电告警] {camera_name}: 连续{fail_count}次未检测到楼层号，触发停电告警！")
                            elif violation_type == 'floor_stuck':
                                print(f"[故障告警] {camera_name}: 楼层号连续{success_count}次无变化，触发故障告警！")
                            # 直接保存图片
                            if self.image_saver:
                                self.image_saver.enqueue(frame_bgr, camera_name, violation_type, [])
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[OCR Worker Error] {e}")
            finally:
                if 'frame_bgr' in locals():
                    self.ocr_queue.task_done()
                    self.update_queue_metrics()
                    del frame_bgr

    def _obj_worker(self):
        """YOLO-World检测工作线程"""
        if not self.init_yolo_world():
            return

        while self.running:
            try:
                item = self.obj_queue.get(timeout=1.0)
                if len(item) == 3:
                    frame_bgr, camera_name, camera_key = item
                else:
                    frame_bgr, camera_name = item
                    camera_key = ""
                self.update_queue_metrics()

                zone = self._get_elevator_detection_zone(camera_name, camera_key=camera_key)
                if zone:
                    h, w = frame_bgr.shape[:2]
                    x1, y1, x2, y2 = zone
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    if x2 > x1 and y2 > y1:
                        roi = frame_bgr[y1:y2, x1:x2]
                        if roi.size > 0:
                            results = self._yolo_world.predict(roi, conf=0.6, imgsz=640, device=0, verbose=False)
                            if results and results[0].boxes:
                                detections = []
                                for box in results[0].boxes.data:
                                    dx1, dy1, dx2, dy2, conf, cls = box.cpu().numpy()
                                    if conf >= 0.4:
                                        ox1, oy1, ox2, oy2 = int(dx1), int(dy1), int(dx2), int(dy2)
                                        ox1, oy1 = ox1 + x1, oy1 + y1
                                        ox2, oy2 = ox2 + x1, oy2 + y1
                                        cls_idx = int(cls)
                                        cls_name = results[0].names[cls_idx]
                                        # 只保留目标类别
                                        if cls_name.lower() in self._target_classes:
                                            detections.append([ox1, oy1, ox2, oy2, cls_name, conf])

                                if detections:
                                    obj_names = ', '.join(set([d[4] for d in detections]))
                                    print(f"[异物告警] {camera_name}: 检测到{len(detections)}个异物 ({obj_names})")
                                    # 直接保存图片（不经过result_queue，避免多消费者竞争）
                                    if self.image_saver:
                                        # 画框
                                        for ox1, oy1, ox2, oy2, cls_name, conf in detections:
                                            cv2.rectangle(frame_bgr, (ox1, oy1), (ox2, oy2), (0, 0, 255), 2)
                                            cv2.putText(frame_bgr, cls_name, (ox1, oy1 - 5),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                        self.image_saver.enqueue(frame_bgr, camera_name, "object", [])
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[YOLO-World Worker Error] {e}")
            finally:
                if 'frame_bgr' in locals():
                    self.obj_queue.task_done()
                    self.update_queue_metrics()
                    del frame_bgr

    def _result_worker(self):
        """结果处理工作线程 - 只处理OCR统计，YOLO-World结果由probe直接消费"""
        while self.running:
            try:
                result = self.result_queue.get(timeout=1.0)
                # OCR和YOLO-World都在worker中直接保存，这里只做统计（可以扩展）
            except queue.Empty:
                continue
            except Exception as e:
                pass

    def _extract_floor_number(self, text):
        """从OCR结果中提取楼层号"""
        if not text:
            return None
        text = text.upper()
        text = text.replace('I', '1').replace('L', '1').replace('|', '1')
        text = text.replace('O', '0').replace('Q', '0')
        text = text.replace('B', '8').replace('E', '8')
        text = text.replace('S', '5').replace('Z', '2')
        text = text.replace(' ', '')
        import re
        match = re.search(r'(\d+)F?', text)
        if match:
            return match.group(1)
        if text.isdigit() and len(text) <= 3:
            return text
        return None

    def _get_elevator_detection_zone(self, camera_name, camera_key=""):
        """获取电梯检测区域 - 支持通过name或camera_key查找"""
        # 首先尝试直接匹配 cameraIndexCode/camera_key，再回退到 camera_name
        lookup_key = str(camera_key or camera_name or "").lower()
        cam_config = self.camera_configs.get(lookup_key, {})
        if not cam_config and camera_key:
            for cam_cfg in self.camera_configs.values():
                if str(cam_cfg.get('cameraIndexCode', '')).lower() == lookup_key:
                    cam_config = cam_cfg
                    break
        if not cam_config:
            cam_config = self.camera_configs.get(str(camera_name or "").lower(), {})
        zone = cam_config.get('elevator_detection_zone')
        if zone:
            # 处理points格式
            if 'points' in zone:
                pts = zone['points']
                if len(pts) >= 3:
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    return (min(xs), min(ys), max(xs), max(ys))
            elif 'x1' in zone:
                return (zone['x1'], zone['y1'], zone['x2'], zone['y2'])
            return zone

        # 然后遍历查找name匹配
        for cam_key, cam_cfg in self.camera_configs.items():
            config_name = cam_cfg.get('name', '').lower()
            if config_name == camera_name.lower():
                zone = cam_cfg.get('elevator_detection_zone')
                if zone:
                    if 'points' in zone:
                        pts = zone['points']
                        if len(pts) >= 3:
                            xs = [p[0] for p in pts]
                            ys = [p[1] for p in pts]
                            return (min(xs), min(ys), max(xs), max(ys))
                    elif 'x1' in zone:
                        return (zone['x1'], zone['y1'], zone['x2'], zone['y2'])
                    return zone
            display_name = cam_cfg.get('display_name', '').lower()
            if display_name == camera_name.lower():
                zone = cam_cfg.get('elevator_detection_zone')
                if zone:
                    if 'points' in zone:
                        pts = zone['points']
                        if len(pts) >= 3:
                            xs = [p[0] for p in pts]
                            ys = [p[1] for p in pts]
                            return (min(xs), min(ys), max(xs), max(ys))
                    elif 'x1' in zone:
                        return (zone['x1'], zone['y1'], zone['x2'], zone['y2'])
                    return zone
        return None

    def get_result(self, timeout=0.1):
        """获取检测结果（非阻塞）"""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        """停止检测器"""
        self.running = False


class ImageSaver:
    """异步存图类：防止写磁盘阻塞 GStreamer 线程"""
    def __init__(self, vlm_manager=None, uploader=None):
        self.queue = queue.Queue(maxsize=IMAGE_SAVER_QUEUE_MAXSIZE)
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        # VLM和上传器
        self.vlm_manager = vlm_manager
        self.uploader = uploader
        self.dataset_dir = DATASET_DIR

    def set_upload_config(self, vlm_manager, uploader):
        self.vlm_manager = vlm_manager
        self.uploader = uploader

    def update_queue_metrics(self):
        """更新队列大小指标到Prometheus"""
        if METRICS_ENABLED:
            pm.QUEUE_SIZE.labels(queue_name='alarm').set(self.queue.qsize())

    def enqueue(self, frame_bgr, camera_name, alarm_type, bbox_list=None, camera_key=""):
        """bbox_list: [(label, confidence, (x1,y1,x2,y2)), ...]"""
        try:
            self.queue.put_nowait((frame_bgr, camera_name, alarm_type, bbox_list, camera_key))
            self.update_queue_metrics()
        except queue.Full:
            if METRICS_ENABLED:
                pm.QUEUE_DROPPED.labels(queue_name='alarm').inc()
            print(f"[警告] ImageSaver队列已满，丢弃 {camera_name} 的 {alarm_type} 图片！")
            self.update_queue_metrics()

    def _worker(self):
        while self.running:
            try:
                item = self.queue.get(timeout=1)
                if len(item) == 5:
                    frame, cam_name, a_type, bbox_list, camera_key = item
                else:
                    frame, cam_name, a_type, bbox_list = item
                    camera_key = ""
                self.update_queue_metrics()
            except queue.Empty:
                continue
            try:
                # 保存原图到dataset目录
                ts_orig = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                orig_filename = f"{cam_name}_{a_type}_{ts_orig}.jpg"
                os.makedirs(self.dataset_dir, exist_ok=True)
                orig_path = os.path.join(self.dataset_dir, orig_filename)
                cv2.imwrite(orig_path, frame)

                # 画 bbox
                if bbox_list:
                    for label, conf, (x1, y1, x2, y2) in bbox_list:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        text = f"{label}"
                        cv2.putText(frame, text, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                filename = f"{cam_name}_{a_type}_{ts}.jpg"

                if a_type == "dangerous_item":
                    save_dir = DANGEROUS_ITEM_DIR
                    detection_type = "dangerous_item"
                elif a_type == "fire":
                    save_dir = FIRE_DIR
                    detection_type = "fire"
                elif a_type == "smoke":
                    save_dir = SMOKE_DIR
                    detection_type = "smoke"
                elif a_type == "cigarette":
                    save_dir = CIGARETTE_DIR
                    detection_type = "cigarette"
                elif a_type == "motor_vio":
                    save_dir = MOTOR_DIR
                    detection_type = "motor_vehicle_violations"
                elif a_type == "non_motor_vio":
                    save_dir = NON_MOTOR_DIR
                    detection_type = "non_motor_violations"
                elif a_type == "power_failure":
                    save_dir = ELEVATOR_POWER_DIR
                    detection_type = "power_failure"
                elif a_type == "floor_stuck":
                    save_dir = ELEVATOR_FLOOR_DIR
                    detection_type = "floor_stuck"
                elif a_type == "object":
                    save_dir = OBJECT_DIR
                    detection_type = "object"
                elif a_type == "fall":
                    save_dir = FALL_DIR
                    detection_type = "fall"
                elif a_type == "fight":
                    save_dir = FIGHT_DIR
                    detection_type = "fight"
                elif a_type == "uncivilized_pet":
                    save_dir = PET_DIR
                    detection_type = "uncivilized_pet"
                else:
                    save_dir = ALARM_IMG_DIR
                    detection_type = a_type

                image_path = os.path.join(save_dir, filename)
                cv2.imwrite(image_path, frame)
                print(f"[图片保存] {save_dir}/{filename}")

                requires_vlm = detection_type in VLM_REQUIRED_TYPES
                need_vlm = False
                if self.vlm_manager and requires_vlm:
                    need_vlm = self.vlm_manager.submit_task(image_path, detection_type)
                    if need_vlm:
                        print(f"[VLM提交] {filename} -> {detection_type}")
                    else:
                        print(f"[VLM跳过] {filename} -> {detection_type} 未成功进入VLM队列，跳过上传和通知")

                if self.uploader:
                    if requires_vlm and not need_vlm:
                        print(f"[上传跳过] {filename} 需要VLM但任务未提交成功")
                    else:
                        self.uploader.upload_detection(
                                image_path,
                                cam_name,
                                detection_type,
                                camera_key=camera_key,
                                wait_vlm=need_vlm,
                                local_vlm_path=image_path if need_vlm else ""
                            )
                        print(f"[上传提交] {filename}")
            except Exception as e:
                print(f"[ImageSaver ERROR] {cam_name if 'cam_name' in locals() else 'unknown'}: {e}")
            finally:
                self.queue.task_done()
                self.update_queue_metrics()

    def stop(self):
        self.running = False
        self.thread.join()


class DetectionHandler:
    """检测结果处理器 - 包含违停、摔倒、打架检测"""

    def __init__(self, config=None, camera_configs=None):
        self.config = config or {}
        self.camera_configs = camera_configs or {}
        self.lock = threading.Lock()
        self.last_seen_lock = threading.Lock()  # 看门狗心跳读写锁
        self.stats = {'motor_vio': 0, 'non_motor_vio': 0, 'fall': 0, 'fire': 0,
                     'smoke': 0, 'cigarette': 0, 'fight': 0, 'pet': 0}
        self.fire_smoke_detector = FireSmokeDetector()
        self._last_fire_smoke_cleanup = 0.0
        self.person_bound_alerts = {}
        self.pet_alerts = {}

        # 违停跟踪器: {tracker_key: {'first_seen': time, 'last_seen': time, 'bbox': bbox, 'violation_reported': bool, 'class_id': int}}
        self.vehicle_trackers = {}

        # 摔倒检测跟踪器: {camera_name: {'persons': {person_id: {'bbox': bbox, 'width': w, 'height': h, 'aspect_ratio': w/h, 'history': [], 'fall_state': 'normal', 'start_time': time}}, 'frame_count': int}}
        self.person_trackers = {}
        self.next_person_id = 1

        # 打架检测跟踪器: {camera_name: {'persons': [], 'last_alert_time': time}}
        self.fight_trackers = {}

        # 电梯状态管理器
        self.elevator_status = ElevatorStatusManager()

        # OCR读者
        self.ocr_reader = None
        self.ocr_ready = False

        # 电梯摄像头最后OCR时间
        self.last_ocr_time = {}
        self.last_obj_check_time = {}

        # 初始化电梯摄像头 - 使用display_name作为key（与探针中camera_name一致）
        if camera_configs:
            for cam_key, cam_config in camera_configs.items():
                display_area = cam_config.get('display_area')
                # 优先使用display_name，其次使用name，最后使用cam_key
                cam_name = cam_config.get('name', cam_key)
                if display_area and display_area.get('enabled', False):
                    self.elevator_status.init_camera(cam_name, display_area)
                    print(f"[电梯监控] 初始化: {cam_name} (key={cam_key})")

        # 创建目录
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(DETECTION_DIR, exist_ok=True)

        print("[INFO] DetectionHandler initialized (with violation, fall, fight detection)")

    def process_detection(self, camera_id, camera_name, class_id, confidence, bbox=None, frame_width=640, frame_height=480, tracker_id=None, camera_key=None):
        """处理检测结果"""
        label = LABEL_NAMES.get(class_id, f"unknown_{class_id}")
        state_key = camera_key or camera_name
        fire_smoke_triggered = False
        motor_vio_triggered = False
        non_motor_vio_triggered = False
        pet_triggered = False

        # Prometheus指标 - 检测计数
        if METRICS_ENABLED:
            pm.DETECTIONS_TOTAL.labels(camera=camera_name, label=label).inc()

        # 违停检测 - car(2) 和 motorcycle(3) 和 bicycle(1)，使用tracker_id追踪
        if class_id in [1, 2, 3]:
            vehicle_violation_type = self._check_vehicle_violation(camera_name, class_id, bbox, confidence, tracker_id, state_key=state_key)
            if vehicle_violation_type == "motor_vio":
                motor_vio_triggered = True
            elif vehicle_violation_type == "non_motor_vio":
                non_motor_vio_triggered = True

        # 火焰/烟雾检测 - 持续检测达到阈值后才告警
        if class_id in [4, 5]:
            fire_smoke_triggered = self._check_fire_smoke(camera_name, label, confidence, state_key=state_key)

        # 宠物检测 - 使用 tracker_id 去重，同一只宠物短时间内只告警一次
        if class_id == 6:
            pet_triggered = self._check_pet(camera_name, tracker_id, state_key=state_key)

        # 摔倒检测 - person(0)，使用tracker_id追踪
        fall_triggered = False
        if class_id == 0:
            result = self._check_fall(camera_name, bbox, confidence, tracker_id, state_key=state_key)
            if result:
                fall_triggered = True

        # 打架检测 - person(0)，使用tracker_id追踪
        fight_triggered = False
        if class_id == 0:
            result = self._check_fight(camera_name, bbox, confidence, tracker_id, state_key=state_key)
            if result:
                fight_triggered = True

        return {
            "camera_id": camera_id,
            "camera_name": camera_name,
            "camera_key": state_key,
            "label": label,
            "class_id": class_id,
            "confidence": confidence,
            "timestamp": datetime.now().strftime('%Y%m%d_%H%M%S'),
            "bbox": bbox,
            "motor_vio_triggered": motor_vio_triggered,
            "non_motor_vio_triggered": non_motor_vio_triggered,
            "fire_smoke_triggered": fire_smoke_triggered,
            "fall_triggered": fall_triggered,
            "fight_triggered": fight_triggered,
            "pet_triggered": pet_triggered
        }

    def init_ocr(self):
        """初始化EasyOCR reader"""
        if self.ocr_ready:
            return True
        try:
            import easyocr
            os.environ['EASYOCR_MODULE_PATH'] = OCR_MODEL_PATH
            self.ocr_reader = easyocr.Reader(['en'], gpu=True)
            self.ocr_ready = True
            print("[OCR] EasyOCR初始化成功")
            return True
        except Exception as e:
            print(f"[OCR] EasyOCR初始化失败: {e}")
            return False

    def _extract_floor_number(self, text):
        """从OCR结果中提取楼层号"""
        if not text:
            return None
        text = text.upper()
        text = text.replace('I', '1').replace('L', '1').replace('|', '1')
        text = text.replace('O', '0').replace('Q', '0')
        text = text.replace('B', '8').replace('E', '8')
        text = text.replace('S', '5').replace('Z', '2')
        text = text.replace(' ', '')
        import re
        match = re.search(r'(\d+)F?', text)
        if match:
            return match.group(1)
        if text.isdigit() and len(text) <= 3:
            return text
        return None

    def perform_ocr_detection(self, frame, camera_name):
        """执行OCR楼层号检测"""
        if camera_name not in self.elevator_status.camera_status:
            return None, 0.0

        if not self.ocr_ready:
            self.init_ocr()

        if self.ocr_reader is None:
            return None, 0.0

        try:
            display_area = self.elevator_status.camera_status[camera_name]['display_area']
            x1, y1, x2, y2 = display_area['x1'], display_area['y1'], display_area['x2'], display_area['y2']
            rotate_angle = display_area.get('angle', 0)

            display_region = frame[y1:y2, x1:x2]
            if display_region.size == 0:
                return None, 0.0

            h, w = display_region.shape[:2]
            if h <= 0 or w <= 0:
                return None, 0.0

            # 放大8倍
            enlarged = cv2.resize(display_region, (w*8, h*8), interpolation=cv2.INTER_CUBIC)

            results = self.ocr_reader.readtext(enlarged)
            if not results:
                return None, 0.0

            best_result = None
            best_confidence = 0.0

            for (bbox, text, confidence) in results:
                floor_number = self._extract_floor_number(text)
                if floor_number is not None and confidence > best_confidence:
                    best_result = floor_number
                    best_confidence = confidence

            return best_result, best_confidence

        except Exception as e:
            return None, 0.0

    def update_person_count(self, camera_name, count):
        """更新电梯内人数"""
        self.elevator_status.update_person_count(camera_name, count)

    def check_elevator_violation(self, camera_name, floor_number, confidence):
        """检查电梯故障/停电"""
        return self.elevator_status.update_floor_detection(camera_name, floor_number, confidence, time.time())

    def _get_elevator_detection_zone(self, camera_name):
        """获取电梯检测区域"""
        cam_config = self.camera_configs.get(camera_name, {})
        zone = cam_config.get('elevator_detection_zone')
        if not zone:
            return None
        if 'points' in zone:
            pts = zone['points']
            if len(pts) >= 3:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                return (min(xs), min(ys), max(xs), max(ys))
        elif 'x1' in zone:
            return (zone['x1'], zone['y1'], zone['x2'], zone['y2'])
        return None

    def _normalize_zone(self, zone):
        if not isinstance(zone, dict) or not zone.get('enabled', True):
            return None
        if 'points' in zone:
            pts = []
            for point in zone.get('points') or []:
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    continue
                pts.append((float(point[0]), float(point[1])))
            if len(pts) >= 3:
                return {'type': 'polygon', 'points': pts}
        if all(k in zone for k in ('x1', 'y1', 'x2', 'y2')):
            return {
                'type': 'rect',
                'x1': float(zone['x1']),
                'y1': float(zone['y1']),
                'x2': float(zone['x2']),
                'y2': float(zone['y2']),
            }
        return None

    def _get_camera_zone(self, camera_name, state_key, zone_field):
        lookup_keys = []
        if state_key:
            lookup_keys.append(str(state_key).lower())
        if camera_name:
            lookup_keys.append(str(camera_name).lower())

        for key in lookup_keys:
            cam_config = self.camera_configs.get(key, {})
            zone = self._normalize_zone(cam_config.get(zone_field))
            if zone is not None:
                return zone

        camera_name_lower = str(camera_name or '').lower()
        state_key_lower = str(state_key or '').lower()
        for cam_cfg in self.camera_configs.values():
            if state_key_lower and str(cam_cfg.get('cameraIndexCode', '')).lower() == state_key_lower:
                zone = self._normalize_zone(cam_cfg.get(zone_field))
                if zone is not None:
                    return zone
            if camera_name_lower and str(cam_cfg.get('name', '')).lower() == camera_name_lower:
                zone = self._normalize_zone(cam_cfg.get(zone_field))
                if zone is not None:
                    return zone
        return None

    def _point_in_polygon(self, x, y, points):
        if not points or len(points) < 3:
            return False

        inside = False
        j = len(points) - 1
        eps = 1e-6
        for i, (xi, yi) in enumerate(points):
            xj, yj = points[j]

            # 边界也视为在区域内，避免贴边目标抖动。
            cross = (x - xj) * (yi - yj) - (y - yj) * (xi - xj)
            if abs(cross) <= eps:
                if min(xj, xi) - eps <= x <= max(xj, xi) + eps and min(yj, yi) - eps <= y <= max(yj, yi) + eps:
                    return True

            if (yi > y) != (yj > y):
                x_intersect = (xj - xi) * (y - yi) / (yj - yi) + xi
                if x < x_intersect:
                    inside = not inside
            j = i
        return inside

    def _is_bbox_center_in_zone(self, bbox, zone):
        if zone is None:
            return False
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        if isinstance(zone, dict):
            if zone.get('type') == 'polygon':
                return self._point_in_polygon(cx, cy, zone.get('points') or [])
            if zone.get('type') == 'rect':
                x1, y1, x2, y2 = zone['x1'], zone['y1'], zone['x2'], zone['y2']
                return min(x1, x2) <= cx <= max(x1, x2) and min(y1, y2) <= cy <= max(y1, y2)
            return False

        x1, y1, x2, y2 = zone
        return x1 <= cx <= x2 and y1 <= cy <= y2

    def _is_elevator_camera(self, camera_name):
        """检查是否是电梯摄像头"""
        return self._get_elevator_detection_zone(camera_name) is not None

    def _is_bbox_in_zone(self, bbox, zone):
        """检查检测框是否在区域内"""
        if zone is None:
            return True
        x1, y1, x2, y2 = zone
        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        return x1 <= cx <= x2 and y1 <= cy <= y2

    def init_yolo_world(self):
        """初始化YOLO-World模型用于异物检测"""
        if hasattr(self, '_yolo_world_ready') and self._yolo_world_ready:
            return True
        try:
            from ultralytics import YOLO
            import os
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolov8m-worldv2.pt')
            if not os.path.exists(model_path):
                model_path = 'yolov8m-worldv2.pt'
            self._yolo_world = YOLO(model_path)
            self._yolo_world_classes = [
                "garbage", "trash", "rubbish", "chair", "stool", "seat",
                "bottle", "plastic bottle", "box", "carton", "package",
                "bag", "backpack", "handbag", "luggage", "suitcase",
                "umbrella", "stick", "rod", "tool", "cup", "coffee cup",
                "takeout box", "food container", "cigarette", "cigarette butt",
                "can", "tin can", "newspaper", "magazine"
            ]
            self._yolo_world.set_classes(self._yolo_world_classes)
            self._yolo_world_ready = True
            print(f"[YOLO-World] 模型初始化完成，检测类别: {len(self._yolo_world_classes)}种")
            return True
        except Exception as e:
            print(f"[YOLO-World] 初始化失败: {e}")
            return False

    def detect_object_with_yolo(self, roi, camera_name):
        """使用YOLO-World检测电梯内异物"""
        if not hasattr(self, '_yolo_world_ready') or not self._yolo_world_ready:
            if not self.init_yolo_world():
                return None

        try:
            results = self._yolo_world.predict(roi, conf=0.4, imgsz=640, device=0, verbose=False)
            if not results or not results[0].boxes:
                return None

            detections = []
            for box in results[0].boxes.data:
                dx1, dy1, dx2, dy2, conf, cls = box.cpu().numpy()
                if conf < 0.4:
                    continue
                ox1, oy1, ox2, oy2 = int(dx1), int(dy1), int(dx2), int(dy2)
                cls_idx = int(cls)
                cls_name = results[0].names[cls_idx]
                detections.append([ox1, oy1, ox2, oy2, cls_name, conf])
            return detections
        except Exception as e:
            print(f"[YOLO-World] 检测失败: {e}")
            return None

    def check_object_in_elevator(self, camera_name, frame, person_count):
        """检查电梯内异物 - 使用YOLO-World检测"""
        if person_count > 0:
            return None

        zone = self._get_elevator_detection_zone(camera_name)
        if zone is None:
            return None

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = zone
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        detections = self.detect_object_with_yolo(roi, camera_name)
        if not detections:
            return None

        current_time = time.time()
        key = f"{camera_name}_object"
        last_save = getattr(self, '_object_last_save_time', {}).get(key, 0)
        if current_time - last_save < 120:
            return None

        if not hasattr(self, '_object_last_save_time'):
            self._object_last_save_time = {}
        self._object_last_save_time[key] = current_time

        # 转换检测框坐标到原图
        result_detections = []
        for dx1, dy1, dx2, dy2, cls_name, conf in detections:
            ox1, oy1, ox2, oy2 = dx1 + x1, dy1 + y1, dx2 + x1, dy2 + y1
            result_detections.append((ox1, oy1, ox2, oy2, cls_name, conf))

        print(f"[🗑️ 异物检测] {camera_name}: 检测到 {len(result_detections)} 个异物")
        self.stats['object'] = self.stats.get('object', 0) + 1
        return result_detections

    def _check_vehicle_violation(self, camera_name, class_id, bbox, confidence, tracker_id=None, state_key=None):
        """检查车辆违停 - 同一 tracker_id 持续 30 秒视为违停，同一 ID 只告警一次"""
        if bbox is None:
            return None

        # 使用tracker_id作为唯一标识，如果没有则用class_id
        camera_state_key = state_key or camera_name
        vehicle_zone = self._get_camera_zone(camera_name, camera_state_key, 'vehicle_violation_zone')
        # 未配置违停区域的摄像头不参与违停检测，避免全画面误报。
        if vehicle_zone is None:
            return None

        if tracker_id is not None:
            key_prefix = f"{camera_state_key}_{tracker_id}"
        else:
            key_prefix = f"{camera_state_key}_{class_id}"

        current_time = time.time()

        with self.lock:
            if not self._is_bbox_center_in_zone(bbox, vehicle_zone):
                if key_prefix in self.vehicle_trackers:
                    del self.vehicle_trackers[key_prefix]
                return None

            if key_prefix not in self.vehicle_trackers:
                # 新检测到车辆
                self.vehicle_trackers[key_prefix] = {
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'bbox': bbox,
                    'violation_reported': False,
                    'class_id': class_id,
                }
                print(f"[违停跟踪] {camera_name}: {LABEL_NAMES.get(class_id)} tid={tracker_id} 首次检测")
            else:
                # 更新最后检测时间
                self.vehicle_trackers[key_prefix]['last_seen'] = current_time
                self.vehicle_trackers[key_prefix]['bbox'] = bbox
                self.vehicle_trackers[key_prefix]['class_id'] = class_id

                # 检查是否超过违停阈值
                duration = current_time - self.vehicle_trackers[key_prefix]['first_seen']
                if duration >= VEHICLE_VIOLATION_DURATION_THRESHOLD and not self.vehicle_trackers[key_prefix]['violation_reported']:
                    # 违停确认，同一ID只告警一次
                    self.vehicle_trackers[key_prefix]['violation_reported'] = True
                    print(f"[违停检测] {camera_name}: {LABEL_NAMES.get(class_id)} tid={tracker_id} 违停已持续 {duration:.0f}秒!")
                    if class_id == 2:
                        self.stats['motor_vio'] = self.stats.get('motor_vio', 0) + 1
                        return "motor_vio"
                    self.stats['non_motor_vio'] = self.stats.get('non_motor_vio', 0) + 1
                    return "non_motor_vio"

            # 清理超时的跟踪器（超过60秒没检测到）
            timeout_keys = []
            for k, v in self.vehicle_trackers.items():
                if current_time - v['last_seen'] > 60:
                    timeout_keys.append(k)
            for k in timeout_keys:
                del self.vehicle_trackers[k]
        return None

    def _check_fire_smoke(self, camera_name, label, confidence, state_key=None):
        """火焰/烟雾检测"""
        current_time = time.time()
        if current_time - self._last_fire_smoke_cleanup >= 30:
            self.fire_smoke_detector.cleanup()
            self._last_fire_smoke_cleanup = current_time

        triggered = self.fire_smoke_detector.process(label, state_key or camera_name, confidence, display_cam=camera_name)
        if triggered:
            self.stats[label] = self.stats.get(label, 0) + 1
        return triggered

    def _check_pet(self, camera_name, tracker_id=None, state_key=None):
        """宠物检测 - 同一 camera + tracker_id 只告警一次，超时后清理状态。"""
        if tracker_id is None or tracker_id == 18446744073709551615:
            return False

        current_time = time.time()
        camera_state_key = state_key or camera_name
        key = f"{camera_state_key}_{tracker_id}_pet"

        with self.lock:
            expired_keys = [
                alert_key for alert_key, state in self.pet_alerts.items()
                if current_time - state.get('last_seen', 0.0) > PERSON_ALERT_STATE_TIMEOUT
            ]
            for alert_key in expired_keys:
                del self.pet_alerts[alert_key]

            state = self.pet_alerts.get(key)
            if state is None:
                self.pet_alerts[key] = {
                    'last_seen': current_time,
                    'tracker_id': tracker_id,
                }
                self.stats['pet'] = self.stats.get('pet', 0) + 1
                print(f"[宠物] {camera_name}: tid={tracker_id} 首次告警")
                return True

            state['last_seen'] = current_time

        return False

    def _check_person_bound_alert(self, camera_name, alert_type, person_tracker_id, label=None, state_key=None):
        """同一 camera + person tracker + alert_type 只告警一次，超时后清理状态。"""
        if person_tracker_id is None or person_tracker_id == 18446744073709551615:
            return False

        current_time = time.time()
        camera_state_key = state_key or camera_name
        key = f"{camera_state_key}_{person_tracker_id}_{alert_type}"

        with self.lock:
            expired_keys = [
                state_key for state_key, state in self.person_bound_alerts.items()
                if current_time - state.get('last_seen', 0.0) > PERSON_ALERT_STATE_TIMEOUT
            ]
            for state_key in expired_keys:
                del self.person_bound_alerts[state_key]

            state = self.person_bound_alerts.get(key)
            if state is None:
                self.person_bound_alerts[key] = {
                    'last_seen': current_time,
                    'person_tracker_id': person_tracker_id,
                    'alert_type': alert_type,
                }
                if alert_type == 'cigarette':
                    self.stats['cigarette'] = self.stats.get('cigarette', 0) + 1
                    print(f"[吸烟] {camera_name}: person_tid={person_tracker_id} 首次告警")
                elif alert_type == 'dangerous_item':
                    self.stats['dangerous_item'] = self.stats.get('dangerous_item', 0) + 1
                    label_desc = label or 'dangerous_item'
                    print(f"[危险品] {camera_name}: person_tid={person_tracker_id}, label={label_desc} 首次告警")
                return True

            state['last_seen'] = current_time

        return False

    def _check_fall(self, camera_name, bbox, confidence, tracker_id=None, state_key=None):
        """摔倒检测 - 优化版：宽高比突变 + 面积检查 + 持续时间3秒"""
        if bbox is None:
            return

        left, top, right, bottom = bbox
        w = right - left
        h = bottom - top

        if w <= 0 or h <= 0:
            return

        aspect_ratio = w / h  # 宽高比
        area = w * h  # 面积
        current_time = time.time()

        # 使用tracker_id作为唯一标识
        camera_state_key = state_key or camera_name
        if tracker_id is not None:
            person_key = f"{camera_state_key}_{tracker_id}"
        else:
            return

        with self.lock:
            if camera_state_key not in self.person_trackers:
                self.person_trackers[camera_state_key] = {
                    'persons': {},
                    'frame_count': 0
                }

            if person_key not in self.person_trackers[camera_state_key]['persons']:
                # 新人
                self.person_trackers[camera_state_key]['persons'][person_key] = {
                    'bbox': bbox,
                    'width': w,
                    'height': h,
                    'aspect_ratio': aspect_ratio,
                    'area': area,
                    'history': [(aspect_ratio, current_time)],
                    'fall_state': 'normal',
                    'start_time': current_time,
                    'last_time': current_time,
                    'fall_reported': False,
                    'was_normal': True  # 标记之前是否处于正常站立状态
                }
            else:
                # 更新已有的人
                p = self.person_trackers[camera_state_key]['persons'][person_key]
                old_ratio = p['aspect_ratio']
                old_area = p.get('area', area)
                p['bbox'] = bbox
                p['width'] = w
                p['height'] = h
                p['aspect_ratio'] = aspect_ratio
                p['area'] = area
                p['last_time'] = current_time
                p['history'].append((aspect_ratio, current_time))

                # 保持历史在30条以内
                if len(p['history']) > 30:
                    p['history'] = p['history'][-30:]

                # === 摔倒检测逻辑 ===
                # 条件1: 当前宽高比 > 1.2（倒地状）
                # 条件2: 之前宽高比 < 0.9（站立状），且在最近10帧内保持正常
                # 条件3: 面积不能太小（排除误检的小目标），不能比原来大太多
                # 条件4: 持续时间 > 3秒（原来1秒太短）

                if p['fall_state'] == 'normal':
                    # 检查是否之前一直处于正常状态（宽高比 < 0.9）
                    recent_normal = all(r < 0.9 for r, _ in p['history'][-10:])
                    p['was_normal'] = recent_normal

                    # 触发疑似摔倒：宽高比突变 + 之前正常 + 面积合理
                    if (aspect_ratio > 1.2 and
                        old_ratio < 0.9 and
                        p['was_normal'] and
                        area > 1000 and  # 面积不能太小
                        area < old_area * 3):  # 面积不能突然增大3倍
                        p['fall_state'] = 'warning'
                        p['fall_start_time'] = current_time
                        print(f"[摔倒警告] {camera_name}: tid={tracker_id} 疑似摔倒 (ratio={aspect_ratio:.2f}, area={area})")

                elif p['fall_state'] == 'warning':
                    duration = current_time - p.get('fall_start_time', current_time)
                    # 确认摔倒：持续 > 3秒（原来1秒太短）+ 宽高比保持 > 1.0 + 面积合理
                    if (aspect_ratio > 1.0 and
                        duration > 3.0 and
                        area > 1000 and
                        not p.get('fall_reported', False)):
                        p['fall_reported'] = True
                        p['fall_state'] = 'fallen'
                        self.stats['fall'] = self.stats.get('fall', 0) + 1
                        print(f"[摔倒检测] {camera_name}: tid={tracker_id} 确认摔倒！持续{duration:.1f}秒, area={area}")
                        return True
                    elif aspect_ratio < 0.9:
                        # 恢复正常
                        p['fall_state'] = 'normal'

            # 清理超时的人（30秒没更新）
            self.person_trackers[camera_state_key]['frame_count'] += 1
            timeout_pids = []
            for pid, pdata in self.person_trackers[camera_state_key]['persons'].items():
                if current_time - pdata.get('last_time', 0) > 30:
                    timeout_pids.append(pid)
            for pid in timeout_pids:
                del self.person_trackers[camera_state_key]['persons'][pid]

    def _check_fight(self, camera_name, bbox, confidence, tracker_id=None, state_key=None):
        """
        打架检测 - precision-first pair heuristic

        The goal here is to reduce false positives in the first-stage filter.
        A pair is only promoted when all of the following are observed:
            1. Both persons are large enough to trust the geometry.
            2. The pair stays in normalized close contact.
            3. Both persons show sustained high motion at the same time.
            4. The pair distance oscillates, which helps separate fighting
               from walking side by side or short contact.
        """
        if bbox is None or tracker_id is None:
            return

        def _bbox_iou(box1, box2):
            ix1 = max(box1[0], box2[0])
            iy1 = max(box1[1], box2[1])
            ix2 = min(box1[2], box2[2])
            iy2 = min(box1[3], box2[3])
            if ix1 >= ix2 or iy1 >= iy2:
                return 0.0
            inter = (ix2 - ix1) * (iy2 - iy1)
            area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
            area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
            denom = area1 + area2 - inter
            return inter / denom if denom > 0 else 0.0

        def _bbox_edge_gap(box1, box2):
            gap_x = max(0.0, max(box1[0], box2[0]) - min(box1[2], box2[2]))
            gap_y = max(0.0, max(box1[1], box2[1]) - min(box1[3], box2[3]))
            return (gap_x ** 2 + gap_y ** 2) ** 0.5

        def _count_distance_swings(distance_history):
            if len(distance_history) < 3:
                return 0
            swings = 0
            prev_sign = 0
            for idx in range(1, len(distance_history)):
                delta = distance_history[idx] - distance_history[idx - 1]
                if abs(delta) < FIGHT_SWING_DELTA_RATIO:
                    continue
                sign = 1 if delta > 0 else -1
                if prev_sign != 0 and sign != prev_sign:
                    swings += 1
                prev_sign = sign
            return swings

        current_time = time.time()
        camera_state_key = state_key or camera_name

        with self.lock:
            if camera_state_key not in self.fight_trackers:
                self.fight_trackers[camera_state_key] = {
                    'persons': {},
                    'pairs': {},
                    'last_alert_time': 0
                }

            ft = self.fight_trackers[camera_state_key]
            person_key = f"{tracker_id}"

            # 计算运动幅度
            motion = 0
            if person_key in ft['persons']:
                prev_bbox = ft['persons'][person_key]['bbox']
                prev_cx = (prev_bbox[0] + prev_bbox[2]) / 2
                prev_cy = (prev_bbox[1] + prev_bbox[3]) / 2
                curr_cx = (bbox[0] + bbox[2]) / 2
                curr_cy = (bbox[1] + bbox[3]) / 2
                motion = ((curr_cx - prev_cx) ** 2 + (curr_cy - prev_cy) ** 2) ** 0.5

            # 更新人的跟踪信息
            bbox_height = max(0.0, bbox[3] - bbox[1])
            bbox_width = max(0.0, bbox[2] - bbox[0])
            ft['persons'][person_key] = {
                'bbox': bbox,
                'motion': motion,
                'time': current_time,
                'height': bbox_height,
                'area': bbox_width * bbox_height,
            }

            # 清理超时的跟踪（5秒内没有更新）
            timeout_tids = [tid for tid, pdata in ft['persons'].items()
                          if current_time - pdata.get('time', 0) > 5]
            for tid in timeout_tids:
                del ft['persons'][tid]

            # 清理超时的pairs
            pair_keys_to_delete = []
            for pair_key in ft['pairs']:
                tid1, tid2 = pair_key.split('_')
                if tid1 not in ft['persons'] or tid2 not in ft['persons']:
                    pair_keys_to_delete.append(pair_key)
            for key in pair_keys_to_delete:
                del ft['pairs'][key]

            persons = list(ft['persons'].items())
            if len(persons) < 2:
                return

            # 遍历所有人对
            for i in range(len(persons)):
                for j in range(i + 1, len(persons)):
                    tid1, pdata1 = persons[i]
                    tid2, pdata2 = persons[j]
                    bbox1 = pdata1['bbox']
                    bbox2 = pdata2['bbox']
                    height1 = pdata1.get('height', 0.0)
                    height2 = pdata2.get('height', 0.0)
                    area1 = pdata1.get('area', 0.0)
                    area2 = pdata2.get('area', 0.0)

                    if min(height1, height2) < FIGHT_MIN_BBOX_HEIGHT:
                        continue
                    if min(area1, area2) < FIGHT_MIN_BBOX_AREA:
                        continue

                    # 计算中心距离
                    c1x = (bbox1[0] + bbox1[2]) / 2
                    c1y = (bbox1[1] + bbox1[3]) / 2
                    c2x = (bbox2[0] + bbox2[2]) / 2
                    c2y = (bbox2[1] + bbox2[3]) / 2
                    dist = ((c1x - c2x) ** 2 + (c1y - c2y) ** 2) ** 0.5

                    pair_ids = sorted((str(tid1), str(tid2)))
                    pair_key = f"{pair_ids[0]}_{pair_ids[1]}"
                    avg_height = max(1.0, (height1 + height2) / 2.0)
                    normalized_distance = dist / avg_height
                    iou = _bbox_iou(bbox1, bbox2)
                    edge_gap_ratio = _bbox_edge_gap(bbox1, bbox2) / avg_height

                    if pair_key not in ft['pairs']:
                        ft['pairs'][pair_key] = {
                            'distance_history': [],
                            'contact_time': 0.0,
                            'mutual_motion_time': 0.0,
                            'last_update_time': current_time,
                            'last_alert_time': 0.0,
                        }
                    pair_data = ft['pairs'][pair_key]
                    dt = min(
                        max(current_time - pair_data.get('last_update_time', current_time), 0.0),
                        FIGHT_STATE_MAX_STEP_SECONDS,
                    )
                    pair_data['last_update_time'] = current_time

                    # 更新距离历史
                    pair_data['distance_history'].append(normalized_distance)
                    if len(pair_data['distance_history']) > FIGHT_HISTORY_SIZE:
                        pair_data['distance_history'].pop(0)

                    motion1 = pdata1.get('motion', 0.0)
                    motion2 = pdata2.get('motion', 0.0)
                    motion_threshold1 = max(FIGHT_MIN_MOTION_PIXELS, height1 * FIGHT_MOTION_RATIO)
                    motion_threshold2 = max(FIGHT_MIN_MOTION_PIXELS, height2 * FIGHT_MOTION_RATIO)
                    mutual_high_motion = motion1 >= motion_threshold1 and motion2 >= motion_threshold2
                    close_contact = (
                        normalized_distance <= FIGHT_CLOSE_DISTANCE_RATIO and
                        (iou >= FIGHT_IOU_THRESHOLD or edge_gap_ratio <= FIGHT_EDGE_GAP_RATIO)
                    )

                    if close_contact:
                        pair_data['contact_time'] = min(pair_data['contact_time'] + dt, 3.0)
                    else:
                        pair_data['contact_time'] = max(0.0, pair_data['contact_time'] - dt * FIGHT_STATE_DECAY)

                    if close_contact and mutual_high_motion:
                        pair_data['mutual_motion_time'] = min(pair_data['mutual_motion_time'] + dt, 3.0)
                    else:
                        pair_data['mutual_motion_time'] = max(
                            0.0,
                            pair_data['mutual_motion_time'] - dt * (FIGHT_STATE_DECAY + 0.5),
                        )

                    history = pair_data['distance_history']
                    enough_history = len(history) >= FIGHT_HISTORY_MIN_SAMPLES
                    swings = _count_distance_swings(history[-FIGHT_SWING_WINDOW:])
                    was_far_recently = enough_history and max(history) >= FIGHT_FAR_DISTANCE_RATIO
                    strong_contact = iou >= FIGHT_STRONG_IOU_THRESHOLD or edge_gap_ratio <= FIGHT_STRONG_EDGE_GAP_RATIO

                    if (
                        enough_history and
                        pair_data['contact_time'] >= FIGHT_CONTACT_DURATION and
                        pair_data['mutual_motion_time'] >= FIGHT_MUTUAL_MOTION_DURATION and
                        swings >= FIGHT_SWING_REQUIRED and
                        (was_far_recently or strong_contact) and
                        current_time - pair_data.get('last_alert_time', 0.0) > FIGHT_PAIR_COOLDOWN
                    ):
                        pair_data['last_alert_time'] = current_time
                        ft['last_alert_time'] = current_time
                        self.stats['fight'] = self.stats.get('fight', 0) + 1
                        print(
                            f"[打架检测] {camera_name}: tid={tid1} 与 tid={tid2} 打架确认! "
                            f"contact={pair_data['contact_time']:.2f}s, "
                            f"mutual_motion={pair_data['mutual_motion_time']:.2f}s, "
                            f"norm_dist={normalized_distance:.2f}, iou={iou:.2f}, "
                            f"gap={edge_gap_ratio:.2f}, swings={swings}, "
                            f"motion=({motion1:.1f},{motion2:.1f})"
                        )
                        return True  # 返回True表示触发了打架告警


# ==============================================================================
# VLM 分析和上传功能
# ==============================================================================

def get_prompt_vlm(dtype):
    """VLM提示词生成函数"""
    dtype = dtype.lower()

    # 危险物品检测
    if 'dangerous' in dtype or 'weapon' in dtype or 'knife' in dtype or 'stick' in dtype or '危险物品' in dtype:
        return """# Role
公共安全分析专家
# 重要说明
图片中的【红色方框】是标注框（bounding box），用于标记需要分析的区域。
红色方框本身不是要分析的物体，你只需要分析红色方框【范围内】的实际画面内容。

# Task
判断红色方框范围内的物品是否具有危险性。

# 判断规则
**危险物品（signal=true）**：
- 棍棒/木棍/铁棍/竹竿/钢管等可挥动的棒状物体
- 刀/匕首/斧头等锐器
- 砖头/碎酒瓶（可作为武器的硬物）

**安全物品（signal=false）**：
- 扫帚/拖把/拖布（清洁工具）
- 晾衣杆/衣架（日常用品）
- 伞/水杯/手机/食物/包包
- 消防栓/灭火器/电梯按钮/门把手等设施
- 看不清物体轮廓、物体太小、被遮挡 → signal=false（宁缺毋滥）

# Output（直接输出JSON，不要其他内容）
{"signal": true/false, "behavior_type": "dangerous_item", "description": "简要描述判断结果"}"""

    # 火灾/烟雾检测
    elif 'fire' in dtype or 'smoke' in dtype:
        obj_type = "明火" if 'fire' in dtype else "烟雾"
        behavior_type = "fire" if 'fire' in dtype else "smoke"
        return f"""# Role
消防鉴别专家
# 重要说明
图片中的【红色方框】是标注框（bounding box），用于标记需要分析的区域。
红色方框本身不是要分析的物体，你只需要分析红色方框【范围内】的实际画面内容。

# Task
辨别红色方框范围内是真实{obj_type}还是误报。

# 判断规则
**signal=false (误报)**：
- 光源干扰：形状规则的亮斑（圆形/方形）、刺眼白光
- 环境干扰：地面/墙面/金属上的反光倒影
- 物体干扰：橙红色衣服、塑料袋、标牌
- 自然环境：雾气、水蒸气（与火灾无关）

**signal=true (真实告警)**：
- 火焰：有明显根部、不规则跳动感
- 烟雾：有火焰相伴，浓密、持续扩散

# Output（直接输出JSON，不要其他内容）
{{"signal": true/false, "behavior_type": "{behavior_type}", "description": "简要描述判断结果"}}"""

    # 香烟检测
    elif 'cigarette' in dtype or '吸烟' in dtype:
        return """# Role
香烟鉴别专家

# 任务
判断红色方框标注区域内是否有人正在吸烟。

# 红色方框说明
红色方框是检测模型标记的可疑区域，你需要分析这个区域内的实际画面内容。

# 正面判断标准（满足任一即 signal=true）
1. 人物嘴边叼着白色/浅色带滤嘴的条状物
2. 人物手指夹着香烟，正在靠近嘴边
3. 人物手中拿着点燃的香烟（可见烟雾/火光）

# 负面判断标准（满足任一即 signal=false）
1. 红色方框内没有人
2. 红色方框内只有静止的香烟（未在使用）
3. 类似香烟的物体：笔、筷子、数据线、吸管等条状物
4. 红色方框内只有烟雾/反光/阴影等干扰
5. 深色条状物无法确认为香烟

# 常见误判场景
- 白色耳机线 → 不是香烟
- 手指指纹/褶皱 → 不是香烟
- 反光/光斑 → 不是香烟
- 深色物体 → 无法确认为香烟，应为 false

# 输出要求
只根据红色方框内的明确画面判断，不要推测。
无法确认时，倾向于 signal=false。

# 输出格式
{"signal": true/false, "behavior_type": "cigarette", "description": "简洁描述方框内实际看到的画面"}"""

    # 机动车违停复核：本项目按 car 类目标复核
    elif dtype == 'motor_vehicle_violations':
        return """# Role
车辆目标复核专家
# 重要说明
图片中的【红色方框】是检测框，只分析红色方框【范围内】的实际画面内容。
红框本身不是目标，不要分析红框外区域。

# Task
判断红色方框内是否确实是【机动车目标】。

# 本项目定义
- 本项目这里的“机动车目标”按 car 类目标复核
- 只要红框内是轿车/SUV/面包车/封闭车身汽车/小货车这类 car 外观，就判定 signal=true
- 无法确认时，一律 signal=false

# signal=true 条件
- 清楚看到封闭车身
- 有车窗/车门/车头车尾等汽车结构
- 车辆主体占据红框主要区域

# signal=false 条件
- 自行车
- 摩托车 / 电动车 / 两轮车
- 三轮车
- 人、箱子、栏杆、反光、阴影
- 只看到很小一部分，无法确认是不是汽车
- 红框中没有明确车辆主体

# 输出格式
{"signal": true/false, "behavior_type": "motor_vehicle_violations", "description": "简要描述红框内看到的是汽车还是非汽车"}"""

    # 非机动车违停复核：按本项目定义，bicycle 和 motorcycle 都算这一类
    elif dtype == 'non_motor_violations':
        return """# Role
车辆目标复核专家
# 重要说明
图片中的【红色方框】是检测框，只分析红色方框【范围内】的实际画面内容。
红框本身不是目标，不要分析红框外区域。

# Task
判断红色方框内是否确实是【非机动车目标】。

# 本项目定义
- 本项目这里的“非机动车目标”按检测类别复核
- 只要红框内明确是 bicycle 或 motorcycle，就判定 signal=true
- 不区分现实交通法规分类，只按本项目检测类别判断
- 无法确认时，一律 signal=false

# signal=true 条件
- 清楚看到自行车：车把、车轮、车架、脚踏结构
- 清楚看到摩托车：两轮、车把、车身/座椅结构

# signal=false 条件
- 轿车/SUV/货车等汽车
- 人、箱子、反光、阴影、栏杆
- 婴儿车、轮椅、手推车
- 只看到局部，无法确认是 bicycle 或 motorcycle
- 红框中没有明确车辆主体

# 输出格式
{"signal": true/false, "behavior_type": "non_motor_violations", "description": "简要描述红框内看到的是自行车/摩托车还是其他物体"}"""

    # ================= 序列行为类检测 =================
    elif 'fall' in dtype:
        return """# Role
动作序列分析专家
# Task
分析连续视频帧，判断是否发生【摔倒】。
# Analysis Logic
1. **摔倒 (signal=true)**：从"站立"到"失去平衡"再到"倒地"的连续过程，且倒地后未立即起身。
2. **非摔倒 (signal=false)**：主动弯腰、下蹲捡东西、系鞋带（重心受控）；或倒地后迅速恢复站立。
# Output
JSON: {"signal": boolean, "behavior_type": "fall", "description": "描述动作变化过程"}"""

    elif 'fight' in dtype:
        return """# Role
暴力行为分析专家
# Task
分析连续视频帧，判断是否存在【打架】。
# Analysis Logic
1. **打架 (signal=true)**：连续的挥拳、踢腿、推搡，且动作剧烈、有伤害意图。
2. **非打架 (signal=false)**：拥抱、嬉戏打闹（动作轻柔）、单一瞬间的手臂挥舞。
# Output
JSON: {"signal": boolean, "behavior_type": "fight", "description": "描述互动过程及激烈程度"}"""

    # ================= 单图类检测 (极致抗误报优化版) =================

    elif 'power' in dtype or 'failure' in dtype or '停电' in dtype:
        return """# Role
电梯故障检测专家
# Task
判断电梯是否【停电】。
# Constraints
**宁缺毋滥原则**：如果不确定，请判定为 False。
# Rules
1. **signal=true (确认停电)**：
   - **全黑**：电梯内漆黑一片，无法看清物体细节。
   - **并且**：显示屏完全熄灭/无任何显示。
2. **signal=false (正常/误报)**：
   - **有光**：只要能看清电梯内的人或物，说明有应急照明或正常照明。
   - **屏幕亮**：显示屏显示任何内容（数字、文字、红字、绿字）。
   - **干扰排除**：看到 "7#-1", "3栋", "东梯" 等文字是摄像头编号，只要屏幕亮着，就不是停电。
# Output
JSON: {"signal": boolean, "behavior_type": "power_failure", "description": "简述光线和屏幕状态"}"""

    elif 'floor' in dtype or 'stuck' in dtype or '故障' in dtype:
        return """# Role
电梯安全专家
# Task
判断画面中是否有人**正在求救或被困**。

# Rules (严格执行)
1. **signal=true (真告警)** - 只有满足以下**所有**条件才算：
   - 画面中**明确有人**（不是空的）
   - 人员**正在激烈挣扎/呼救**（扒门、拍门、大声呼叫、疯狂挥手）
   - 门是**关闭的**且人员被夹住无法离开

2. **signal=false (误报/正常)** - 满足以下**任一**条件就算False：
   - 画面中**没有人** → 空电梯，不算困人
   - 门是**开着的** → 能自行离开，不算困人
   - 人员在**看手机/打电话** → 情绪稳定，在等电梯
   - 人员**站着/坐着不动** → 没有求救动作
   - 无法从画面判断电梯是否故障 → 默认正常

# Output
JSON: {"signal": boolean, "behavior_type": "elevator_stuck", "description": "简要描述：1)是否有人 2)门状态 3)是否有求救动作"}"""

    elif 'animal' in dtype or 'pet' in dtype or '宠物' in dtype:
        return """# Role
文明养宠监督员
# 重要说明
图片中的【红色方框】是标注框（bounding box），用于标记需要分析的区域。
红色方框本身不是要分析的物体，你只需要分析红色方框【范围内】的实际画面内容。

# Task
判断红色方框范围内的宠物是否违规（未牵绳）。

# 判断规则
**signal=true (违规)**：
- 宠物在地面自由活动，身上没有绳索
- 绳索拖地但无人拉着

**signal=false (合规/存疑)**：
- 看不清是否有绳 → signal=false（宁缺毋滥）
- 宠物被抱在怀里、在笼子里
- 绳子清晰可见有人牵着

# Output（直接输出JSON，不要其他内容）
{"signal": true/false, "behavior_type": "uncivilized_pet", "description": "简要描述判断结果"}"""

    elif 'object' in dtype:
        return """# Role
电梯异物告警判断系统
# 重要说明
图片中的【红色方框】是标注框（bounding box），用于标记需要分析的区域。
红色方框本身不是要分析的物体，你只需要分析红色方框【范围内】的实际画面内容。

# Task
判断红色方框范围内是否有异物。

# 判断流程
1. **找人**：红色方框范围内有人吗？
   - 有人（头/手/脚/身体）→ signal=false

2. **找设施**：红色方框范围内是电梯设施吗？
   - 电梯门/墙面/扶手/按钮/广告屏/地面 → signal=false

3. **确认异物**：红色方框范围内有异物吗？
   - 电动车/自行车/纸箱/包裹/垃圾袋/桶/行李箱 → signal=true
   - 什么都没有 → signal=false

# Output（直接输出JSON，不要其他内容）
{"signal": true/false, "behavior_type": "object_detections", "description": "简要描述判断结果"}"""

    elif 'ev' in dtype or '电动车' in dtype or 'motorcycle' in dtype:
        return """# Role
电动车检测专家
# 重要说明
图片中的【红色方框】是标注框（bounding box），用于标记需要分析的区域。
红色方框本身不是要分析的物体，你只需要分析红色方框【范围内】的实际画面内容。

# Task
判断红色方框范围内是否为电动车。

## 判断规则

### 不是电动车（signal=false）
- 自行车：两个小轮，车身细
- 滑板车/平衡车：小巧、站立骑行
- 轮椅/婴儿车：有座椅、推车结构
- 人的腿：穿着裤子，关节可弯曲
- 其他物品：纸箱、塑料袋等
- 看不清特征：只看到部分车身，无法判断 → signal=false

### 是电动车（signal=true）
- 脚踏板：下方有矩形脚踏板
- 电瓶：踏板位置有明显凸起
- 车把：龙头状车把
- 后座：后轮上方有长条座椅
- 整体比自行车大，车身较粗

# Output（直接输出JSON，不要其他内容）
{"signal": true/false, "behavior_type": "ev_violations", "description": "简要描述判断结果"}"""

    # 火灾/烟雾检测
    elif 'fire' in dtype or 'smoke' in dtype:
        obj_type = "明火" if 'fire' in dtype else "烟雾"
        behavior_type = "fire" if 'fire' in dtype else "smoke"
        return f"""# Role
消防鉴别专家
# 重要说明
图片中的【红色方框】是标注框（bounding box），用于标记需要分析的区域。
红色方框本身不是要分析的物体，你只需要分析红色方框【范围内】的实际画面内容。

# Task
辨别红色方框范围内是真实{obj_type}还是误报。

# 判断规则
**signal=false (误报)**：
- 光源干扰：形状规则的亮斑（圆形/方形）、刺眼白光
- 环境干扰：地面/墙面/金属上的反光倒影
- 物体干扰：橙红色衣服、塑料袋、标牌
- 自然环境：雾气、水蒸气（与火灾无关）

**signal=true (真实告警)**：
- 火焰：有明显根部、不规则跳动感
- 烟雾：有火焰相伴，浓密、持续扩散

# Output（直接输出JSON，不要其他内容）
{{"signal": true/false, "behavior_type": "{behavior_type}", "description": "简要描述判断结果"}}"""

    # 默认兜底
    else:
        return """# Role
安全监控分析员
# Task
判断画面是否存在极度危险的异常。如果不确定，请返回 False。
# Output
JSON: {"signal": boolean, "behavior_type": "unknown", "description": "简述情况"}"""


def analyze_with_vlm(image_path, detection_type):
    """VLM分析核心函数"""
    try:
        import os
        import json
        import base64

        # 使用Ollama API（GPU加速）
        import requests
        model_name = "qwen3.5:4b"
        api_url = "http://127.0.0.1:11434/api/generate"

        images_b64 = []
        source_type = "single"
        if os.path.isdir(image_path):
            jpg_files = sorted(glob_module.glob(os.path.join(image_path, '*.jpg')))
            if len(jpg_files) > 8:
                step = len(jpg_files) // 8
                jpg_files = jpg_files[::step][:8]
            if not jpg_files: return None
            source_type = "sequence"
            for p in jpg_files:
                with open(p, 'rb') as f: images_b64.append(base64.b64encode(f.read()).decode('utf-8'))
        else:
            if not os.path.exists(image_path): return None
            with open(image_path, 'rb') as f: images_b64.append(base64.b64encode(f.read()).decode('utf-8'))

        payload = {
            "model": model_name,
            "prompt": get_prompt_vlm(detection_type),
            "images": images_b64,
            "stream": False,
            "keep_alive": -1,  # 强制模型永久驻留GPU显存，不卸载
            "options": {"temperature": 0.1, "num_ctx": 4096, "num_gpu": 999}
        }

        # 调用Ollama API
        try:
            response = requests.post(api_url, json=payload, timeout=VLM_HTTP_TIMEOUT)
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '')
            else:
                print(f"[VLM请求失败] {response.status_code}，回退本地模型...")
                return None
        except Exception as e:
            print(f"[VLM连接失败] {e}，回退本地模型...")
            return None

        if not response_text:
            return None

        source_type = "single" if not os.path.isdir(image_path) else "sequence"

        # 处理响应文本
        parsed_json = None
        try:
            import re
            json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)

            parsed_json = json.loads(response_text)
            desc = parsed_json.get('description', '')

            # 语义分析修正
            behavior_type = parsed_json.get('behavior_type', '').lower()

            # 火灾/烟雾检测
            if 'fire' in behavior_type or 'smoke' in behavior_type or '火' in desc or '烟' in desc:
                false_alarm_keywords = ['灯光', '路灯', '车灯', '反光', '反射', '光斑', '光晕', '不是火灾', '不是明火', '不是烟雾', '没有火灾', '没有烟雾']
                if any(word in desc for word in false_alarm_keywords) and parsed_json.get('signal'):
                    print(f"[VLM语义修正] 火灾/烟雾检测：检测到光源/反光干扰表述，强制修正为 False")
                    parsed_json['signal'] = False
                    parsed_json['risk_level'] = 0.1

            status_icon = "✅" if parsed_json['signal'] else "❌"
            print(f"[VLM分析] {detection_type}: {status_icon} {desc[:40]}...")

        except:
            print(f"[VLM分析] JSON解析警告: {response_text[:40]}...")
            parsed_json = {
                'signal': False,
                'behavior_type': detection_type,
                'description': response_text[:100] + "..." if len(response_text) > 100 else response_text
            }

        # 保存VLM结果
        try:
            type_subdir_map = {
                'fire': 'fire_detections',
                'fire_detections': 'fire_detections',
                'smoke': 'smoke_detections',
                'smoke_detections': 'smoke_detections',
                'cigarette': 'cigarette_detections',
                'cigarette_detections': 'cigarette_detections',
                'dangerous_item': 'dangerous_item_detections',
                'dangerous_item_detections': 'dangerous_item_detections',
                'motor_vehicle_violations': 'motor_vehicle_violations',
                'non_motor_violations': 'non_motor_violations',
                'power_failure': 'elevator_power_failures',
                'elevator_power_failures': 'elevator_power_failures',
                'floor_stuck': 'elevator_floor_stuck',
                'elevator_floor_stuck': 'elevator_floor_stuck',
                'object': 'object_detections',
                'object_detections': 'object_detections',
                'fall': 'fall_detections',
                'fall_detections': 'fall_detections',
                'fight': 'fight_detections',
                'fight_detections': 'fight_detections',
                'uncivilized_pet': 'pet_detections',
                'pet_detections': 'pet_detections',
            }
            sub_dir = type_subdir_map.get(detection_type, 'unknown')

            filename = os.path.basename(image_path)
            # 使用绝对路径，确保保存路径一致
            if image_path.startswith('/workspace/'):
                json_path = f"/workspace/output/{sub_dir}/{filename.replace('.jpg', '_vlm_analysis.json')}"
            elif image_path.startswith('/'):
                # 已经是绝对路径，直接提取目录部分
                json_path = f"{os.path.dirname(image_path)}/{filename.replace('.jpg', '_vlm_analysis.json')}"
            else:
                # 相对路径，转换为绝对路径（基于当前工作目录）
                abs_path = os.path.abspath(image_path)
                json_path = f"{os.path.dirname(abs_path)}/{filename.replace('.jpg', '_vlm_analysis.json')}"

            os.makedirs(os.path.dirname(json_path), exist_ok=True)

            with open(json_path, 'w', encoding='utf-8') as f:
                output_data = {
                    'image_path': image_path,
                    'detection_type': detection_type,
                    'source_type': source_type,
                    'timestamp': datetime.now(BEIJING_TZ).strftime('%Y-%m-%d %H:%M:%S'),
                    'vlm_analysis': response_text,
                    'parsed_result': parsed_json
                }
                json.dump(output_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[VLM保存失败] {e}")

        return parsed_json
    except Exception as e:
        print(f"[VLM错误] {e}")
    return None


class VLMQueueManager:
    """VLM分析队列管理器"""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, 'initialized'): return
        from concurrent.futures import ThreadPoolExecutor
        self.executor = ThreadPoolExecutor(max_workers=VLM_MAX_WORKERS, thread_name_prefix='VLM')
        self.pending_tasks = queue.Queue(maxsize=VLM_PENDING_QUEUE_MAXSIZE)
        self.processing_count = 0
        self.completed_count = 0
        self.failed_count = 0
        self.dropped_count = 0
        self.lock = threading.Lock()
        self.initialized = True

    def update_queue_metrics(self):
        """更新队列大小指标到Prometheus"""
        if METRICS_ENABLED:
            pm.QUEUE_SIZE.labels(queue_name='vlm').set(self.pending_tasks.qsize())
            pm.QUEUE_SIZE.labels(queue_name='vlm_processing').set(self.processing_count)

    def submit_task(self, image_path, detection_type):
        # 只对需要VLM的类型提交任务
        if detection_type not in VLM_REQUIRED_TYPES:
            return False
        task = {
            'image_path': image_path,
            'detection_type': detection_type,
            'submit_time': time.time()
        }
        queued = self._enqueue_task(task)
        if not queued:
            return False
        self.update_queue_metrics()
        self._process_queue()
        return True

    def _enqueue_task(self, task):
        dropped_task = None
        with self.lock:
            if self.pending_tasks.full():
                try:
                    dropped_task = self.pending_tasks.get_nowait()
                    self.dropped_count += 1
                    if METRICS_ENABLED:
                        pm.QUEUE_DROPPED.labels(queue_name='vlm').inc()
                except queue.Empty:
                    dropped_task = None
            try:
                self.pending_tasks.put_nowait(task)
                queued = True
            except queue.Full:
                self.dropped_count += 1
                if METRICS_ENABLED:
                    pm.QUEUE_DROPPED.labels(queue_name='vlm').inc()
                queued = False

        self.update_queue_metrics()
        if dropped_task:
            print(f"[VLM队列] 队列已满，丢弃旧任务: {dropped_task['image_path']}")
        elif not queued:
            print(f"[VLM队列] 队列仍满，丢弃新任务: {task['image_path']}")

        return queued

    def _process_queue(self):
        print(f"[VLM队列] 待处理: {self.pending_tasks.qsize()}, 进行中: {self.processing_count}")
        while True:
            try:
                with self.lock:
                    can_schedule = self.processing_count < VLM_MAX_WORKERS
                if not can_schedule:
                    break
                try:
                    task = self.pending_tasks.get(timeout=0.1)
                    print(f"[VLM队列] 获取任务: {task['image_path']}")
                    with self.lock:
                        self.processing_count += 1
                    self.update_queue_metrics()
                    self.executor.submit(self._execute_task, task)
                except queue.Empty:
                    self.update_queue_metrics()
                    break
            except Exception as e:
                print(f"[VLM队列错误] {e}")
                self.update_queue_metrics()
                break

    def _execute_task(self, task):
        print(f"[VLM执行] 开始分析: {task['image_path']}")
        try:
            analyze_with_vlm(task['image_path'], task['detection_type'])
            with self.lock: self.completed_count += 1
            print(f"[VLM执行] 完成: {task['image_path']}")
        except Exception as e:
            print(f"[VLM分析失败] {task['image_path']}: {e}")
            with self.lock: self.failed_count += 1
        finally:
            with self.lock: self.processing_count -= 1
            self.update_queue_metrics()
            self._process_queue()

    def get_stats(self):
        with self.lock:
            return {
                'pending': self.pending_tasks.qsize(),
                'processing': self.processing_count,
                'completed': self.completed_count,
                'failed': self.failed_count,
                'dropped': self.dropped_count,
            }


class RealTimeUploader:
    def __init__(self, camera_configs=None):
        self.base_url = "http://umod.me:6969/api/"
        self.file_upload_url = f"{self.base_url}common/upload"
        self.notification_url = f"{self.base_url}notification/task/emergency"

        self.app_id = "be5aab82-2fda-4f8f-a3a9-f09bf8c04909"
        self.app_secret = "46b85751-44e2-40b5-b4f7-c926327656a4"
        self.auth_header = {
            "X-App-Id": self.app_id,
            "X-App-Secret": self.app_secret
        }

        self.hls_base_url = "http://umod.me:6969/hls"
        if camera_configs:
            self.camera_configs = camera_configs
        else:
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'camera_zones_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.camera_configs = {k.lower(): v for k, v in config.get('cameras', {}).items()}
            else:
                self.camera_configs = {}
        self.upload_queue = queue.Queue(maxsize=UPLOAD_QUEUE_MAXSIZE)
        self.queue_lock = threading.Lock()
        self.dropped_count = 0
        self.running = False
        self.upload_thread = None

    def start(self):
        self.running = True
        self.upload_thread = threading.Thread(target=self._upload_worker, daemon=True)
        self.upload_thread.start()

    def update_queue_metrics(self):
        """更新队列大小指标到Prometheus"""
        if METRICS_ENABLED:
            pm.QUEUE_SIZE.labels(queue_name='upload').set(self.upload_queue.qsize())

    def stop(self):
        self.running = False
        if self.upload_thread: self.upload_thread.join(timeout=1)

    def get_video_stream_url(self, camera_name, camera_key=""):
        camera_key_lower = str(camera_key or "").lower()
        camera_name_lower = camera_name.lower()

        direct_match = self.camera_configs.get(camera_key_lower) if camera_key_lower else None
        if not direct_match:
            direct_match = self.camera_configs.get(camera_name_lower)
        if direct_match and direct_match.get('rtsp_url'):
            return direct_match['rtsp_url']
        if direct_match and direct_match.get('cameraIndexCode'):
            return f"hikvision://{direct_match['cameraIndexCode']}"

        for cam_cfg in self.camera_configs.values():
            if camera_key_lower and str(cam_cfg.get('cameraIndexCode', '')).lower() == camera_key_lower and cam_cfg.get('rtsp_url'):
                return cam_cfg['rtsp_url']
            if camera_key_lower and str(cam_cfg.get('cameraIndexCode', '')).lower() == camera_key_lower and cam_cfg.get('cameraIndexCode'):
                return f"hikvision://{cam_cfg['cameraIndexCode']}"
            if cam_cfg.get('name', '').lower() == camera_name_lower and cam_cfg.get('rtsp_url'):
                return cam_cfg['rtsp_url']
            if cam_cfg.get('name', '').lower() == camera_name_lower and cam_cfg.get('cameraIndexCode'):
                return f"hikvision://{cam_cfg['cameraIndexCode']}"
            if cam_cfg.get('display_name', '').lower() == camera_name_lower and cam_cfg.get('rtsp_url'):
                return cam_cfg['rtsp_url']
            if cam_cfg.get('display_name', '').lower() == camera_name_lower and cam_cfg.get('cameraIndexCode'):
                return f"hikvision://{cam_cfg['cameraIndexCode']}"

        # 兼容旧命名规则作为兜底
        cam_num = camera_name_lower.replace('camera', '').replace(' ', '').lstrip('0') or '1'
        return f"http://umod.me:6969/live/cam{cam_num}.flv"

    def _enqueue_upload_task(self, data):
        dropped_item = None
        with self.queue_lock:
            if self.upload_queue.full():
                try:
                    dropped_item = self.upload_queue.get_nowait()
                    self.dropped_count += 1
                    if METRICS_ENABLED:
                        pm.QUEUE_DROPPED.labels(queue_name='upload').inc()
                except queue.Empty:
                    dropped_item = None
            try:
                self.upload_queue.put_nowait(data)
                queued = True
            except queue.Full:
                self.dropped_count += 1
                if METRICS_ENABLED:
                    pm.QUEUE_DROPPED.labels(queue_name='upload').inc()
                queued = False

        self.update_queue_metrics()
        if dropped_item:
            print(f"[UploadQueue] 队列已满，丢弃旧任务: {dropped_item.get('temp_path')}")
        elif not queued:
            print(f"[UploadQueue] 队列仍满，丢弃新任务: {data.get('temp_path')}")

        return queued

    def _normalize_detection_type(self, detection_type):
        type_normalize_map = {
            'fire_detections': 'fire',
            'smoke_detections': 'smoke',
            'cigarette_detections': 'cigarette',
            'dangerous_item_detections': 'dangerous_item',
            'motor_vehicle_violations': 'motor_vehicle_violations',
            'non_motor_violations': 'non_motor_violations',
            'elevator_power_failures': 'power_failure',
            'elevator_floor_stuck': 'floor_stuck',
            'object_detections': 'object',
            'fall_detections': 'fall',
            'fight_detections': 'fight',
            'pet_detections': 'uncivilized_pet',
        }
        return type_normalize_map.get(detection_type, detection_type)

    def _resolve_vlm_json_path(self, local_vlm_path, detection_type):
        if not local_vlm_path:
            return ""
        if os.path.isdir(local_vlm_path):
            return os.path.join(local_vlm_path, 'vlm_analysis.json')

        normalized_type = self._normalize_detection_type(detection_type)
        folder_name_map = {
            'fire': 'fire_detections',
            'smoke': 'smoke_detections',
            'cigarette': 'cigarette_detections',
            'dangerous_item': 'dangerous_item_detections',
            'motor_vehicle_violations': 'motor_vehicle_violations',
            'non_motor_violations': 'non_motor_violations',
            'power_failure': 'elevator_power_failures',
            'floor_stuck': 'elevator_floor_stuck',
            'object': 'object_detections',
            'fall': 'fall_detections',
            'fight': 'fight_detections',
            'uncivilized_pet': 'pet_detections',
        }
        folder_name = folder_name_map.get(normalized_type, detection_type)

        filename = os.path.basename(local_vlm_path)
        if local_vlm_path.startswith('/workspace/'):
            return f"/workspace/output/{folder_name}/{filename.replace('.jpg', '_vlm_analysis.json')}"
        if local_vlm_path.startswith('/'):
            return f"{os.path.dirname(local_vlm_path)}/{filename.replace('.jpg', '_vlm_analysis.json')}"

        abs_path = os.path.abspath(local_vlm_path)
        return f"{os.path.dirname(abs_path)}/{filename.replace('.jpg', '_vlm_analysis.json')}"

    def _try_load_vlm_result(self, vlm_json_path):
        if not vlm_json_path or not os.path.exists(vlm_json_path):
            return None
        try:
            with open(vlm_json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (OSError, ValueError, json.JSONDecodeError):
            return None

    def _wait_for_vlm_result(self, camera_name, vlm_json_path):
        max_wait = time.time() + VLM_RESULT_WAIT_TIMEOUT
        while time.time() < max_wait:
            vlm_data = self._try_load_vlm_result(vlm_json_path)
            if vlm_data is not None:
                print(f"[{camera_name}] ✅ VLM分析完成: {os.path.basename(vlm_json_path)}")
                return vlm_data
            time.sleep(VLM_RESULT_POLL_INTERVAL)

        print(f"[{camera_name}] ⚠️ VLM分析超时 ({VLM_RESULT_WAIT_TIMEOUT}s)，跳过上传和通知")
        return {
            '_timeout_override': True,
            'signal': None,
            'status': '超时丢弃',
            'analysis': f'VLM分析超时({VLM_RESULT_WAIT_TIMEOUT}s)，已跳过上传和通知'
        }

    def upload_detection(self, file_path, camera_name, detection_type, camera_ip="", wait_vlm=False, local_vlm_path="", camera_key=""):
        try:
            if not os.path.exists(file_path):
                print(f"[上传错误] 文件不存在: {file_path}")
                return
            stream_url = self.get_video_stream_url(camera_name, camera_key=camera_key)
            data = {
                'temp_path': file_path,
                'camera_name': camera_name,
                'camera_key': camera_key,
                'detection_type': detection_type,
                'is_sequence': False,
                'camera_ip': stream_url,
                'wait_vlm': wait_vlm,
                'local_vlm_path': local_vlm_path
            }
            self._enqueue_upload_task(data)
        except Exception as e:
            print(f"[上传错误] 准备数据失败: {e}")

    def _upload_worker(self):
        print(f"[UploadWorker] 启动，running={self.running}")
        while self.running:
            try:
                item = self.upload_queue.get(timeout=1)
                self.update_queue_metrics()
            except queue.Empty:
                continue
            try:
                print(f"[UploadWorker] 收到任务: camera={item.get('camera_name')}, type={item.get('detection_type')}")
                file_path = item.get('temp_path')
                camera_name = item['camera_name']
                camera_key = item.get('camera_key', '')
                detection_type = item['detection_type']
                camera_ip = item.get('camera_ip', '')
                wait_vlm = item.get('wait_vlm', False)
                local_vlm_path = item.get('local_vlm_path', '')

                normalized_type = self._normalize_detection_type(detection_type)
                need_wait_vlm = wait_vlm or (normalized_type in VLM_REQUIRED_TYPES and local_vlm_path)
                vlm_json_path = self._resolve_vlm_json_path(local_vlm_path, detection_type) if need_wait_vlm else ""
                vlm_result_data = None

                if need_wait_vlm and vlm_json_path:
                    vlm_result_data = self._wait_for_vlm_result(camera_name, vlm_json_path)
                    if isinstance(vlm_result_data, dict):
                        parsed_result = vlm_result_data.get('parsed_result', {})
                        vlm_signal = parsed_result.get('signal', vlm_result_data.get('signal', None))
                        if vlm_result_data.get('_timeout_override') or vlm_signal is None:
                            print(f"[{camera_name}] ⏭️ VLM超时或未完成，跳过上传和通知: {detection_type}")
                            continue

                print(f"[UploadWorker] 开始上传: {file_path}")
                server_path = self._upload_file(file_path)
                print(f"[UploadWorker] 上传结果: {server_path}")
                if not server_path:
                    print(f"[上传失败] 文件上传失败: {file_path}")
                    continue

                print(f"[UploadWorker] 开始发送通知: camera={camera_name}, type={detection_type}")
                self._send_notification(
                    camera_name,
                    detection_type,
                    server_path,
                    camera_ip,
                    camera_key=camera_key,
                    vlm_json_path=vlm_json_path,
                    vlm_result_data=vlm_result_data
                )
                print(f"[UploadWorker] 任务完成")
            except Exception as e:
                print(f"[上传错误] {e}")
            finally:
                self.upload_queue.task_done()
                self.update_queue_metrics()

    def _upload_file(self, file_path):
        try:
            abs_path = os.path.abspath(file_path)
            if not os.path.exists(abs_path):
                return None
            filename = os.path.basename(abs_path)
            with open(abs_path, 'rb') as f:
                files = {'file': (filename, f, 'image/jpeg')}
                response = requests.post(self.file_upload_url, files=files, headers=self.auth_header, timeout=30)
            if response.status_code == 200:
                result = response.json()
                if result.get('code') == 0:
                    if 'url' in result:
                        fp = result['url']
                        print(f"[文件上传成功] {fp}")
                        return fp
                    elif 'data' in result and 'url' in result.get('data', {}):
                        fp = result['data']['url']
                        print(f"[文件上传成功] {fp}")
                        return fp
            return None
        except Exception as e:
            print(f"[文件上传异常] {e}")
            return None

    def _send_notification(self, camera_name, detection_type, file_path, camera_ip="", camera_key="", vlm_json_path="", vlm_result_data=None):
        try:
            cam_key = str(camera_key or camera_name).lower()
            cam_config = self.camera_configs.get(cam_key, {})
            if not cam_config and camera_key:
                for cfg in self.camera_configs.values():
                    if str(cfg.get('cameraIndexCode', '')).lower() == cam_key:
                        cam_config = cfg
                        break
            if not cam_config:
                cam_config = self.camera_configs.get(camera_name.lower(), {})
            display_name = cam_config.get('name', camera_name)

            type_map = {
                'fire': ('火灾检测', f'{display_name} 检测到火焰', 'A'),
                'smoke': ('烟雾检测', f'{display_name} 检测到烟雾', 'A'),
                'cigarette': ('吸烟检测', f'{display_name} 检测到吸烟行为', 'A'),
                'dangerous_item': ('危险行为', f'{display_name} 检测到危险行为', 'B'),
                'motor_vehicle_violations': ('机动车违停', f'{display_name} 存在机动车违停', 'C'),
                'non_motor_violations': ('非机动车违停', f'{display_name} 存在非机动车违停', 'C'),
                'power_failure': ('电梯停电', f'{display_name} 电梯发生停电故障', 'A'),
                'floor_stuck': ('电梯故障', f'{display_name} 电梯楼层号长时间无变化', 'A'),
                'uncivilized_pet': ('不文明养宠', f'{display_name} 检测到不文明养宠行为', 'C'),
            }

            normalized_type = self._normalize_detection_type(detection_type)
            type_info = type_map.get(normalized_type, ('安全告警', f'{display_name} 发生异常事件', 'C'))
            title_suffix, content_base, risk_level = type_info
            timestamp = datetime.now(BEIJING_TZ).strftime('%Y-%m-%d %H:%M:%S')
            content = f"{timestamp} {display_name} 发生{title_suffix}。"

            # 读取VLM结果
            vlm_analysis = ""
            vlm_status = ""
            vlm_signal = None

            # 需要VLM验证的检测类型
            vlm_required_types_check = VLM_REQUIRED_TYPES

            if isinstance(vlm_result_data, dict):
                vlm_analysis = vlm_result_data.get('parsed_result', {}).get('description', vlm_result_data.get('vlm_analysis', ''))
                vlm_signal = vlm_result_data.get('parsed_result', {}).get('signal', None)
                vlm_status = "警告" if vlm_signal is True else ("误报" if vlm_signal is False else "")
                print(f"[VLM读取] signal={vlm_signal}, {os.path.basename(vlm_json_path) if vlm_json_path else 'inline'}")
            elif normalized_type in vlm_required_types_check and vlm_json_path:
                vlm_data = self._try_load_vlm_result(vlm_json_path)
                if vlm_data is not None:
                    vlm_analysis = vlm_data.get('parsed_result', {}).get('description', vlm_data.get('vlm_analysis', ''))
                    vlm_signal = vlm_data.get('parsed_result', {}).get('signal', None)
                    vlm_status = "警告" if vlm_signal is True else ("误报" if vlm_signal is False else "")
                    print(f"[VLM读取] signal={vlm_signal}, {os.path.basename(vlm_json_path)}")

            if normalized_type in vlm_required_types_check and vlm_signal is None:
                print(f"[{camera_name}] ⏭️ VLM分析未完成(signal=None)，跳过通知: {title_suffix}")
                return

            # 误报也发送通知
            if vlm_signal is False:
                title_suffix = "误报核实"
                content = f"{timestamp} {display_name} {type_info[0]}经AI核实为误报。"
                print(f"[{camera_name}] 📢 发送误报通知: {title_suffix}")

            related_type_map = {
                'fire': 'fire_detection',
                'smoke': 'smoke_detection',
                'cigarette': 'cigarette_detection',
                'dangerous_item': 'dangerous_item_detection',
                'motor_vehicle_violations': 'motor_vehicle_violation',
                'non_motor_violations': 'non_motor_violation',
                'fight': 'fight_detection',
                'fall': 'fall_detection',
                'uncivilized_pet': 'pet_detection',
            }

            payload = {
                "type": "emergency_alert",
                "msg_type": "emergency_alert",
                "template": "system_announcement",
                "data": {
                    "title": f"摄像头{title_suffix}",
                    "content": content,
                    "related_id": f"MALERT-{datetime.now(BEIJING_TZ).strftime('%Y%m%d%H%M%S')}",
                    "risk_level": risk_level,
                    "related_type": related_type_map.get(normalized_type, 'safety_alert'),
                    "image": file_path,
                    "camera_ip": camera_ip,
                    "vlm_status": vlm_status,
                    "vlm_analysis": vlm_analysis
                },
                "target_channel": "staff",
                "role_ids": [0],
                "user_ids": []
            }
            print(f"[通知发送] payload: {json.dumps(payload, ensure_ascii=False)}")

            import requests
            response = requests.post(self.notification_url, json=payload, headers=self.auth_header, timeout=30)
            if response.status_code == 200:
                result = response.json()
                print(f"[通知调试] 响应内容: {result}")
                success = (result.get('code') == 0 and result.get('msg') == '操作成功') or \
                          '成功' in str(result.get('data', {}).get('message', '')) or \
                          'queued' in str(result.get('message', ''))
                if success:
                    print(f"[通知发送成功] {title_suffix} -> {camera_name}")
                    # Prometheus指标 - 告警触发计数
                    if METRICS_ENABLED:
                        pm.ALARMS_TRIGGERED.labels(camera=camera_name, alarm_type=normalized_type).inc()
                else:
                    print(f"[通知发送失败] code={result.get('code')}, message={result.get('message')}")
            else:
                print(f"[通知发送失败] {response.status_code}: {response.text[:100]}")
        except Exception as e:
            print(f"[发送通知异常] {e}")


class FrameProcessor:
    """异步帧处理器：所有重活都在这个线程里做，不阻塞 GStreamer 主循环"""

    def __init__(self, detection_handler, image_saver, vlm_manager, uploader, async_elevator_detector, class_thresholds, camera_id_to_name, camera_id_to_key, gpu_id, pipeline_id, frame_stats, camera_process_last_seen, camera_process_ever_seen, process_seen_lock):
        self.detection_handler = detection_handler
        self.image_saver = image_saver
        self.vlm_manager = vlm_manager
        self.uploader = uploader
        self.async_elevator_detector = async_elevator_detector
        self.class_thresholds = class_thresholds
        self.camera_id_to_name = camera_id_to_name
        self.camera_id_to_key = camera_id_to_key
        self.gpu_id = gpu_id
        self.pipeline_id = pipeline_id
        self.log_prefix = f"[GPU{gpu_id} Pipeline-{pipeline_id}]"
        self.frame_stats = frame_stats
        self.camera_process_last_seen = camera_process_last_seen
        self.camera_process_ever_seen = camera_process_ever_seen
        self.process_seen_lock = process_seen_lock
        self.input_queue = queue.Queue(maxsize=FRAME_PROCESSOR_QUEUE_MAXSIZE)
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        print(f"{self.log_prefix} FrameProcessor 异步处理线程已启动")

    def update_queue_metrics(self):
        if METRICS_ENABLED:
            pm.QUEUE_SIZE.labels(queue_name='frame_processor').set(self.input_queue.qsize())

    def enqueue(self, frame_info):
        """probe 调这个，只入队不阻塞"""
        try:
            self.input_queue.put_nowait(frame_info)
            self.update_queue_metrics()
            return True
        except queue.Full:
            dropped_batches = 0
            dropped_frames = 0
            while True:
                try:
                    dropped_old = self.input_queue.get_nowait()
                except queue.Empty:
                    break
                dropped_batches += 1
                dropped_frames += len(dropped_old.get('frames', []))
                try:
                    self.input_queue.task_done()
                except ValueError:
                    pass

            if dropped_batches > 0:
                self.frame_stats['replaced_batches'] = self.frame_stats.get('replaced_batches', 0) + dropped_batches
                self.frame_stats['replaced_frames'] = self.frame_stats.get('replaced_frames', 0) + dropped_frames
                try:
                    self.input_queue.put_nowait(frame_info)
                    #print(
                    #    f"{self.log_prefix} FrameProcessor队列已满，清理旧积压 {dropped_batches} 批/{dropped_frames} 帧，"
                    #    f"保留最新批次 {len(frame_info.get('frames', []))} 帧",
                    #    flush=True,
                   # )
                    if METRICS_ENABLED:
                        pm.QUEUE_DROPPED.labels(queue_name='frame_processor').inc()
                    self.update_queue_metrics()
                    return True
                except queue.Full:
                    pass

            self.frame_stats['dropped_batches'] = self.frame_stats.get('dropped_batches', 0) + 1
            self.frame_stats['dropped_frames'] = self.frame_stats.get('dropped_frames', 0) + len(frame_info.get('frames', []))
            if METRICS_ENABLED:
                pm.QUEUE_DROPPED.labels(queue_name='frame_processor').inc()
            self.update_queue_metrics()
            return False  # 极端竞争下仍丢弃最新批次，gstreamer 继续跑

    def stop(self):
        self.running = False
        self.thread.join(timeout=5)

    def _worker(self):
        """worker 线程：while True 慢慢处理，不影响 pipeline"""
        while self.running:
            try:
                batch_payload = self.input_queue.get(timeout=1)
                self.update_queue_metrics()
            except queue.Empty:
                continue

            try:
                batch_frames = batch_payload.get('frames', [])
                num_frames = batch_payload.get('num_frames', len(batch_frames))
                enqueued_at = batch_payload.get('timestamp')
                current_time = time.time()
                queue_delay = max(0.0, current_time - enqueued_at) if enqueued_at else 0.0

                self.frame_stats['total_processed'] += num_frames
                self.frame_stats['batch_count'] += 1
                self.frame_stats['last_process_batch_time'] = current_time
                self.frame_stats['last_process_queue_delay'] = queue_delay
                self.frame_stats['last_process_batch_size'] = num_frames

                if self.frame_stats['batch_count'] % 100 == 0:
                    total_time = current_time - self.frame_stats.get('start_time', current_time)
                    avg_fps = self.frame_stats['total_processed'] / total_time if total_time > 0 else 0
                    print(f"{self.log_prefix} 帧统计: 批次={self.frame_stats['batch_count']}, 本批次={num_frames}, 平均速度={avg_fps:.1f}帧/秒, 排队延迟={queue_delay:.3f}s", flush=True)
                if queue_delay >= PROCESS_STALL_WARN_THRESHOLD and self.input_queue.qsize() > 0:
                    print(f"{self.log_prefix} 处理队列延迟 {queue_delay:.2f}s, batch_frames={num_frames}, queue_size={self.input_queue.qsize()}")

                self.frame_stats['last_batch_time'] = current_time
                if 'start_time' not in self.frame_stats:
                    self.frame_stats['start_time'] = current_time

                try:
                    with open(f"/tmp/gpu{self.gpu_id}_heartbeat", "w") as f:
                        f.write(str(int(current_time)))
                except Exception:
                    pass

                for frame_data in batch_frames:
                    source_id = frame_data['source_id']
                    orig_w = frame_data['orig_w']
                    orig_h = frame_data['orig_h']
                    camera_name = frame_data['camera_name']
                    camera_key = frame_data.get('camera_key', camera_name)
                    detections = frame_data.get('detections', [])
                    frame_bgr = frame_data.get('frame')
                    need_elevator_ocr = frame_data.get('need_elevator_ocr', False)
                    need_elevator_obj = frame_data.get('need_elevator_obj', False)
                    frame_captured_at = frame_data.get('frame_captured_at')

                    with self.process_seen_lock:
                        self.camera_process_last_seen[source_id] = current_time
                        self.camera_process_ever_seen[source_id] = True

                    should_save_frame = False
                    alarm_label = "alert"
                    bbox_list = []
                    person_bboxes = [(d['bbox'], d['tracker_id']) for d in detections if d['class_id'] == 0]
                    frame_is_fresh = frame_captured_at is not None and (current_time - frame_captured_at) <= ELEVATOR_FRAME_MAX_AGE

                    def get_frame_copy():
                        if frame_bgr is None:
                            raise RuntimeError("frame snapshot unavailable")
                        return frame_bgr.copy()

                    def set_alarm(label_name, box_label, conf, box):
                        return True, label_name, [(box_label, conf, box)]

                    for detection in detections:
                        confidence = detection['confidence']
                        class_id = detection['class_id']
                        tracker_id = detection['tracker_id']
                        label = detection['label']
                        bbox = detection['bbox']

                        if class_id in [7, 8]:
                            best_person_tid = None
                            best_intersection_area = 0
                            for pbox, ptid in person_bboxes:
                                ix1 = max(bbox[0], pbox[0])
                                iy1 = max(bbox[1], pbox[1])
                                ix2 = min(bbox[2], pbox[2])
                                iy2 = min(bbox[3], pbox[3])
                                if ix1 < ix2 and iy1 < iy2:
                                    intersection_area = (ix2 - ix1) * (iy2 - iy1)
                                    if intersection_area > best_intersection_area:
                                        best_intersection_area = intersection_area
                                        best_person_tid = ptid

                            if best_person_tid is not None:
                                if class_id == 8 and self.detection_handler._check_person_bound_alert(
                                    camera_name, "dangerous_item", best_person_tid, label=label, state_key=camera_key
                                ):
                                    should_save_frame, alarm_label, bbox_list = set_alarm(
                                        "dangerous_item", label, confidence, bbox
                                    )
                                elif class_id == 7 and self.detection_handler._check_person_bound_alert(
                                    camera_name, "cigarette", best_person_tid, label=label, state_key=camera_key
                                ):
                                    should_save_frame, alarm_label, bbox_list = set_alarm(
                                        "cigarette", label, confidence, bbox
                                    )

                        try:
                            handler_result = self.detection_handler.process_detection(
                                camera_id=source_id,
                                camera_name=camera_name,
                                class_id=class_id,
                                confidence=confidence,
                                bbox=bbox,
                                frame_width=orig_w,
                                frame_height=orig_h,
                                tracker_id=tracker_id,
                                camera_key=camera_key
                            )
                            if handler_result.get('fall_triggered'):
                                should_save_frame, alarm_label, bbox_list = set_alarm(
                                    "fall", label, confidence, bbox
                                )
                            if handler_result.get('motor_vio_triggered'):
                                should_save_frame, alarm_label, bbox_list = set_alarm(
                                    "motor_vio", label, confidence, bbox
                                )
                            if handler_result.get('non_motor_vio_triggered'):
                                should_save_frame, alarm_label, bbox_list = set_alarm(
                                    "non_motor_vio", label, confidence, bbox
                                )
                            if handler_result.get('fire_smoke_triggered'):
                                should_save_frame, alarm_label, bbox_list = set_alarm(
                                    label, label, confidence, bbox
                                )
                            if handler_result.get('fight_triggered'):
                                should_save_frame, alarm_label, bbox_list = set_alarm(
                                    "fight", label, confidence, bbox
                                )
                            if handler_result.get('pet_triggered'):
                                should_save_frame, alarm_label, bbox_list = set_alarm(
                                    "uncivilized_pet", label, confidence, bbox
                                )
                        except Exception as e:
                            print(f"[HANDLER ERROR] {e}")

                    if should_save_frame:
                        try:
                            self.image_saver.enqueue(get_frame_copy(), camera_name, alarm_label, bbox_list, camera_key=camera_key)
                        except Exception as e:
                            print(f"[IMAGE EXTRACT ERROR] {camera_name}: {e}")

                    if self.async_elevator_detector and camera_name in self.detection_handler.elevator_status.camera_status:
                        person_count = len(person_bboxes)
                        if need_elevator_ocr:
                            if frame_is_fresh:
                                try:
                                        self.async_elevator_detector.enqueue_ocr(get_frame_copy(), camera_name, person_count, camera_key=camera_key)
                                except Exception as e:
                                    print(f"[OCR ENQUEUE ERROR] {camera_name}: {e}")
                            else:
                                print(f"[OCR SKIP] {camera_name}: frame stale or unavailable")

                        if need_elevator_obj:
                            if frame_is_fresh:
                                try:
                                    if self.async_elevator_detector.enqueue_obj(get_frame_copy(), camera_name, camera_key=camera_key):
                                        self.detection_handler.last_obj_check_time[camera_name] = current_time
                                except Exception as e:
                                    print(f"[OBJ ENQUEUE ERROR] {camera_name}: {e}")
                            else:
                                print(f"[OBJ SKIP] {camera_name}: frame stale or unavailable")
            except Exception as e:
                print(f"{self.log_prefix} FrameProcessor ERROR: {e}")
            finally:
                self.input_queue.task_done()
                self.update_queue_metrics()


class DeepStreamApp:
    def __init__(self, cameras_data, camera_configs=None, detection_handler=None, base_dir=None, gpu_id=0, pipeline_id=0, global_thresholds=None):
        import os
        if base_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_dir = base_dir
        self.gpu_id = gpu_id  # GPU编号
        self.pipeline_id = pipeline_id
        self.pipeline_name = f"Pipeline-{pipeline_id}"
        self.log_prefix = f"[GPU{gpu_id} {self.pipeline_name}]"
        self.heartbeat_file = f"/tmp/gpu{gpu_id}_heartbeat"
        self.cameras_data = cameras_data  # List of dicts with url, name, cameraIndexCode
        self.camera_configs = camera_configs or {}
        self.num_sources = len(cameras_data)
        self.rtsp_urls = [c["url"] for c in cameras_data]
        # 存储每个源的原始URI，用于重连时添加时间戳强制刷新
        self.source_original_uris = {i: c["url"] for i, c in enumerate(cameras_data)}
        self.source_camera_index_codes = {i: c.get("cameraIndexCode", "") for i, c in enumerate(cameras_data)}
        self.rtsp_codec_states = {}
        self.preview_client = ArtemisPreviewClient()
        self.camera_id_to_key = {
            i: (
                c.get("cameraIndexCode")
                or f"camera_{i}"
            )
            for i, c in enumerate(cameras_data)
        }

        # 构建 O(1) 极速查询的类别阈值字典
        global_thresholds = global_thresholds or {}
        default_thresh = global_thresholds.get("default", CONFIDENCE_THRESHOLD)
        self.class_thresholds = {}
        for c_id, c_name in LABEL_NAMES.items():
            self.class_thresholds[c_id] = global_thresholds.get(c_name, default_thresh)
        print(f"{self.log_prefix} 类别阈值表: {self.class_thresholds}")

        # 业务逻辑处理器
        self.detection_handler = detection_handler or DetectionHandler(
            config=None,
            camera_configs=self.camera_configs
        )
        self.pipeline_started_at = time.time()

        # 源心跳与处理心跳分离，避免业务线程卡顿被误判为断流
        self.camera_last_seen = {i: 0.0 for i in range(self.num_sources)}
        self.camera_ever_seen = {i: False for i in range(self.num_sources)}
        self.camera_connect_started_at = {i: 0.0 for i in range(self.num_sources)}
        self.camera_process_last_seen = {i: 0.0 for i in range(self.num_sources)}
        self.camera_process_ever_seen = {i: False for i in range(self.num_sources)}
        self.resetting_cameras = set()
        self.watchdog_lock = threading.Lock()
        self.last_seen_lock = threading.Lock()
        self.process_seen_lock = threading.Lock()
        self.probe_stats_lock = threading.Lock()
        self.revive_queue = queue.Queue()
        self.reconnect_fail_counts = {}
        self.last_reconnect_time = {}
        self.reconnect_grace_until = {}
        self.last_bus_warning_log = {}
        self.last_resource_error_log = {}
        self.last_resource_requeue_log = {}
        self.worker_started_at = time.time()
        self._main_loop = None
        self.restart_requested = False
        self.shutdown_reason = None

        # 异步存图器
        self.image_saver = ImageSaver()

        # 仅在存在电梯摄像头配置时初始化额外模型，避免无效占用资源
        self.async_elevator_detector = None
        if self.detection_handler.elevator_status.camera_status:
            self.async_elevator_detector = AsyncElevatorDetector(
                self.detection_handler.elevator_status,
                self.camera_configs,
                image_saver=self.image_saver
            )

        # VLM分析器和上传器
        self.vlm_manager = VLMQueueManager()
        print(f"[VLMManager] 初始化完成, executor: {self.vlm_manager.executor}")
        self.uploader = RealTimeUploader(camera_configs=self.camera_configs)
        self.uploader.start()

        # 将VLM和上传器传递给ImageSaver
        self.image_saver.set_upload_config(self.vlm_manager, self.uploader)

        # 摄像头ID到名称的映射
        self.camera_id_to_name = {i: cam.get("name", f"camera_{i}") for i, cam in enumerate(cameras_data)}
        self._elevator_ocr_request_time = {}

        # 帧处理统计
        self.frame_stats = {
            'total_processed': 0,
            'last_batch_time': time.time(),
            'batch_count': 0,
            'probe_batches': 0,
            'probe_frames': 0,
            'last_probe_batch_time': 0.0,
            'last_probe_batch_size': 0,
            'last_process_batch_time': 0.0,
            'last_process_queue_delay': 0.0,
            'last_process_batch_size': 0,
            'dropped_batches': 0,
            'dropped_frames': 0,
        }
        self.source_probe_frame_counts = {i: 0 for i in range(self.num_sources)}
        self.source_latest_pts_ns = {i: None for i in range(self.num_sources)}
        self.decode_stats_snapshot_time = time.time()
        self.decode_stats_snapshot_total_frames = 0
        self.decode_stats_snapshot_per_source = dict(self.source_probe_frame_counts)
        self.decode_pts_snapshot_per_source = dict(self.source_latest_pts_ns)
        self.peak_decode_fps = 0.0

        # ✅ 异步帧处理器 - 所有重活都在独立线程里做，不阻塞GStreamer主循环
        self.frame_processor = FrameProcessor(
            detection_handler=self.detection_handler,
            image_saver=self.image_saver,
            vlm_manager=self.vlm_manager,
            uploader=self.uploader,
            async_elevator_detector=self.async_elevator_detector,
            class_thresholds=self.class_thresholds,
            camera_id_to_name=self.camera_id_to_name,
            camera_id_to_key=self.camera_id_to_key,
            gpu_id=self.gpu_id,
            pipeline_id=self.pipeline_id,
            frame_stats=self.frame_stats,
            camera_process_last_seen=self.camera_process_last_seen,
            camera_process_ever_seen=self.camera_process_ever_seen,
            process_seen_lock=self.process_seen_lock
        )

        self.pipeline = Gst.Pipeline()

        self._create_elements()

        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        # 🛡️ 启动看门狗线程
        self.watchdog_running = True
        self._stop_event = threading.Event()  # ✅ 用于优雅停止所有worker线程
        self.watchdog_thread = threading.Thread(target=self._watchdog_worker, daemon=True)
        self.watchdog_thread.start()

        # 🌟 新增：启动单线程"复活维修工"
        self.revive_threads = []
        for revive_worker_id in range(max(1, REVIVE_WORKER_COUNT)):
            revive_thread = threading.Thread(
                target=self._revive_worker,
                args=(revive_worker_id,),
                daemon=True,
            )
            revive_thread.start()
            self.revive_threads.append(revive_thread)

        # 🛡️ 启动独立心跳线程（不依赖 frame processing）
        def heartbeat_worker():
            while not self._stop_event.is_set():
                try:
                    heartbeat_ts = max(
                        self.worker_started_at,
                        self.frame_stats.get('last_probe_batch_time', 0.0),
                        self.frame_stats.get('last_process_batch_time', 0.0),
                        self.frame_stats.get('last_batch_time', 0.0),
                    )
                    with open(self.heartbeat_file, "w") as f:
                        f.write(str(int(heartbeat_ts)))
                except:
                    pass
                if self._stop_event.wait(timeout=10):  # 等待或超时
                    break
        self.heartbeat_thread = threading.Thread(target=heartbeat_worker, daemon=True)
        self.heartbeat_thread.start()

    def _create_elements(self):
        print(f"{self.log_prefix} Creating pipeline with {self.num_sources} sources", flush=True)

        # ==============================================
        # 🛠️ 调优 1: Stream Muxer - 最大化利用 Batch=128 Engine
        # ==============================================
        self.streammux = Gst.ElementFactory.make("nvstreammux", "mux")
        self.streammux.set_property("batch-size", 128)  # 与路数匹配，最大化利用 128 的 Engine
        self.streammux.set_property("live-source", 1)
        self.streammux.set_property("width", 640)
        self.streammux.set_property("height", 640)
        self.streammux.set_property("batched-push-timeout", 15000)  # 15毫秒超时
        self.streammux.set_property("nvbuf-memory-type", 3)  # GPU 内存
        self.streammux.set_property("sync-inputs", 0)

        # Inference
        pgie = Gst.ElementFactory.make("nvinfer", "pgie")
        pgie.set_property("config-file-path", os.path.join(self.base_dir, "config_infer_primary.txt"))

        # ==============================================
        # 追踪器 (Tracker) 配置
        # ==============================================
        tracker = Gst.ElementFactory.make("nvtracker", "tracker")
        # 宽高必须是 32 的倍数
        tracker.set_property('tracker-width', 640)
        tracker.set_property('tracker-height', 384)

        # 自动寻找底层追踪库 (兼容 6.x 和 7.0 版本)
        tracker_so = "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"
        if not os.path.exists(tracker_so):
            tracker_so = "/opt/nvidia/deepstream/deepstream-7.0/lib/libnvds_nvmultiobjecttracker.so"
        tracker.set_property('ll-lib-file', tracker_so)
        tracker_config = os.path.join(self.base_dir, "config_tracker.txt")
        if os.path.exists(tracker_config):
            tracker.set_property('ll-config-file', tracker_config)

        # Converter & OSD
        nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "conv")
        nvvidconv.set_property("nvbuf-memory-type", 3)  # GPU 内存

        # [关键步骤] 强制转换格式为 RGBA (GPU显存)
        capsfilter = Gst.ElementFactory.make("capsfilter", "capsfilter")
        caps = Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA")
        capsfilter.set_property("caps", caps)

        nvosd = Gst.ElementFactory.make("nvdsosd", "osd")
        nvosd.set_property("process-mode", 1)

        # Sink
        sink = Gst.ElementFactory.make("fakesink", "sink")
        sink.set_property("sync", 0)
        sink.set_property("async", False)

        # 添加主要元素
        self.pipeline.add(self.streammux)
        self.pipeline.add(pgie)
        self.pipeline.add(tracker)
        self.pipeline.add(nvvidconv)
        self.pipeline.add(capsfilter)
        self.pipeline.add(nvosd)
        self.pipeline.add(sink)

        # 创建每个源
        for i, cam in enumerate(self.cameras_data):
            url = cam["url"]
            cam_name = cam.get("name", f"camera_{i}")

            print(f"{self.log_prefix} Creating source {i}: {url[:50]}...", flush=True)

            # Queue - 避免 RTSP 延迟堆积
            queue = Gst.ElementFactory.make("queue", f"queue_{i}")
            queue.set_property("max-size-buffers", 30)
            queue.set_property("leaky", 2)  # 下游满时丢弃旧数据
            self.pipeline.add(queue)

            # 根据URL类型选择不同的源处理方式
            if url.startswith("rtsp://"):
                # RTSP streams are handled explicitly so H264/H265 pads can be
                # selected from caps at runtime instead of relying on config.
                print(f"{self.log_prefix} Using explicit RTSP source for {cam_name}")
                rtspsrc = Gst.ElementFactory.make("rtspsrc", f"src_{i}")
                rtspsrc.set_property("location", url)
                for prop_name, prop_value in (
                    ("protocols", 4),
                    ("latency", 300),
                    ("tcp-timeout", 5000000),
                    ("timeout", 5000000),
                    ("drop-on-latency", True),
                    ("do-retransmission", False),
                    ("ntp-sync", False),
                ):
                    try:
                        rtspsrc.set_property(prop_name, prop_value)
                    except Exception:
                        pass

                rtph264depay = Gst.ElementFactory.make("rtph264depay", f"depay_h264_{i}")
                h264parse = Gst.ElementFactory.make("h264parse", f"parse_h264_{i}")
                rtph265depay = Gst.ElementFactory.make("rtph265depay", f"depay_h265_{i}")
                h265parse = Gst.ElementFactory.make("h265parse", f"parse_h265_{i}")
                decoder = Gst.ElementFactory.make("nvv4l2decoder", f"decoder_{i}")
                decoder.set_property("num-extra-surfaces", 1)
                decoder.set_property("drop-frame-interval", 0)

                for element in (rtspsrc, rtph264depay, h264parse, rtph265depay, h265parse, decoder):
                    if not element:
                        raise RuntimeError(f"failed to create RTSP element for source {i}")
                    self.pipeline.add(element)

                rtph264depay.link(h264parse)
                rtph265depay.link(h265parse)

                decoder_src_pad = decoder.get_static_pad("src")
                queue_sink_pad = queue.get_static_pad("sink")
                decoder_src_pad.link(queue_sink_pad)

                queue_src_pad = queue.get_static_pad("src")
                mux_sink_pad = self.streammux.get_request_pad(f"sink_{i}")
                queue_src_pad.link(mux_sink_pad)

                self.rtsp_codec_states[i] = {
                    "h264_depay": rtph264depay,
                    "h264_parser": h264parse,
                    "h265_depay": rtph265depay,
                    "h265_parser": h265parse,
                    "decoder": decoder,
                    "selected": None,
                }

                def make_rtspsrc_callback(source_id=i, camera_name=cam_name):
                    def callback(el, src_pad):
                        caps = src_pad.query_caps(None)
                        caps_text = caps.to_string() if caps else ""
                        caps_lower = caps_text.lower()
                        if "media=(string)video" not in caps_lower and "video" not in caps_lower:
                            if RTSP_PAD_SKIP_LOG:
                                print(
                                    f"{self.log_prefix} RTSP PAD SKIP: {camera_name} non-video caps={caps_text[:120]}",
                                    flush=True,
                                )
                            return

                        if "h264" in caps_lower:
                            codec = "h264"
                            depay_elem = self.rtsp_codec_states[source_id]["h264_depay"]
                            parser_elem = self.rtsp_codec_states[source_id]["h264_parser"]
                        elif "h265" in caps_lower or "hevc" in caps_lower:
                            codec = "h265"
                            depay_elem = self.rtsp_codec_states[source_id]["h265_depay"]
                            parser_elem = self.rtsp_codec_states[source_id]["h265_parser"]
                        else:
                            print(
                                f"{self.log_prefix} RTSP PAD WARNING: {camera_name} unsupported video caps={caps_text[:180]}",
                                flush=True,
                            )
                            return

                        state = self.rtsp_codec_states[source_id]
                        if state.get("selected") and state["selected"] != codec:
                            if RTSP_PAD_SKIP_LOG:
                                print(
                                    f"{self.log_prefix} RTSP PAD SKIP: {camera_name} already selected {state['selected']}, ignore {codec}",
                                    flush=True,
                                )
                            return

                        decoder_elem = state["decoder"]
                        decoder_sink_pad = decoder_elem.get_static_pad("sink")
                        parser_src_pad = parser_elem.get_static_pad("src")
                        if decoder_sink_pad.is_linked():
                            peer_pad = decoder_sink_pad.get_peer()
                            peer_parent = peer_pad.get_parent_element() if peer_pad else None
                            if peer_parent != parser_elem:
                                print(
                                    f"{self.log_prefix} RTSP LINK WARNING: {camera_name} decoder already linked to {peer_parent.get_name() if peer_parent else 'unknown'}, ignore {codec}",
                                    flush=True,
                                )
                                return
                        else:
                            parser_result = parser_src_pad.link(decoder_sink_pad)
                            if parser_result == Gst.PadLinkReturn.OK:
                                if RTSP_LINK_LOG:
                                    print(
                                        f"{self.log_prefix} RTSP LINK: {camera_name} {codec.upper()} parser -> decoder OK",
                                        flush=True,
                                    )
                            else:
                                print(
                                    f"{self.log_prefix} RTSP LINK WARNING: {camera_name} {codec} parser -> decoder failed: {parser_result}",
                                    flush=True,
                                )
                                return

                        sink_pad = depay_elem.get_static_pad("sink")
                        if sink_pad.is_linked():
                            return
                        result = src_pad.link(sink_pad)
                        if result == Gst.PadLinkReturn.OK:
                            state["selected"] = codec
                            if RTSP_LINK_LOG:
                                print(
                                    f"{self.log_prefix} RTSP LINK: {camera_name} caps={codec.upper()} rtspsrc -> depay OK",
                                    flush=True,
                                )
                        else:
                            print(
                                f"{self.log_prefix} RTSP LINK WARNING: {camera_name} {codec} rtspsrc -> depay failed: {result}",
                                flush=True,
                            )
                    return callback

                rtspsrc.connect("pad-added", make_rtspsrc_callback())
                rtspsrc.set_locked_state(True)

            elif url.endswith(".flv") or url.endswith(".m3u8") or url.startswith("http://") or url.startswith("https://"):
                # FLV/HTTP流 - 使用 uridecodebin
                print(f"{self.log_prefix} Using URI source for {cam_name}")
                uridecodebin = Gst.ElementFactory.make("uridecodebin", f"uridecode_{i}")
                uridecodebin.set_property("uri", url)
                def on_source_setup(decodebin, source, source_id=i):
                    try:
                        try:
                            source.set_property("name", f"rtspsrc_{source_id}")
                        except Exception:
                            pass
                        factory = source.get_factory()
                        factory_name = factory.get_name() if factory else source.get_name()
                        if factory_name == "rtspsrc":
                            for prop_name, prop_value in (
                                ("protocols", 4),
                                ("latency", 300),
                                ("tcp-timeout", 5000000),
                                ("timeout", 5000000),
                                ("drop-on-latency", True),
                                ("do-retransmission", False),
                                ("ntp-sync", False),
                            ):
                                try:
                                    source.set_property(prop_name, prop_value)
                                except Exception:
                                    pass
                    except Exception:
                        pass
                try:
                    uridecodebin.connect("source-setup", on_source_setup)
                except Exception:
                    pass
                uridecodebin.set_locked_state(True)  # ⚡ 锁定状态！引擎热身完毕前不准点火！
                self.pipeline.add(uridecodebin)

                def on_deep_element_added(bin, sub_bin, element, user_data):
                    try:
                        factory_name = element.get_factory().get_name()

                        # 🎯 狙击 1：抓获 NVIDIA 硬件解码器
                        if "nvv4l2decoder" in factory_name:
                            element.set_property("drop-frame-interval", 0)
                            element.set_property("num-extra-surfaces", 1)
                            # if hasattr(element.props, 'skip_frames'):
                                # element.set_property("skip-frames", 2)
                                # print(f"[HW OPTIMIZE 🚀] 成功在黑盒深处抓获解码器 {element.get_name()}！强行注入物理跳帧！")

                            # print(f"[HW SAFE] 已为 {element.get_name()} 开启安全软件跳帧！")

                        # 🎯 狙击 2：抓获 HTTP 底层下载器
                        elif "http" in factory_name or "souphttpsrc" in factory_name:
                            element.set_property("timeout", 15)
                            element.set_property("retries", 3)
                            if hasattr(element.props, 'keep_alive'):
                                element.set_property("keep-alive", True)

                    except Exception as e:
                        pass

                # 保存回调到 self，供 _revive_worker 重建时使用
                if not hasattr(self, '_on_deep_element_added_func'):
                    self._on_deep_element_added_func = on_deep_element_added

                uridecodebin.connect("deep-element-added", on_deep_element_added, None)

                # 连接 uridecodebin 的 pad 到 queue（动态链接）
                def make_uridecode_callback(queue_elem):
                    def callback(el, src_pad):
                        caps = src_pad.query_caps(None)
                        if caps and "video" in caps.to_string().lower():
                            sink_pad = queue_elem.get_static_pad("sink")
                            # 🛡️【核心防御】如果已经被上次掉线的死链接占着，先拔掉！
                            if sink_pad.is_linked():
                                peer = sink_pad.get_peer()
                                if peer:
                                    try: peer.unlink(sink_pad)
                                    except: pass
                            src_pad.link(sink_pad)
                    return callback

                # 保存 make_uridecode_callback 到 self，供 _revive_worker 重建时使用
                if not hasattr(self, '_make_uridecode_callback_func'):
                    self._make_uridecode_callback_func = make_uridecode_callback

                uridecodebin.connect("pad-added", make_uridecode_callback(queue))

                # uridecodebin -> queue -> streammux (静态链接)
                queue_src_pad = queue.get_static_pad("src")
                mux_sink_pad = self.streammux.get_request_pad(f"sink_{i}")
                queue_src_pad.link(mux_sink_pad)

            else:
                # 不支持的流类型，打印错误并跳过
                print(f"{self.log_prefix} 不支持的流类型: {url}，支持的类型: .flv, .m3u8, rtsp://, http://, https://")

            # ==============================================
            # 🛠️ 调优 4: 缓解 Python GIL 回调阻塞
            # ==============================================
            time.sleep(0.01)  # 打散底层初始化洪峰

        # streammux -> pgie -> tracker -> nvvidconv -> capsfilter -> nvosd -> sink
        self.streammux.link(pgie)
        pgie.link(tracker)
        tracker.link(nvvidconv)
        nvvidconv.link(capsfilter)
        capsfilter.link(nvosd)
        nvosd.link(sink)

    def add_probe(self):
        """添加探针到 nvosd sink pad (RGBA转换后)"""
        osd = self.pipeline.get_by_name("osd")
        pad = osd.get_static_pad("sink")
        pad.add_probe(Gst.PadProbeType.BUFFER, self._probe)
        print(f"{self.log_prefix} Probe added on nvosd sink")

    def _probe(self, pad, info):
        """复制轻量metadata；仅在需要图像级后处理时同步抓帧"""
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.PASS

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if not batch_meta:
            return Gst.PadProbeReturn.PASS

        batch_frames = []
        temp = batch_meta.frame_meta_list
        while temp:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(temp.data)
            except Exception:
                break
            source_id = frame_meta.source_id
            orig_w = frame_meta.source_frame_width or 1920
            orig_h = frame_meta.source_frame_height or 1080
            camera_name = self.camera_id_to_name.get(source_id, f"camera_{source_id}")
            camera_key = self.camera_id_to_key.get(source_id, f"camera_{source_id}")
            frame_timestamp = time.time()
            frame_pts_ns = getattr(frame_meta, "buf_pts", 0)
            is_elevator_camera = camera_name in self.detection_handler.elevator_status.camera_status
            with self.last_seen_lock:
                self.camera_last_seen[source_id] = frame_timestamp
                self.camera_ever_seen[source_id] = True
            with self.watchdog_lock:
                if source_id in self.reconnect_fail_counts:
                    del self.reconnect_fail_counts[source_id]
                self.reconnect_grace_until.pop(source_id, None)
            with self.probe_stats_lock:
                self.source_probe_frame_counts[source_id] = self.source_probe_frame_counts.get(source_id, 0) + 1
                if frame_pts_ns not in (None, 0, Gst.CLOCK_TIME_NONE):
                    self.source_latest_pts_ns[source_id] = frame_pts_ns

            mux_w, mux_h = 640, 640
            scale_x = orig_w / mux_w
            scale_y = orig_h / mux_h
            capture_frame = False
            detections = []

            l_obj = frame_meta.obj_meta_list
            while l_obj:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except Exception:
                    break

                confidence = obj_meta.confidence
                class_id = obj_meta.class_id
                tracker_id = obj_meta.object_id

                if tracker_id != 18446744073709551615:
                    req_threshold = self.class_thresholds.get(class_id, CONFIDENCE_THRESHOLD)
                    if confidence >= req_threshold:
                        label = LABEL_NAMES.get(class_id, f"class_{class_id}")
                        raw_left = obj_meta.rect_params.left
                        raw_top = obj_meta.rect_params.top
                        raw_width = obj_meta.rect_params.width
                        raw_height = obj_meta.rect_params.height

                        true_left = raw_left * scale_x
                        true_top = raw_top * scale_y
                        true_right = (raw_left + raw_width) * scale_x
                        true_bottom = (raw_top + raw_height) * scale_y

                        bbox = (
                            max(0, min(orig_w - 1, int(true_left))),
                            max(0, min(orig_h - 1, int(true_top))),
                            max(0, min(orig_w - 1, int(true_right))),
                            max(0, min(orig_h - 1, int(true_bottom))),
                        )

                        detections.append({
                            'confidence': confidence,
                            'class_id': class_id,
                            'tracker_id': tracker_id,
                            'label': label,
                            'bbox': bbox,
                        })

                        if class_id in FRAME_SNAPSHOT_REQUIRED_CLASS_IDS:
                            capture_frame = True

                try:
                    l_obj = l_obj.next
                except Exception:
                    break

            person_count = sum(1 for detection in detections if detection['class_id'] == 0)
            need_elevator_ocr = False
            need_elevator_obj = False
            if is_elevator_camera:
                last_ocr_request = self._elevator_ocr_request_time.get(camera_name, 0)
                need_elevator_ocr = (frame_timestamp - last_ocr_request) >= ELEVATOR_OCR_INTERVAL
                last_obj_time = self.detection_handler.last_obj_check_time.get(camera_name, 0)
                need_elevator_obj = person_count == 0 and (frame_timestamp - last_obj_time) >= 10.0
                if need_elevator_ocr or need_elevator_obj:
                    capture_frame = True

            frame_bgr = None
            if capture_frame:
                try:
                    n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
                    frame_rgba = np.array(n_frame, copy=True, order='C')
                    frame_bgr = cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2BGR)
                    frame_bgr = cv2.resize(frame_bgr, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                except Exception as e:
                    print(f"[PROBE FRAME ERROR] {camera_name}: {e}")

            batch_frames.append({
                'source_id': source_id,
                'camera_name': camera_name,
                'camera_key': camera_key,
                'orig_w': orig_w,
                'orig_h': orig_h,
                'detections': detections,
                'frame': frame_bgr,
                'need_elevator_ocr': need_elevator_ocr,
                'need_elevator_obj': need_elevator_obj,
                'frame_captured_at': frame_timestamp if frame_bgr is not None else None,
            })

            try:
                temp = temp.next
            except Exception:
                break

        if not batch_frames:
            return Gst.PadProbeReturn.PASS

        probe_time = time.time()
        self.frame_stats['probe_batches'] = self.frame_stats.get('probe_batches', 0) + 1
        self.frame_stats['probe_frames'] = self.frame_stats.get('probe_frames', 0) + len(batch_frames)
        self.frame_stats['last_probe_batch_time'] = probe_time
        self.frame_stats['last_probe_batch_size'] = len(batch_frames)

        enqueued = self.frame_processor.enqueue({
            'frames': batch_frames,
            'num_frames': len(batch_frames),
            'timestamp': probe_time,
        })
        if not enqueued:
            print(f"{self.log_prefix} FrameProcessor队列已满，丢弃批次 {len(batch_frames)} 帧")
        else:
            for frame_data in batch_frames:
                if frame_data.get('need_elevator_ocr') and frame_data.get('frame_captured_at') is not None:
                    self._elevator_ocr_request_time[frame_data['camera_name']] = frame_data['frame_captured_at']

        return Gst.PadProbeReturn.PASS

    def run(self):
        self.add_probe()
        print(f"{self.log_prefix} Starting pipeline...", flush=True)
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        print(f"{self.log_prefix} Set state result: {ret}", flush=True)
        loop = GLib.MainLoop()
        self._main_loop = loop

        # =======================================================
        # ⚡ 【核心破局】：20秒延迟点火，完美避开引擎加载期的 404 过期风暴！
        # =======================================================
        def delayed_ignition():
            print(f"\n{self.log_prefix} IGNITION: AI 引擎预热完毕，开始批量点燃所有视频源...", flush=True)
            for i in range(self.num_sources):
                if self._stop_event.is_set():
                    break
                try:
                    src = self.pipeline.get_by_name(f"uridecode_{i}")
                    if not src:
                        src = self.pipeline.get_by_name(f"src_{i}")
                    if not src:
                        continue

                    camera_name = self.camera_id_to_name.get(i, f"camera_{i}")
                    source_name = src.get_name() if hasattr(src, "get_name") else ""
                    if source_name.startswith("uridecode_") or source_name.startswith("src_"):
                        try:
                            fresh_uri = self._refresh_source_uri(i, camera_name)
                            if not fresh_uri:
                                raise RuntimeError("empty openUrl")
                            if source_name.startswith("uridecode_"):
                                src.set_property("uri", fresh_uri)
                            else:
                                src.set_property("location", fresh_uri)
                        except Exception as e:
                            self.camera_connect_started_at[i] = time.time()
                            print(
                                f"{self.log_prefix} IGNITION WARNING: {camera_name} 获取 RTSP openUrl 失败，跳过本轮点火: {e}",
                                flush=True,
                            )
                            continue

                    self.camera_connect_started_at[i] = time.time()
                    src.set_locked_state(False)
                    src.set_state(Gst.State.PLAYING)
                    time.sleep(SOURCE_START_STAGGER)
                except Exception as e:
                    print(f"{self.log_prefix} IGNITION ERROR: source {i} -> {e}", flush=True)
            with self.probe_stats_lock:
                self.decode_stats_snapshot_time = time.time()
                self.decode_stats_snapshot_total_frames = self.frame_stats.get('probe_frames', 0)
                self.decode_stats_snapshot_per_source = dict(self.source_probe_frame_counts)
                self.decode_pts_snapshot_per_source = dict(self.source_latest_pts_ns)
            print(f"{self.log_prefix} IGNITION: 所有视频源点火完毕\n", flush=True)
            return False  # 只执行一次

        def start_delayed_ignition():
            ignition_thread = threading.Thread(
                target=delayed_ignition,
                name=f"ignition-gpu{self.gpu_id}-pipe{self.pipeline_id}",
                daemon=True,
            )
            ignition_thread.start()
            return False

        # 启动 20 秒后在后台线程执行批量点火，避免阻塞 GLib 主循环
        GLib.timeout_add_seconds(20, start_delayed_ignition)

        def scheduled_worker_rotation():
            self.restart_requested = True
            self.shutdown_reason = (
                f"planned rotation after {WORKER_MAX_UPTIME_SECONDS / 3600:.1f}h uptime"
            )
            print(
                f"{self.log_prefix} WORKER ROTATION: 到达 {WORKER_MAX_UPTIME_SECONDS / 3600:.1f} 小时运行上限，准备优雅重启",
                flush=True,
            )
            self._stop_event.set()
            if self._main_loop is not None:
                self._main_loop.quit()
            return False

        if WORKER_MAX_UPTIME_SECONDS > 0:
            GLib.timeout_add_seconds(
                max(1, int(WORKER_MAX_UPTIME_SECONDS)),
                scheduled_worker_rotation,
            )

        try:
            loop.run()
        except:
            pass
        finally:
            print(f"{self.log_prefix} Stopping...")
            # ✅ 先唤醒所有worker线程立即停止（这样watchdog不用等15秒才退出）
            self._stop_event.set()
            try:
                self._safe_set_state(self.pipeline, Gst.State.NULL, timeout=5.0)
            except Exception:
                pass
            # 停止帧处理器
            if hasattr(self, 'frame_processor') and self.frame_processor:
                self.frame_processor.stop()
            # 停止上传器
            if hasattr(self, 'uploader') and self.uploader:
                self.uploader.stop()
            # 停止存图器
            if hasattr(self, 'image_saver') and self.image_saver:
                self.image_saver.stop()
            # ✅ watchdog 只认 _stop_event，已无需 watchdog_running 变量
            self._main_loop = None

    def _safe_set_state(self, element, state, timeout=3.0):
        """安全设置GStreamer元素状态，带超时保护防止死锁"""
        if not element:
            return True
        result = [False]
        def task():
            element.set_state(state)
            result[0] = True
        t = threading.Thread(target=task)
        t.daemon = True
        t.start()
        t.join(timeout=timeout)
        return result[0]

    def _collect_source_queue_observation(self):
        """统计 decoder 后、streammux 前本地 queue 的积压时间。"""
        high_queue_sources = []
        queue_sources_window = 0
        avg_queue_time_sum = 0.0
        max_queue_time_seconds = 0.0

        for source_id in range(self.num_sources):
            queue_elem = self.pipeline.get_by_name(f"queue_{source_id}")
            if not queue_elem:
                continue

            try:
                queue_time_ns = queue_elem.get_property("current-level-time")
            except Exception:
                continue

            if queue_time_ns in (None, Gst.CLOCK_TIME_NONE):
                continue

            queue_time_seconds = max(0.0, queue_time_ns / Gst.SECOND)
            queue_sources_window += 1
            avg_queue_time_sum += queue_time_seconds
            max_queue_time_seconds = max(max_queue_time_seconds, queue_time_seconds)

            if queue_time_seconds >= SOURCE_QUEUE_TIME_WARN_SECONDS:
                try:
                    queue_buffers = int(queue_elem.get_property("current-level-buffers"))
                except Exception:
                    queue_buffers = -1
                high_queue_sources.append((source_id, queue_time_seconds, queue_buffers))

        high_queue_sources.sort(key=lambda item: item[1], reverse=True)
        high_queue_sources = high_queue_sources[:DECODE_STATS_TOPN]

        return {
            'queue_sources_window': queue_sources_window,
            'avg_queue_time_seconds': (
                avg_queue_time_sum / queue_sources_window if queue_sources_window > 0 else 0.0
            ),
            'max_queue_time_seconds': max_queue_time_seconds,
            'high_queue_sources': high_queue_sources,
            'queue_time_warn_threshold': SOURCE_QUEUE_TIME_WARN_SECONDS,
        }

    def _build_decode_observation(self, current_time, online_count):
        """基于 probe 计数构建窗口吞吐统计，用于估算当前可承载的解码帧率"""
        with self.probe_stats_lock:
            total_probe_frames = self.frame_stats.get('probe_frames', 0)
            current_per_source = dict(self.source_probe_frame_counts)
            current_pts_per_source = dict(self.source_latest_pts_ns)

        elapsed = max(1e-6, current_time - self.decode_stats_snapshot_time)
        delta_total_frames = max(0, total_probe_frames - self.decode_stats_snapshot_total_frames)
        total_fps = delta_total_frames / elapsed
        self.peak_decode_fps = max(self.peak_decode_fps, total_fps)

        active_sources_window = 0
        low_fps_sources = []
        slow_media_sources = []
        expected_total_fps = online_count * EXPECTED_SOURCE_FPS
        low_fps_threshold = EXPECTED_SOURCE_FPS * DECODE_LOW_FPS_RATIO
        pts_sources_window = 0
        media_rate_sum = 0.0
        min_media_rate = 1.0
        max_lag_growth_seconds = 0.0

        for source_id, current_count in current_per_source.items():
            prev_count = self.decode_stats_snapshot_per_source.get(source_id, 0)
            delta_count = max(0, current_count - prev_count)
            source_fps = delta_count / elapsed
            if delta_count > 0:
                active_sources_window += 1
                if source_fps < low_fps_threshold:
                    low_fps_sources.append((source_id, source_fps))

            prev_pts_ns = self.decode_pts_snapshot_per_source.get(source_id)
            current_pts_ns = current_pts_per_source.get(source_id)
            if (
                prev_pts_ns not in (None, 0, Gst.CLOCK_TIME_NONE)
                and current_pts_ns not in (None, 0, Gst.CLOCK_TIME_NONE)
                and current_pts_ns >= prev_pts_ns
            ):
                delta_pts_seconds = (current_pts_ns - prev_pts_ns) / Gst.SECOND
                media_rate = delta_pts_seconds / elapsed
                lag_growth_seconds = max(0.0, elapsed - delta_pts_seconds)
                pts_sources_window += 1
                media_rate_sum += media_rate
                min_media_rate = min(min_media_rate, media_rate)
                max_lag_growth_seconds = max(max_lag_growth_seconds, lag_growth_seconds)
                if (
                    media_rate < PTS_PROGRESS_WARN_RATIO
                    or lag_growth_seconds >= FRAME_DELAY_WARN_SECONDS
                ):
                    slow_media_sources.append((source_id, media_rate, lag_growth_seconds))

        low_fps_sources.sort(key=lambda item: item[1])
        low_fps_sources = low_fps_sources[:DECODE_STATS_TOPN]
        slow_media_sources.sort(key=lambda item: (item[1], -item[2]))
        slow_media_sources = slow_media_sources[:DECODE_STATS_TOPN]
        equivalent_25fps_sources = total_fps / EXPECTED_SOURCE_FPS if EXPECTED_SOURCE_FPS > 0 else 0.0
        avg_active_source_fps = total_fps / active_sources_window if active_sources_window > 0 else 0.0
        throughput_ratio = (total_fps / expected_total_fps) if expected_total_fps > 0 else 0.0
        avg_media_rate = media_rate_sum / pts_sources_window if pts_sources_window > 0 else 0.0

        self.decode_stats_snapshot_time = current_time
        self.decode_stats_snapshot_total_frames = total_probe_frames
        self.decode_stats_snapshot_per_source = current_per_source
        self.decode_pts_snapshot_per_source = current_pts_per_source

        return {
            'window_seconds': elapsed,
            'total_fps': total_fps,
            'peak_fps': self.peak_decode_fps,
            'equivalent_sources': equivalent_25fps_sources,
            'avg_active_source_fps': avg_active_source_fps,
            'active_sources_window': active_sources_window,
            'expected_total_fps': expected_total_fps,
            'throughput_ratio': throughput_ratio,
            'low_fps_sources': low_fps_sources,
            'low_fps_threshold': low_fps_threshold,
            'pts_sources_window': pts_sources_window,
            'avg_media_rate': avg_media_rate,
            'min_media_rate': min_media_rate if pts_sources_window > 0 else 0.0,
            'max_lag_growth_seconds': max_lag_growth_seconds,
            'slow_media_sources': slow_media_sources,
            'pts_rate_warn_threshold': PTS_PROGRESS_WARN_RATIO,
        }

    # 🛡️ 看门狗worker
    def _watchdog_worker(self):
        """独立看门狗：优先观察 source/probe 心跳，异常时交给复活线程处理"""
        print(f"{self.log_prefix} WATCHDOG 启动")
        stats_counter = 0

        while not self._stop_event.is_set():
            if self._stop_event.wait(timeout=15):
                print(f"{self.log_prefix} WATCHDOG 收到停止信号，退出")
                break

            stats_counter += 1
            current_time = time.time()
            last_probe_batch_time = (
                self.frame_stats.get('last_probe_batch_time')
                or self.frame_stats.get('last_batch_time', current_time)
            )

            if last_probe_batch_time and current_time - last_probe_batch_time > WATCHDOG_PIPELINE_STALL_TIMEOUT:
                probe_idle = current_time - last_probe_batch_time
                print(f"{self.log_prefix} FATAL: {probe_idle:.1f}s 无 probe 输出，判定底层流水线卡死")
                self._stop_event.set()
                try:
                    self._safe_set_state(self.pipeline, Gst.State.NULL, timeout=3.0)
                except Exception:
                    pass
                os._exit(1)

            with self.last_seen_lock:
                source_last_seen_map = dict(self.camera_last_seen)
                source_ever_seen_map = dict(self.camera_ever_seen)
            source_connect_started_map = dict(self.camera_connect_started_at)

            online_count = 0
            offline_count = 0
            recovering_count = 0
            revive_submitted_this_tick = 0

            for source_id, last_seen in list(source_last_seen_map.items()):
                source_ever_seen = source_ever_seen_map.get(source_id, False)
                connect_started_at = source_connect_started_map.get(source_id, 0.0)

                if not connect_started_at:
                    recovering_count += 1
                    continue

                source_idle = max(0.0, current_time - last_seen) if last_seen else (current_time - connect_started_at)

                with self.watchdog_lock:
                    is_resetting = source_id in self.resetting_cameras
                    grace_until = self.reconnect_grace_until.get(source_id, 0.0)

                if is_resetting or current_time < grace_until:
                    recovering_count += 1
                    continue

                timeout_seconds = WATCHDOG_SOURCE_TIMEOUT if source_ever_seen else SOURCE_INITIAL_CONNECT_TIMEOUT
                if source_idle is None or source_idle <= timeout_seconds:
                    if source_ever_seen:
                        online_count += 1
                    else:
                        recovering_count += 1
                    continue
                offline_count += 1

                camera_name = self.camera_id_to_name.get(source_id, f"camera_{source_id}")
                fail_count = self.reconnect_fail_counts.get(source_id, 0)
                if fail_count > 0:
                    cooldown = [60, 120, 300, 600][min(fail_count - 1, 3)]
                    last_time = self.last_reconnect_time.get(source_id, 0)
                    if current_time - last_time < cooldown:
                        continue

                if revive_submitted_this_tick >= WATCHDOG_REVIVE_BATCH_LIMIT:
                    continue

                with self.watchdog_lock:
                    if source_id in self.resetting_cameras:
                        continue
                    self.resetting_cameras.add(source_id)
                    self.reconnect_grace_until[source_id] = current_time + RTSP_REVIVE_VALIDATE_TIMEOUT
                    revive_submitted_this_tick += 1

                print(
                    f"\n{self.log_prefix} WATCHDOG TIMEOUT: {camera_name} (ID:{source_id}) "
                    f"source_idle={source_idle:.1f}s，准备重连"
                )

                try:
                    src_element = self.pipeline.get_by_name(f"src_{source_id}")
                    if not src_element:
                        src_element = self.pipeline.get_by_name(f"uridecode_{source_id}")
                    if src_element and not self._safe_set_state(src_element, Gst.State.NULL, timeout=3.0):
                        print(f"{self.log_prefix} WATCHDOG WARNING: {camera_name} set_state 超时，继续移交复活队列")
                    self.revive_queue.put((source_id, camera_name))
                except Exception as e:
                    print(f"{self.log_prefix} WATCHDOG ERROR: {camera_name} 拔管异常: {e}")
                    with self.watchdog_lock:
                        if source_id in self.resetting_cameras:
                            self.resetting_cameras.remove(source_id)

            if stats_counter >= WATCHDOG_STATS_INTERVAL_TICKS:
                stats_counter = 0
                decode_obs = self._build_decode_observation(current_time, online_count)
                queue_obs = self._collect_source_queue_observation()
                print(
                    f"{self.log_prefix} 状态: 在线={online_count}, 掉线={offline_count}, "
                    f"恢复中={recovering_count}, "
                    f"probe_batches={self.frame_stats.get('probe_batches', 0)}, "
                    f"process_batches={self.frame_stats.get('batch_count', 0)}, "
                    f"queue={self.frame_processor.input_queue.qsize()}"
                )
                print(
                    f"{self.log_prefix} 解码观测: window={decode_obs['window_seconds']:.1f}s, "
                    f"total_fps={decode_obs['total_fps']:.1f}, peak_fps={decode_obs['peak_fps']:.1f}, "
                    f"equivalent_25fps_sources={decode_obs['equivalent_sources']:.1f}, "
                    f"active_sources={decode_obs['active_sources_window']}, "
                    f"avg_active_source_fps={decode_obs['avg_active_source_fps']:.1f}, "
                    f"expected_fps={decode_obs['expected_total_fps']:.1f}, "
                    f"throughput_ratio={decode_obs['throughput_ratio']:.2f}"
                )
                print(
                    f"{self.log_prefix} 本地排队延迟: "
                    f"last_process_queue_delay={self.frame_stats.get('last_process_queue_delay', 0.0):.3f}s"
                )
                print(
                    f"{self.log_prefix} 媒体时间推进(buf_pts): sources={decode_obs['pts_sources_window']}, "
                    f"avg_rate={decode_obs['avg_media_rate']:.2f}x, "
                    f"min_rate={decode_obs['min_media_rate']:.2f}x, "
                    f"max_lag_growth={decode_obs['max_lag_growth_seconds']:.2f}s"
                )
                print(
                    f"{self.log_prefix} Source队列缓存: sources={queue_obs['queue_sources_window']}, "
                    f"avg_queue_time={queue_obs['avg_queue_time_seconds']:.3f}s, "
                    f"max_queue_time={queue_obs['max_queue_time_seconds']:.3f}s"
                )
                if decode_obs['low_fps_sources']:
                    low_fps_summary = "; ".join(
                        f"{self.camera_id_to_name.get(source_id, f'camera_{source_id}')}: {source_fps:.1f}fps"
                        for source_id, source_fps in decode_obs['low_fps_sources']
                    )
                    print(
                        f"{self.log_prefix} 低帧率源(<{decode_obs['low_fps_threshold']:.1f}fps): "
                        f"{low_fps_summary}"
                    )
                if decode_obs['slow_media_sources']:
                    slow_media_summary = "; ".join(
                        f"{self.camera_id_to_name.get(source_id, f'camera_{source_id}')}: "
                        f"rate={media_rate:.2f}x/lag+{lag_growth:.1f}s"
                        for source_id, media_rate, lag_growth in decode_obs['slow_media_sources']
                    )
                    print(
                        f"{self.log_prefix} 媒体推进偏慢源(<{decode_obs['pts_rate_warn_threshold']:.2f}x): "
                        f"{slow_media_summary}"
                    )
                if queue_obs['high_queue_sources']:
                    high_queue_summary = "; ".join(
                        f"{self.camera_id_to_name.get(source_id, f'camera_{source_id}')}: "
                        f"{queue_time:.2f}s/{queue_buffers}buf"
                        for source_id, queue_time, queue_buffers in queue_obs['high_queue_sources']
                    )
                    print(
                        f"{self.log_prefix} Source队列高缓存(>{queue_obs['queue_time_warn_threshold']:.2f}s): "
                        f"{high_queue_summary}"
                    )

    def _refresh_source_uri(self, src_id, cam_name):
        camera_index_code = self.source_camera_index_codes.get(src_id)
        original_uri = self.source_original_uris.get(src_id, "")
        if camera_index_code:
            new_uri = self.preview_client.get_preview_url_with_retry(camera_index_code, cam_name)
            self.source_original_uris[src_id] = new_uri
            return new_uri
        if original_uri and ("hls" in original_uri.lower() or ".m3u8" in original_uri or ".flv" in original_uri):
            return f"{original_uri}?t={int(time.time() * 1000)}"
        return original_uri

    def _revive_worker(self, worker_id=0):
        """单线程串行点火：保护网络服务器不被并发 DDoS"""
        print(f"{self.log_prefix} REVIVE WORKER 已启动")

        while not self._stop_event.is_set():
            try:
                src_id, cam_name = self.revive_queue.get(timeout=1)

                # 🌟【核心缓冲】：每复活一个设备，强行休息 0.1 秒！排队进站！
                time.sleep(0.1)

                try:
                    attempt_started_at = time.time()
                    validate_timeout = REVIVE_VALIDATE_TIMEOUT
                    self.last_reconnect_time[src_id] = attempt_started_at
                    with self.watchdog_lock:
                        self.reconnect_grace_until[src_id] = attempt_started_at + REVIVE_VALIDATE_TIMEOUT

                    # ==========================================
                    # 🚀 简单重连：只重启uridecodebin，queue→streammux静态链接保持不变
                    # ✅ 重要：不要unlink queue→streammux，否则数据通路永久断开
                    # ==========================================
                    uri = self.pipeline.get_by_name(f"uridecode_{src_id}")
                    rtsp_src = self.pipeline.get_by_name(f"src_{src_id}")

                    if uri:
                        # 1. 停止元素（带超时保护）
                        self._safe_set_state(uri, Gst.State.NULL, timeout=3.0)

                        # 2. 给 OS 释放端口的时间
                        time.sleep(1.0)

                        # 3. 更新 URI（添加时间戳强制刷新播放列表）
                        original_uri = self._refresh_source_uri(src_id, cam_name)
                        if original_uri:
                            if "hls" in original_uri.lower() or ".m3u8" in original_uri or ".flv" in original_uri:
                                timestamped_uri = f"{original_uri}?t={int(time.time() * 1000)}"
                                uri.set_property("uri", timestamped_uri)
                            else:
                                uri.set_property("uri", original_uri)
                                print(f"{self.log_prefix} REVIVE: {cam_name} RTSP URI reset")

                        # 4. 重新点火
                        self._safe_set_state(uri, Gst.State.PLAYING, timeout=5.0)
                        if original_uri.startswith("rtsp://"):
                            print(f"{self.log_prefix} REVIVE: {cam_name} RTSP reconnect kicked, waiting for probe")
                            continue

                        print(f"{self.log_prefix} REVIVE: {cam_name} 重连点火完成")
                    elif rtsp_src:
                        self._safe_set_state(rtsp_src, Gst.State.NULL, timeout=3.0)
                        time.sleep(1.0)

                        original_uri = self._refresh_source_uri(src_id, cam_name)
                        if src_id in self.rtsp_codec_states:
                            self.rtsp_codec_states[src_id]["selected"] = None
                        if original_uri:
                            rtsp_src.set_property("location", original_uri)

                        self.camera_connect_started_at[src_id] = time.time()
                        rtsp_src.set_locked_state(False)
                        self._safe_set_state(rtsp_src, Gst.State.PLAYING, timeout=5.0)
                        print(f"{self.log_prefix} REVIVE: {cam_name} RTSP reconnect kicked, waiting for probe")
                        validate_timeout = RTSP_REVIVE_VALIDATE_TIMEOUT
                    else:
                        raise RuntimeError(f"source element not found for camera {src_id}")

                    # 4. 等待较短窗口，确认是否恢复出帧，避免大量掉线时串行队列被拖死
                    reconnect_start = time.time()
                    reconnected_successfully = False
                    consecutive_success = 0
                    while time.time() - reconnect_start < validate_timeout:
                        time.sleep(1)
                        if self.camera_ever_seen.get(src_id, False) and \
                           (time.time() - self.camera_last_seen.get(src_id, 0)) < REVIVE_SUCCESS_IDLE_SECONDS:
                            consecutive_success += 1
                            if consecutive_success >= REVIVE_SUCCESS_REQUIRED_STREAK:
                                reconnected_successfully = True
                                break
                        else:
                            consecutive_success = 0  # 中断则重新计数

                    if reconnected_successfully:
                        # 成功重连后重置失败计数，记录重连时间
                        if src_id in self.reconnect_fail_counts:
                            del self.reconnect_fail_counts[src_id]
                        self.last_reconnect_time[src_id] = time.time()
                        with self.watchdog_lock:
                            self.reconnect_grace_until.pop(src_id, None)
                        # 设置60秒静默期，期间忽略 transient hlsdemux 错误
                        if not hasattr(self, '_reconnect_silence'):
                            self._reconnect_silence = {}
                        self._reconnect_silence[f"{src_id}_reconnect_silence_until"] = time.time() + 60
                        print(f"{self.log_prefix} REVIVE OK: {cam_name} 重连成功，失败计数已清除")
                    else:
                        print(f"{self.log_prefix} REVIVE FAIL: {cam_name} {REVIVE_VALIDATE_TIMEOUT:.0f}秒内无数据，重连失败")
                        self.reconnect_fail_counts[src_id] = self.reconnect_fail_counts.get(src_id, 0) + 1
                        fail_count = self.reconnect_fail_counts[src_id]
                        self.last_reconnect_time[src_id] = time.time()
                        cooldown = [60, 120, 300, 600][min(fail_count - 1, 3)]
                        with self.watchdog_lock:
                            self.reconnect_grace_until.pop(src_id, None)
                        print(f"{self.log_prefix} REVIVE FAIL: {cam_name} 失败{fail_count}次，下次掉线后需等待{cooldown}秒才能重连")

                except Exception as e:
                    print(f"{self.log_prefix} REVIVE ERROR: {cam_name} 复活失败: {e}")
                    # 增加失败计数，用于下次掉线时的退避
                    self.reconnect_fail_counts[src_id] = self.reconnect_fail_counts.get(src_id, 0) + 1
                    fail_count = self.reconnect_fail_counts[src_id]
                    self.last_reconnect_time[src_id] = time.time()
                    cooldown = [60, 120, 300, 600][min(fail_count - 1, 3)]
                    with self.watchdog_lock:
                        self.reconnect_grace_until.pop(src_id, None)
                    print(f"{self.log_prefix} REVIVE FAIL: {cam_name} 失败{fail_count}次，下次掉线后需等待{cooldown}秒才能重连")
                finally:
                    # 解除防重入锁
                    with self.watchdog_lock:
                        if src_id in self.resetting_cameras:
                            self.resetting_cameras.remove(src_id)
                    self.revive_queue.task_done()

            except queue.Empty:
                continue

    def _on_bus_message(self, bus, msg):
        if msg.type == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            err_msg = err.message
            src_name = msg.src.get_name() if msg.src else "unknown"
            match = re.search(r'_(\d+)', src_name)
            source_id = int(match.group(1)) if match else None
            camera_name = self.camera_id_to_name.get(source_id, f"camera_{source_id}") if source_id is not None else src_name
            resource_error = any(
                token in err_msg
                for token in (
                    "Could not read",
                    "Could not write",
                    "Could not open",
                    "Not Found",
                    "enough data",
                    "data stream error",
                )
            )

            if resource_error and source_id is not None:
                with self.watchdog_lock:
                    if source_id in self.resetting_cameras or time.time() < self.reconnect_grace_until.get(source_id, 0.0):
                        return

            if resource_error and source_id is not None:
                current_error_time = time.time()
                silence_key = f"{source_id}_reconnect_silence_until"
                silence_until = getattr(self, '_reconnect_silence', {}).get(silence_key, 0)
                if current_error_time < silence_until:
                    return

                log_interval = (
                    BUS_DATA_STREAM_WARN_LOG_INTERVAL
                    if "data stream error" in err_msg
                    else BUS_RESOURCE_ERROR_LOG_INTERVAL
                )
                last_log_time = self.last_resource_error_log.get(source_id, 0.0)
                if current_error_time - last_log_time >= log_interval:
                    self.last_resource_error_log[source_id] = current_error_time
                    print(f"{self.log_prefix} ERROR: {camera_name} <- {src_name}: {err_msg}", flush=True)

                if source_id in self.reconnect_fail_counts:
                    fail_count = self.reconnect_fail_counts[source_id]
                    cooldown = min(30 * (2 ** fail_count), 300)
                    with self.watchdog_lock:
                        last_fail_time = getattr(self, '_last_fail_time', {}).get(source_id, 0.0)
                        if current_error_time - last_fail_time < cooldown:
                            return
                        self._last_fail_time = getattr(self, '_last_fail_time', {})
                        self._last_fail_time[source_id] = current_error_time

                with self.watchdog_lock:
                    if source_id not in self.resetting_cameras:
                        self.resetting_cameras.add(source_id)
                        self.reconnect_grace_until[source_id] = current_error_time + RTSP_REVIVE_VALIDATE_TIMEOUT
                        self.revive_queue.put((source_id, camera_name))
                        last_requeue_log_time = self.last_resource_requeue_log.get(source_id, 0.0)
                        if current_error_time - last_requeue_log_time >= BUS_RESOURCE_REQUEUE_LOG_INTERVAL:
                            self.last_resource_requeue_log[source_id] = current_error_time
                            print(f"{self.log_prefix} ERROR: queued {camera_name} for async revive", flush=True)
                return

            if "data stream error" in err_msg:
                should_log = True
                if source_id is not None:
                    last_log_time = self.last_bus_warning_log.get(source_id, 0.0)
                    if time.time() - last_log_time < BUS_DATA_STREAM_WARN_LOG_INTERVAL:
                        should_log = False
                    else:
                        self.last_bus_warning_log[source_id] = time.time()
                if should_log:
                    if source_id is not None:
                        print(
                            f"{self.log_prefix} BUS WARN: {camera_name} <- {src_name}: {err_msg}",
                            flush=True,
                        )
                    else:
                        print(f"{self.log_prefix} BUS WARN: 来源 {src_name}: {err_msg}", flush=True)
                return

            if source_id is not None:
                print(
                    f"{self.log_prefix} ERROR: {camera_name} <- {src_name}: {err_msg}",
                    flush=True,
                )
            else:
                print(f"{self.log_prefix} ERROR: 来源 {src_name}: {err_msg}", flush=True)

            if resource_error and source_id is not None:
                silence_key = f"{source_id}_reconnect_silence_until"
                silence_until = getattr(self, '_reconnect_silence', {}).get(silence_key, 0)
                if time.time() < silence_until:
                    print(f"{self.log_prefix} ERROR: {camera_name} 在静默期内，忽略 transient 错误")
                    return

                if source_id in self.reconnect_fail_counts:
                    fail_count = self.reconnect_fail_counts[source_id]
                    cooldown = min(30 * (2 ** fail_count), 300)
                    with self.watchdog_lock:
                        last_fail_time = getattr(self, '_last_fail_time', {}).get(source_id, 0)
                        if time.time() - last_fail_time < cooldown:
                            print(f"{self.log_prefix} ERROR: {camera_name} 仍在冷却中（失败{fail_count}次），等待{cooldown}秒后重连...")
                            return
                        self._last_fail_time = getattr(self, '_last_fail_time', {})
                        self._last_fail_time[source_id] = time.time()

                with self.watchdog_lock:
                    if source_id not in self.resetting_cameras:
                        self.resetting_cameras.add(source_id)
                        self.reconnect_grace_until[source_id] = time.time() + REVIVE_VALIDATE_TIMEOUT
                        self.revive_queue.put((source_id, camera_name))
                        print(f"{self.log_prefix} ERROR: 已将 {camera_name} 移交队列异步复活")

        elif msg.type == Gst.MessageType.EOS:
            src = msg.src
            src_name = src.get_name() if src else "unknown"
            match = re.search(r'_(\d+)', src_name)
            source_id = int(match.group(1)) if match else None
            if source_id is not None:
                camera_name = self.camera_id_to_name.get(source_id, f"camera_{source_id}")
                print(f"{self.log_prefix} EOS: {camera_name} <- {src_name}", flush=True)
            else:
                print(f"{self.log_prefix} EOS: Stream end: {src_name}", flush=True)




# ==============================================================================
# 子进程运行函数 - 每个进程绑定一块GPU
# ==============================================================================
def run_worker(gpu_id, subset_cameras, camera_configs, base_dir, global_thresholds):
    # 1. 戴上眼罩：对当前进程隐藏其他显卡！
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"🚀 [Worker {gpu_id}] 启动，接管 {len(subset_cameras)} 路摄像头...", flush=True)

    # 2. 突破文件描述符限制
    import resource
    resource.setrlimit(resource.RLIMIT_NOFILE, (65535, 65535))

    # 3. 现在才可以安全地初始化 GStreamer 和导入 DeepStream
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import GLib, Gst
    Gst.init(None)

    # 4. 全局延迟注入重型硬件库
    global pyds, cv2, np
    import pyds
    import cv2
    import numpy as np

    # 4. 确保工作进程输出目录存在
    os.makedirs(os.path.join(base_dir, "output"), exist_ok=True)

    # 5. 单 pipeline 运行：当前每个 GPU 只维护一条推理流水线
    pipeline_id = 0
    print(f"[INFO] [Worker {gpu_id}] 单 pipeline 模式启动，接管 {len(subset_cameras)} 路摄像头", flush=True)
    app = DeepStreamApp(
        subset_cameras,
        camera_configs=camera_configs,
        base_dir=base_dir,
        gpu_id=gpu_id,
        pipeline_id=pipeline_id,
        global_thresholds=global_thresholds,
    )
    app.run()
    if getattr(app, 'restart_requested', False):
        print(f"{app.log_prefix} Worker planned rotation complete: {app.shutdown_reason}", flush=True)
        sys.exit(WORKER_ROTATE_EXIT_CODE)


if __name__ == '__main__':
    import json
    import os
    import multiprocessing
    import math
    import time
    import signal

    # 启动Prometheus指标HTTP服务器
    if METRICS_ENABLED:
        try:
            import prometheus_metrics as pm
            threading.Thread(target=pm.start_metrics_server, args=(9090,), daemon=True).start()
        except Exception as e:
            print(f"[WARN] Failed to start Prometheus metrics server: {e}")

    # 1. 必须使用 spawn 模式
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, 'camera_zones_config.json')

    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)

    camera_configs = normalize_camera_configs(config)
    confidence_thresholds = config.get("confidence_thresholds", {})
    preview_client = ArtemisPreviewClient()
    cameras_data = build_cameras_data_from_config(camera_configs, preview_client)

    total_cams = len(cameras_data)
    print(f"[INFO] 总计有效摄像头数: {total_cams}")

    NUM_GPUS = 1
    chunk_size = math.ceil(total_cams / NUM_GPUS)

    process_manager = {}

    def start_worker_process(gpu_id, subset, cam_configs, conf_thresholds):
        p = multiprocessing.Process(target=run_worker, args=(gpu_id, subset, cam_configs, script_dir, conf_thresholds))
        p.daemon = True
        p.start()
        return p

    # 3. 首次点火，启动所有子进程
    for i in range(NUM_GPUS):
        subset = cameras_data[i*chunk_size : (i+1)*chunk_size]
        if not subset:
            continue

        # 启动前清理旧的心跳文件
        heartbeat_file = f"/tmp/gpu{i}_heartbeat"
        if os.path.exists(heartbeat_file):
            os.remove(heartbeat_file)

        p = start_worker_process(i, subset, camera_configs, confidence_thresholds)
        process_manager[i] = {
            'process': p,
            'subset': subset,
            'config': camera_configs,
            'thresholds': confidence_thresholds,
            'last_heartbeat_time': time.time() # 记录上次心跳的时间
        }

    # 4. 开启 7x24 小时守护轮询 (Process Watchdog)
    try:
        print(f"\n[MANAGER] 🛡️ 主进程守护犬已就位，摒弃TXT日志，采用底层文件心跳监控...")
        while True:
            time.sleep(10)  # 每 10 秒巡查一次
            current_time = time.time()

            for gpu_id, info in process_manager.items():
                current_p = info['process']
                heartbeat_file = f"/tmp/gpu{gpu_id}_heartbeat"

                # ==========================================
                # ⚡ 检查 1：子进程是否因 SegFault 自行消亡？
                # ==========================================
                if not current_p.is_alive():
                    exit_code = current_p.exitcode
                    if exit_code == WORKER_ROTATE_EXIT_CODE:
                        print(f"\n[MANAGER 🔄] Worker {gpu_id} 按计划轮换退出，退出码: {exit_code}")
                    else:
                        print(f"\n[MANAGER 🚨] 警告！Worker {gpu_id} 意外死亡或自尽！退出码: {exit_code}")
                    print(f"[MANAGER ⏳] 强制 {MANAGER_RESTART_COOLDOWN:.0f} 秒网络冷却期，释放底层资源...")
                    time.sleep(MANAGER_RESTART_COOLDOWN)

                    new_p = start_worker_process(gpu_id, info['subset'], info['config'], info['thresholds'])
                    process_manager[gpu_id]['process'] = new_p
                    process_manager[gpu_id]['last_heartbeat_time'] = time.time()
                    print(f"[MANAGER ✅] Worker {gpu_id} 重启指令下发成功！")
                    continue  # 🚨 极其关键：处理完死亡重启后，直接跳过本轮循环！防止双胞胎BUG！

                # ==========================================
                # ⚡ 检查 2：子进程活着，但底层是否死锁卡死？
                # ==========================================
                last_beat = info['last_heartbeat_time']

                # 读取子进程写入的独立心跳文件（绕过日志缓冲陷阱）
                try:
                    if os.path.exists(heartbeat_file):
                        with open(heartbeat_file, "r") as f:
                            last_beat = int(f.read().strip())
                            info['last_heartbeat_time'] = last_beat
                except Exception:
                    pass

                idle_time = current_time - last_beat

                # 如果超过 180 秒没有更新心跳文件，判定为无解死锁，执行物理强杀！
                if idle_time > 180:
                    print(f"\n[MANAGER 💀] 发现 GPU {gpu_id} 已 {idle_time:.0f} 秒无心跳！判定为底层死锁！")
                    print(f"[MANAGER 💀] 正在执行系统级强杀 (SIGKILL)...")

                    try:
                        # 用最暴力的操作系统调用杀进程，绝不手软
                        os.kill(current_p.pid, signal.SIGKILL)
                    except Exception as e:
                        print(f"强杀失败: {e}")

                    current_p.join(timeout=3)

                    print(f"[MANAGER ⏳] 强杀完毕！进入 {MANAGER_KILL_RESTART_COOLDOWN:.0f} 秒冷却期等待内核回收显存...")
                    time.sleep(MANAGER_KILL_RESTART_COOLDOWN)

                    new_p = start_worker_process(gpu_id, info['subset'], info['config'], info['thresholds'])
                    process_manager[gpu_id]['process'] = new_p
                    process_manager[gpu_id]['last_heartbeat_time'] = time.time()
                    print(f"[MANAGER ✅] Worker {gpu_id} 浴火重生！")

    except KeyboardInterrupt:
        print("\n[MANAGER] 收到 Ctrl+C 手动退出信号，正在安全关闭所有底层子进程...")
        for gpu_id, info in process_manager.items():
            p = info['process']
            if p.is_alive():
                try:
                    os.kill(p.pid, signal.SIGKILL)
                except: pass
                p.join(timeout=1)
        print("[MANAGER] 所有资源已安全释放，系统退出。")
