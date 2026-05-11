# prometheus_metrics.py - Prometheus 指标暴露模块
from prometheus_client import Counter, Gauge, Histogram, Info, generate_latest, CONTENT_TYPE_LATEST, start_http_server, GCCollector, PROCESS_COLLECTOR, PlatformCollector
import threading
import time
import os

# 注册Python内置 Collector（CPU、内存、GC等）
try:
    GCCollector()
    PlatformCollector()
    PROCESS_COLLECTOR = True
except:
    PROCESS_COLLECTOR = False

# psutil 硬件监控
_psutil_available = False
try:
    import psutil
    _psutil_available = True
except:
    pass

# GPU监控初始化（延迟导入）
_pynvml = None
_gpu_available = False
try:
    import pynvml
    pynvml.nvmlInit()
    _pynvml = pynvml
    _gpu_available = True
except:
    pass

# -----------------------------------------------------------------------------
# 1. Info指标 - 应用信息
# -----------------------------------------------------------------------------
APP_INFO = Info('deepstream_app', 'Application info')
APP_INFO.info({'version': '6.8', 'type': 'multi_camera_stream'})

# -----------------------------------------------------------------------------
# 2. Counter指标 - 累计计数
# -----------------------------------------------------------------------------
FRAMES_PROCESSED = Counter(
    'deepstream_frames_processed_total',
    'Total number of frames processed',
    ['camera', 'status']  # status: success, dropped, error
)

DETECTIONS_TOTAL = Counter(
    'deepstream_detections_total',
    'Total number of detections by class',
    ['camera', 'label']
)

INFERENCE_REQUESTS = Counter(
    'deepstream_inference_requests_total',
    'Total number of inference requests'
)

UPLOAD_REQUESTS = Counter(
    'deepstream_upload_requests_total',
    'Total upload requests',
    ['status']  # success, failed
)

ALARMS_TRIGGERED = Counter(
    'deepstream_alarms_triggered_total',
    'Total alarms triggered',
    ['camera', 'alarm_type']
)

# -----------------------------------------------------------------------------
# 3. Gauge指标 - 当前值
# -----------------------------------------------------------------------------
CAMERA_STREAM_STATE = Gauge(
    'deepstream_camera_stream_state',
    'Camera stream state (1=running, 0=stopped)',
    ['camera']
)

QUEUE_SIZE = Gauge(
    'deepstream_queue_size',
    'Current queue size',
    ['queue_name']  # yolo, vlm, upload, alarm
)

QUEUE_DROPPED = Counter(
    'deepstream_queue_dropped_total',
    'Total number of frames dropped due to queue full',
    ['queue_name']
)

GPU_MEMORY_USED = Gauge(
    'deepstream_gpu_memory_used_bytes',
    'GPU memory used in bytes',
    ['gpu_id']
)

GPU_MEMORY_TOTAL = Gauge(
    'deepstream_gpu_memory_total_bytes',
    'GPU memory total in bytes',
    ['gpu_id']
)

GPU_UTILIZATION = Gauge(
    'deepstream_gpu_utilization_percent',
    'GPU utilization percentage',
    ['gpu_id']
)

PROCESS_THREADS = Gauge(
    'deepstream_process_threads',
    'Number of threads in the process'
)

# ---- 服务器硬件指标 (psutil) ----
SYSTEM_CPU_USAGE = Gauge(
    'system_cpu_usage_percent',
    'System CPU usage percentage'
)

SYSTEM_MEMORY_TOTAL = Gauge(
    'system_memory_total_bytes',
    'Total system memory in bytes'
)

SYSTEM_MEMORY_USED = Gauge(
    'system_memory_used_bytes',
    'Used system memory in bytes'
)

SYSTEM_MEMORY_PERCENT = Gauge(
    'system_memory_usage_percent',
    'System memory usage percentage'
)

SYSTEM_DISK_TOTAL = Gauge(
    'system_disk_total_bytes',
    'Total disk space in bytes'
)

SYSTEM_DISK_USED = Gauge(
    'system_disk_used_bytes',
    'Used disk space in bytes'
)

SYSTEM_DISK_PERCENT = Gauge(
    'system_disk_usage_percent',
    'Disk usage percentage'
)

SYSTEM_NETWORK_SENT = Gauge(
    'system_network_sent_bytes',
    'Total bytes sent'
)

SYSTEM_NETWORK_RECV = Gauge(
    'system_network_recv_bytes',
    'Total bytes received'
)

# ---- Python GC 指标 (改名避免与内置冲突) ----
APP_GC_COLLECTIONS = Counter(
    'app_gc_collections_total',
    'Total number of garbage collections by generation',
    ['generation']  # 0, 1, 2
)

APP_GC_OBJECTS = Gauge(
    'app_gc_objects',
    'Current number of objects in GC by generation',
    ['generation']
)

# -----------------------------------------------------------------------------
# 4. Histogram指标 - 分布统计
# -----------------------------------------------------------------------------
INFERENCE_LATENCY = Histogram(
    'deepstream_inference_latency_seconds',
    'Inference latency in seconds',
    ['camera'],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0)
)

FRAME_PROCESSING_TIME = Histogram(
    'deepstream_frame_processing_seconds',
    'Frame processing time in seconds',
    ['camera'],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
)

UPLOAD_LATENCY = Histogram(
    'deepstream_upload_latency_seconds',
    'Upload latency in seconds',
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
)

# -----------------------------------------------------------------------------
# 5. HTTP服务器管理
# -----------------------------------------------------------------------------
_metrics_server = None
_server_lock = threading.Lock()

def start_metrics_server(port=9090):
    """启动Prometheus指标HTTP服务器"""
    global _metrics_server
    with _server_lock:
        if _metrics_server is None:
            start_http_server(port)
            _metrics_server = port
            print(f"[METRICS] Prometheus metrics server started on port {port}")
    return port

def get_metrics():
    """获取所有指标的当前值"""
    return generate_latest()

def get_content_type():
    """获取Prometheus格式类型"""
    return CONTENT_TYPE_LATEST

def update_gpu_metrics():
    """更新GPU指标"""
    if not _gpu_available or _pynvml is None:
        return
    try:
        count = _pynvml.nvmlDeviceGetCount()
        for i in range(count):
            handle = _pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = _pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = _pynvml.nvmlDeviceGetUtilizationRates(handle)
            GPU_MEMORY_USED.labels(gpu_id=str(i)).set(mem.used)
            GPU_MEMORY_TOTAL.labels(gpu_id=str(i)).set(mem.total)
            GPU_UTILIZATION.labels(gpu_id=str(i)).set(util.gpu)
    except:
        pass

def update_system_metrics():
    """更新服务器硬件指标（CPU、内存、磁盘、网络）"""
    if not _psutil_available:
        return
    try:
        # CPU
        SYSTEM_CPU_USAGE.set(psutil.cpu_percent(interval=0.1))

        # 内存
        mem = psutil.virtual_memory()
        SYSTEM_MEMORY_TOTAL.set(mem.total)
        SYSTEM_MEMORY_USED.set(mem.used)
        SYSTEM_MEMORY_PERCENT.set(mem.percent)

        # 磁盘
        disk = psutil.disk_usage('.')
        SYSTEM_DISK_TOTAL.set(disk.total)
        SYSTEM_DISK_USED.set(disk.used)
        SYSTEM_DISK_PERCENT.set(disk.percent)

        # 网络IO
        net = psutil.net_io_counters()
        SYSTEM_NETWORK_SENT.set(net.bytes_sent)
        SYSTEM_NETWORK_RECV.set(net.bytes_recv)
    except:
        pass

def update_gc_metrics():
    """更新Python GC指标"""
    try:
        import gc
        for generation in range(3):
            counts = gc.get_count()
            APP_GC_OBJECTS.labels(generation=str(generation)).set(counts[generation])
            # 手动触发统计（不实际回收）
            collected = gc.collect(generation)
            APP_GC_COLLECTIONS.labels(generation=str(generation)).inc(collected)
    except:
        pass
