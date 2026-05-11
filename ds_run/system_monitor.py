# system_monitor.py - 增强版性能监控
import psutil
import threading
import time
import json
import os
from datetime import datetime

class PerformanceManager:
    def __init__(self, update_interval=2.0):
        self.update_interval = update_interval
        self.running = False
        self.thread = None
        self.metrics = {
            'cpu': {'percent': 0},
            'memory': {'percent': 0, 'used': 0, 'total': 0},
            'disk': {'read_bytes': 0, 'write_bytes': 0},
            'network': {'sent': 0, 'recv': 0},
            'gpu': [],
            'process': {'cpu_percent': 0, 'memory_percent': 0, 'threads': 0, 'open_files': 0},
            'inference': {'total_count': 0, 'avg_time': 0, 'fps': 0},
            'queue': {'yolo': 0, 'vlm': 0}
        }

        # 统计推理性能
        self._inference_times = []
        self._last_inference_count = 0
        self._last_inference_time = time.time()

        # 1. 优先尝试使用 pynvml (nvidia-ml-py) - 官方库，最稳定
        self.gpu_mode = 'none'
        try:
            import pynvml
            pynvml.nvmlInit()
            self.pynvml = pynvml
            self.gpu_mode = 'nvml'
            print("✅ GPU监控模式: NVIDIA NVML (官方驱动)")
        except:
            # 2. 如果失败，尝试使用 GPUtil
            try:
                import GPUtil
                if len(GPUtil.getGPUs()) > 0:
                    self.GPUtil = GPUtil
                    self.gpu_mode = 'gputil'
                    print("✅ GPU监控模式: GPUtil")
            except:
                print("⚠️ 未检测到可用的 GPU 监控库 (需安装 nvidia-ml-py 或 gputil)")

        # 网络IO初始值
        self._last_net_io = None
        self._last_disk_io = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def update_inference_stats(self, batch_size, inference_time):
        """更新推理统计"""
        self._inference_times.append(inference_time)
        if len(self._inference_times) > 100:
            self._inference_times.pop(0)

    def update_queue_stats(self, yolo_queue_size, vlm_queue_size):
        """更新队列大小"""
        self.metrics['queue']['yolo'] = yolo_queue_size
        self.metrics['queue']['vlm'] = vlm_queue_size

    def _get_gpu_info(self):
        gpus = []
        try:
            if self.gpu_mode == 'nvml':
                count = self.pynvml.nvmlDeviceGetCount()
                for i in range(count):
                    handle = self.pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = self.pynvml.nvmlDeviceGetName(handle)
                    if isinstance(name, bytes): name = name.decode('utf-8')

                    util = self.pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
                    temp = self.pynvml.nvmlDeviceGetTemperature(handle, 0)

                    gpus.append({
                        'id': i,
                        'name': name,
                        'load': f"{util.gpu}%",
                        'memory_used': f"{mem.used / 1024**2:.0f}MB",
                        'memory_total': f"{mem.total / 1024**2:.0f}MB",
                        'memory_percent': f"{mem.used / mem.total * 100:.1f}%",
                        'temperature': f"{temp}°C"
                    })

            elif self.gpu_mode == 'gputil':
                for gpu in self.GPUtil.getGPUs():
                    gpus.append({
                        'id': gpu.id,
                        'name': gpu.name,
                        'load': f"{gpu.load * 100:.0f}%",
                        'memory_used': f"{gpu.memoryUsed:.0f}MB",
                        'memory_total': f"{gpu.memoryTotal:.0f}MB",
                        'temperature': f"{gpu.temperature}°C"
                    })
        except Exception as e:
            pass
        return gpus

    def _monitor_loop(self):
        pid = os.getpid()
        proc = psutil.Process(pid)

        while self.running:
            try:
                cpu = psutil.cpu_percent(interval=0.1)
                mem = psutil.virtual_memory()
                disk = psutil.disk_usage('.')
                proc_cpu = proc.cpu_percent()
                proc_mem = proc.memory_percent()

                # 网络IO
                net_io = psutil.net_io_counters()
                if self._last_net_io:
                    net_sent = net_io.bytes_sent - self._last_net_io.bytes_sent
                    net_recv = net_io.bytes_recv - self._last_net_io.bytes_recv
                else:
                    net_sent = net_recv = 0
                self._last_net_io = net_io

                # 磁盘IO
                disk_io = psutil.disk_io_counters()
                if self._last_disk_io:
                    disk_read = disk_io.read_bytes - self._last_disk_io.read_bytes
                    disk_write = disk_io.write_bytes - self._last_disk_io.write_bytes
                else:
                    disk_read = disk_write = 0
                self._last_disk_io = disk_io

                # 推理FPS计算
                now = time.time()
                fps = 0
                if self._inference_times:
                    avg_time = sum(self._inference_times) / len(self._inference_times)
                    if avg_time > 0:
                        fps = 1.0 / avg_time * 64  # 批量大小64

                gpu_data = self._get_gpu_info()

                self.metrics = {
                    'timestamp': datetime.now().strftime('%H:%M:%S'),
                    'cpu': {'percent': f"{cpu}%"},
                    'memory': {'percent': f"{mem.percent}%", 'used': f"{mem.used / 1024**3:.1f}GB", 'total': f"{mem.total / 1024**3:.1f}GB"},
                    'disk': {'read_mb': f"{disk_read / 1024**2:.1f}MB/s", 'write_mb': f"{disk_write / 1024**2:.1f}MB/s"},
                    'network': {'sent_mb': f"{net_sent / 1024**2:.2f}MB/s", 'recv_mb': f"{net_recv / 1024**2:.2f}MB/s"},
                    'process': {
                        'cpu_percent': f"{proc_cpu:.1f}%",
                        'memory_percent': f"{proc_mem:.1f}%",
                        'threads': proc.num_threads(),
                        'open_files': len(proc.open_files()) if hasattr(proc, 'open_files') else 0
                    },
                    'gpu': gpu_data,
                    'inference': {
                        'avg_time_ms': f"{sum(self._inference_times) / len(self._inference_times) * 1000:.1f}ms" if self._inference_times else "0ms",
                        'fps': f"{fps:.1f}"
                    },
                    'queue': self.metrics['queue']
                }

                # 自动保存到文件
                self.save_metrics_to_file()

                time.sleep(self.update_interval)
            except Exception:
                pass

    def save_metrics_to_file(self, filepath='system_metrics.json'):
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.metrics, f, ensure_ascii=False, indent=2)
        except: pass

    def get_summary(self):
        """获取摘要字符串"""
        m = self.metrics
        gpu_str = ""
        if m['gpu']:
            g = m['gpu'][0]
            gpu_str = f" | GPU: {g['memory_used']}/{g['memory_total']} ({g['load']})"

        return (f"[资源] CPU:{m['cpu']['percent']} | "
                f"RAM:{m['memory']['percent']} | "
                f"进程:{m['process']['threads']}线程 | "
                f"队列:YOLO={m['queue']['yolo']} VLM={m['queue']['vlm']} | "
                f"推理:{m['inference']['avg_time_ms']} {m['inference']['fps']}fps"
                + gpu_str)

# 兼容旧代码调用
class SystemMonitor:
    def __init__(self): pass
