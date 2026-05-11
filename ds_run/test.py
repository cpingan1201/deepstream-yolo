#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一多目标检测系统 v5.7 - 纯PyTorch版本 (test.py)
移除了DeepStream加速，保留所有检测逻辑
"""

import os
import ssl
import logging
logging.getLogger("transformers").setLevel(logging.WARNING)  # 关闭transformers加载日志
# 禁用SSL证书验证（解决EasyOCR下载模型问题）
ssl._create_default_https_context = ssl._create_unverified_context

import cv2
import numpy as np
import json
import time
import threading
from datetime import datetime, timezone, timedelta
import queue
import requests
import base64
import sys
import argparse
import gc
import re
import math
import glob

# 导入性能监控模块
try:
    from system_monitor import PerformanceManager
    PERFORMANCE_MONITOR_AVAILABLE = True
except ImportError:
    PERFORMANCE_MONITOR_AVAILABLE = False
    print("⚠️ 性能监控模块不可用")

# 导入Prometheus监控模块
try:
    from prometheus_metrics import get_prometheus_manager
    PROMETHEUS_AVAILABLE = True
    print("✅ Prometheus监控模块已加载")
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("⚠️ Prometheus监控模块不可用（需安装: pip install prometheus_client psutil pynvml）")

# 导入EasyOCR
try:
    import easyocr
    EASYOCR_AVAILABLE = True
    print("✅ EasyOCR已加载")
except ImportError:
    EASYOCR_AVAILABLE = False
    print("⚠️ EasyOCR不可用，楼层号检测功能将不可用")

# 添加ultralytics路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from ultralytics import YOLO
    from ultralytics.trackers.bot_sort import BOTSORT
    BOT_SORT_AVAILABLE = True
except ImportError:
    BOT_SORT_AVAILABLE = False

# 环境配置
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# 定义北京时区
BEIJING_TZ = timezone(timedelta(hours=8))

def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', '_', str(name))

# ==============================================================================
# 🤖 VLM 智能分析模块
# ==============================================================================

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
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='VLM')
        self.pending_tasks = queue.Queue()
        self.processing_count = 0
        self.completed_count = 0
        self.failed_count = 0
        self.lock = threading.Lock()
        self.prometheus_mgr = None  # 将在UnifiedSystem中设置
        self.initialized = True

    def _semantic_analysis_fix(self, parsed_json, description):
        """语义分析修正-检测描述中的矛盾表述"""
        try:
            behavior_type = parsed_json.get('behavior_type', '').lower()

            # 摔倒检测的语义矛盾模式
            if 'fall' in behavior_type or '摔倒' in description:
                contradiction_patterns = [
                    (['弯腰', '前倾', '捡', '蹲下', '系鞋带', '整理'], ['未摔倒', '没有摔倒', '不是摔倒', '保持平衡', '姿态稳定']),
                    (['正在', '在', '进行'], ['捡东西', '工作', '活动', '移动']),
                    (['站立', '直立', '站着', '双腿支撑', '保持平衡'], None)
                ]

                for pattern_group in contradiction_patterns:
                    if isinstance(pattern_group[0], list) and pattern_group[1]:
                        has_safe_action = any(word in description for word in pattern_group[0])
                        has_stable_desc = any(word in description for word in pattern_group[1])
                        if has_safe_action and has_stable_desc and parsed_json.get('signal'):
                            print(f"[VLM语义修正] 摔倒检测：检测到矛盾表述，强制修正为 False")
                            parsed_json['signal'] = False
                            return

                    elif isinstance(pattern_group[0], list) and pattern_group[1] is None:
                        has_standing = any(word in description for word in pattern_group[0])
                        if has_standing and parsed_json.get('signal'):
                            print(f"[VLM语义修正] 摔倒检测：检测到直立表述，强制修正为 False")
                            parsed_json['signal'] = False
                            return

                pickup_patterns = ['在捡', '捡东西', '弯腰捡', '蹲下捡', '捡拾']
                if any(pattern in description for pattern in pickup_patterns):
                    if parsed_json.get('signal'):
                        print(f"[VLM语义修正] 摔倒检测：检测到捡拾动作表述，强制修正为 False")
                        parsed_json['signal'] = False
                        return

            # 打架检测的语义矛盾模式
            if 'fight' in behavior_type or '打架' in description:
                safe_fight_patterns = [
                    (['拥抱', '握手', '搭肩', '帮助', '扶'], None),
                    (['嬉戏', '打闹', '玩耍', '游戏', '开玩笑'], None),
                    (['运动', '比赛', '锻炼', '训练', '健身'], None),
                    (['没有打架', '未打架', '不是打架', '没有冲突', '没有攻击'], None)
                ]

                for pattern_group in safe_fight_patterns:
                    has_safe_pattern = any(word in description for word in pattern_group[0])
                    if has_safe_pattern and parsed_json.get('signal'):
                        print(f"[VLM语义修正] 打架检测：检测到安全表述，强制修正为 False")
                        parsed_json['signal'] = False
                        return


            # 火灾/烟雾检测的语义矛盾模式 - 只对明确的光源/反光干扰进行修正
            if 'fire' in behavior_type or 'smoke' in behavior_type or '火' in description or '烟' in description:
                # 描述矛盾检测：description说误报但signal=true
                desc_contradiction_patterns = [
                    ['误报', '判断为误报', '因此判断为'],
                    ['无任何烟雾', '无任何火焰', '无烟雾', '无火焰']
                ]
                for pattern_group in desc_contradiction_patterns:
                    if any(word in description for word in pattern_group) and parsed_json.get('signal'):
                        print(f"[VLM语义修正] 火灾/烟雾检测：description与signal矛盾，强制修正为 False")
                        parsed_json['signal'] = False
                        return

                # 光源/反光干扰
                false_alarm_patterns = [
                    ['灯光', '路灯', '车灯', '监控补光', '补光灯', '手电'],
                    ['反光', '反射', '光斑', '光晕', '耀斑', '倒影'],
                    ['不是火灾', '不是明火', '不是烟雾', '没有火灾', '没有烟雾', '未发现']
                ]

                for pattern_group in false_alarm_patterns:
                    if any(word in description for word in pattern_group) and parsed_json.get('signal'):
                        print(f"[VLM语义修正] 火灾/烟雾检测：检测到光源/反光干扰表述，强制修正为 False")
                        parsed_json['signal'] = False
                        return

        except Exception as e:
            print(f"[VLM语义分析失败] {e}")

    def submit_task(self, image_path, detection_type):
        task = {
            'image_path': image_path,
            'detection_type': detection_type,
            'submit_time': time.time()
        }
        self.pending_tasks.put(task)
        self._process_queue()
        return True

    def _process_queue(self):
        while True:
            try:
                if self.processing_count < 2:
                    try:
                        task = self.pending_tasks.get(timeout=0.1)
                        with self.lock: self.processing_count += 1
                        self.executor.submit(self._execute_task, task)
                    except queue.Empty: break
                else: break
            except Exception: break

    def _execute_task(self, task):
        try:
            start_time = time.time()
            analyze_with_vlm(task['image_path'], task['detection_type'])
            duration = time.time() - start_time
            with self.lock:
                self.completed_count += 1

            # 更新Prometheus指标
            if self.prometheus_mgr:
                try:
                    self.prometheus_mgr.record_vlm_analysis_direct(
                        task['detection_type'], 'success', duration
                    )
                except:
                    pass
        except Exception as e:
            print(f"[VLM分析失败] {task['image_path']}: {e}")
            with self.lock:
                self.failed_count += 1
            # 更新Prometheus失败指标
            if self.prometheus_mgr:
                try:
                    self.prometheus_mgr.record_vlm_analysis_direct(
                        task['detection_type'], 'failed'
                    )
                except:
                    pass
        finally:
            with self.lock:
                self.processing_count -= 1
            self._update_prometheus_metrics()
            self._process_queue()

    def get_stats(self):
        with self.lock:
            stats = {
                'pending': self.pending_tasks.qsize(),
                'processing': self.processing_count,
                'completed': self.completed_count,
                'failed': self.failed_count
            }
        # 更新Prometheus队列指标
        self._update_prometheus_metrics()
        return stats

    def _update_prometheus_metrics(self):
        """更新Prometheus VLM队列指标"""
        if self.prometheus_mgr:
            try:
                with self.lock:
                    self.prometheus_mgr.set_vlm_queue_size_direct(
                        self.pending_tasks.qsize(),
                        self.processing_count,
                        self.completed_count,
                        self.failed_count
                    )
            except:
                pass

def get_prompt_vlm(dtype):
    """VLM提示词生成函数"""
    dtype = dtype.lower()

    # ================= 序列行为类检测 =================
    if 'fall' in dtype:
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

    elif 'dangerous' in dtype or 'weapon' in dtype or 'knife' in dtype or '危险物品' in dtype:
        return """# Role
公共安全分析专家
# 重要说明
图片中的【红色方框】是标注框（bounding box），用于标记需要分析的区域。
红色方框本身不是要分析的物体，你只需要分析红色方框【范围内】的实际画面内容。

# Task
判断红色方框范围内的物品是否具有危险性。

# 判断规则（严格遵守，不得自行发挥）
**危险物品（signal=true）**：
- 刀/匕首/斧头/锯子等锐器
- 棍棒/木棍/铁棍/竹竿/钢管/棒球棍/塑料管等任何长条形物体
- 砖头/石块/碎酒瓶等硬物

**安全物品（signal=false）** - 必须100%确认是以下物品：
- 雨伞（含伞柄）
- 拐杖/助行器（医疗辅助用具，明显较细）
- 扫帚/拖把（清洁工具，手柄木质或塑料）
- 看不清物体轮廓、物体太小、被遮挡

**绝对禁止**：不能将"棍棒"、"木棍"、"铁棍"、"竹竿"、"钢管"、"棒状物"、"管状物"判断为非危险物品。

# Output（直接输出JSON，不要其他内容）
{"signal": true/false, "behavior_type": "dangerous_item", "description": "简要描述判断结果"}"""


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
   - 电动车/自行车/纸箱/包裹/垃圾袋/桶/行李箱等 → signal=true
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

    elif 'motor_vehicle' in dtype or '机动车' in dtype or 'car' in dtype:
        return """# Role
机动车检测专家
# 重要说明
图片中的【红色方框】是标注框（bounding box），用于标记需要分析的区域。
红色方框本身不是要分析的物体，你只需要分析红色方框【范围内】的实际画面内容。

# Task
判断红色方框范围内是否为机动车（汽车、货车、卡车等）。

## 判断规则

### 机动车（signal=true）
- 轿车、SUV、面包车：封闭式车身，有车窗
- 货车、卡车：有货箱，车身较高
- 公交车、大巴：车身长，有多个车窗
- 整体结构完整，明显是车辆

### 非机动车（signal=false）
- 自行车：两个轮子，车身细，无发动机
- 电动车、摩托车：有发动机，但应归类为ev_violations
- 其他物品：石头、树枝、影子等误检
- 看不清特征：只看到部分物体 → signal=false

# Output（直接输出JSON，不要其他内容）
{"signal": true/false, "behavior_type": "motor_vehicle_violations", "description": "简要描述判断结果"}"""

    elif 'non_motor' in dtype or '非机动车' in dtype or 'bicycle' in dtype:
        return """# Role
非机动车检测专家
# 重要说明
图片中的【红色方框】是标注框（bounding box），用于标记需要分析的区域。
红色方框本身不是要分析的物体，你只需要分析红色方框【范围内】的实际画面内容。

# Task
判断红色方框范围内是否为非机动车（自行车、电动车、摩托车等）。

## 判断规则

### 非机动车（signal=true）
- 自行车：两个轮子，有车把和车座
- 电动车：有踏板，电瓶明显
- 摩托车：有发动机，车身比自行车粗

### 非非机动车（signal=false）
- 机动车：汽车、货车等封闭式车身
- 其他物品：石头、行李箱、购物车等
- 看不清特征：只看到部分物体 → signal=false

# Output（直接输出JSON，不要其他内容）
{"signal": true/false, "behavior_type": "non_motor_violations", "description": "简要描述判断结果"}"""

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
- 烟雾：必须与火焰同时存在，浓密、持续扩散

**⚠️ 重要规则**：
- 如果红色方框内是**空白区域/纯色背景/不确定**，必须signal=false
- 如果看到"什么都没有"、"空白"、"不确定"、"看不清"，必须signal=false
- 如果description中出现"误报"、"不是"、"无火焰"、"仅烟雾"等否定词，必须signal=false
- **宁缺毋滥**：宁可漏报，不要误报

# Output（直接输出JSON，不要其他内容）
{{"signal": true/false, "behavior_type": "{behavior_type}", "description": "简要描述判断结果"}}"""

    # 香烟检测
    elif 'cigarette' in dtype:
        return """# Role
香烟鉴别专家

# 你的任务
图片中有一个红色方框标注区域。请仔细看这个红色方框【里面的画面内容】，判断是否有人拿着香烟或嘴里叼着香烟。

# 判断步骤（必须按顺序执行）

## 步骤1：看红色方框里面
找到图片中的红色方框，仔细观察红色方框【内部】的画面。

## 步骤2：找香烟
在红色方框内部找：
- 是否有白色/浅色的香烟（通常有滤嘴）
- 香烟可能在手上、嘴边、或桌上

## 步骤3：确认状态
- 手上拿着：手掌握着香烟
- 嘴里叼着：嘴唇含着香烟
- 手上没有拿 → signal=false
- 嘴里没有叼 → signal=false
- 只是类似香烟的物体（笔、筷子、数据线） → signal=false

# 判断标准
**signal=true**：
- 红色方框内有人手拿着香烟
- 红色方框内有人嘴叼着香烟

**signal=false**：
- 红色方框内没有香烟
- 红色方框内只有人，没有香烟
- 类似香烟的物体（笔、筷子、深色条状物）

# 输出格式（直接JSON）
{"signal": true/false, "behavior_type": "cigarette", "description": "描述红色方框内看到了什么"}"""

    # 默认兜底
    else:
        return """# Role
安全监控分析员
# Task
判断画面是否存在极度危险的异常。如果不确定，请返回 False。
# Output
JSON: {"signal": boolean, "behavior_type": "unknown", "description": "简述情况"}"""


# VLM模型缓存（单例模式）
_vlm_model_cache = {
    'model_path': None,
    'model': None,
    'processor': None
}

def _get_vlm_model():
    """获取缓存的VLM模型，如果不存在则加载"""
    import os
    import torch
    from transformers import Qwen3VLProcessor, Qwen3VLForConditionalGeneration

    # 检查本地模型路径
    local_model_paths = [
        "/root/models/Qwen3-VL-4B-Instruct-AWQ-4bit",
        "/root/models/Qwen3-VL-2B-Instruct",
        "/root/models/qwen3_vl_4b",
        "/root/models/Qwen3-VL-4B-Instruct-AWQ-4bit-cpatonn",
        "/root/models/Qwen3-VL-4B-Instruct-GGUF"
    ]

    model_path = None
    for path in local_model_paths:
        if os.path.exists(path):
            model_path = path
            break

    if model_path is None:
        return None, None, None

    # 如果已经加载了模型，直接返回缓存
    if _vlm_model_cache['model'] is not None and _vlm_model_cache['model_path'] == model_path:
        print("[VLM] 使用缓存模型")
        return _vlm_model_cache['model'], _vlm_model_cache['processor'], model_path

    # 否则加载模型
    print(f"[VLM] 加载模型: {model_path}")
    processor = Qwen3VLProcessor.from_pretrained(model_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    # 缓存模型
    _vlm_model_cache['model_path'] = model_path
    _vlm_model_cache['model'] = model
    _vlm_model_cache['processor'] = processor

    print("[VLM] 模型加载完成，已缓存")
    return model, processor, model_path


def analyze_with_vlm(image_path, detection_type):
    """VLM分析核心函数"""
    try:
        import os
        import json
        import base64
        import glob
        from datetime import datetime
        from PIL import Image

        # 使用Ollama API（GPU加速）
        import requests
        model_name = "qwen3-vl:2b-instruct-q4_K_M"
        api_url = "http://localhost:11434/api/generate"

        images_b64 = []
        source_type = "single"
        if os.path.isdir(image_path):
            jpg_files = sorted(glob.glob(os.path.join(image_path, '*.jpg')))
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
            "format": "json",
            "keep_alive": -1,  # 模型永久驻留GPU
            "options": {"temperature": 0.1, "num_ctx": 4096, "num_gpu": 999}
        }

        # 调用Ollama API
        try:
            response = requests.post(api_url, json=payload, timeout=180)
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '')
            else:
                print(f"[VLM请求失败] {response.status_code}，回退本地模型...")
                return None
        except Exception as e:
            print(f"[VLM连接失败] {e}，回退本地模型...")
            return None

        # 后续处理
        if not response_text:
            return None

        source_type = "single" if not os.path.isdir(image_path) else "sequence"

        # 处理响应文本
        parsed_json = None
        try:
            # 提取可能包含在markdown代码块中的JSON
            import re
            json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)

            parsed_json = json.loads(response_text)
            desc = parsed_json.get('description', '')

            # 语义分析修正 - 每个类别单独处理
            behavior_type = parsed_json.get('behavior_type', '').lower()

            # 摔倒检测
            if 'fall' in behavior_type or '摔倒' in desc:
                if any(word in desc for word in ['弯腰', '前倾', '捡', '蹲下', '系鞋带']) and \
                   any(word in desc for word in ['未摔倒', '没有摔倒', '不是摔倒', '保持平衡', '姿态稳定']) and \
                   parsed_json.get('signal'):
                    print(f"[VLM语义修正] 摔倒检测：检测到矛盾表述，强制修正为 False")
                    parsed_json['signal'] = False
                elif any(word in desc for word in ['站立', '直立', '站着', '双腿支撑']) and parsed_json.get('signal'):
                    print(f"[VLM语义修正] 摔倒检测：检测到直立表述，强制修正为 False")
                    parsed_json['signal'] = False
                elif any(pattern in desc for pattern in ['在捡', '捡东西', '弯腰捡', '蹲下捡', '捡拾']) and parsed_json.get('signal'):
                    print(f"[VLM语义修正] 摔倒检测：检测到捡拾动作表述，强制修正为 False")
                    parsed_json['signal'] = False

            # 打架检测
            if 'fight' in behavior_type or '打架' in desc:
                if any(word in desc for word in ['拥抱', '握手', '搭肩', '帮助', '扶', '嬉戏', '打闹', '玩耍', '游戏']) and parsed_json.get('signal'):
                    print(f"[VLM语义修正] 打架检测：检测到安全表述，强制修正为 False")
                    parsed_json['signal'] = False

            # 危险物品检测 - 描述矛盾时强制修正
            if 'dangerous_item' in behavior_type or '危险物品' in desc:
                # 描述说"具有危险性"但又说"误报"是矛盾的
                if parsed_json.get('signal') and any(word in desc for word in ['误报', '判断为误报', '因此判断为']):
                    print(f"[VLM语义修正] 危险物品检测：description与signal矛盾，强制修正为 False")
                    parsed_json['signal'] = False

            # 火灾/烟雾检测 - 只对明确的光源/反光误报进行修正
            if 'fire' in behavior_type or 'smoke' in behavior_type or '火' in desc or '烟' in desc:
                # 描述矛盾检测：description说误报但signal=true
                desc_contradiction_keywords = ['误报', '判断为误报', '因此判断为', '无任何烟雾', '无任何火焰', '无烟雾', '无火焰']
                if any(word in desc for word in desc_contradiction_keywords) and parsed_json.get('signal'):
                    print(f"[VLM语义修正] 火灾/烟雾检测：description与signal矛盾，强制修正为 False")
                    parsed_json['signal'] = False
                # 光源/反光干扰
                false_alarm_keywords = ['灯光', '路灯', '车灯', '反光', '反射', '光斑', '光晕', '不是火灾', '不是明火', '不是烟雾', '没有火灾', '没有烟雾']
                if any(word in desc for word in false_alarm_keywords) and parsed_json.get('signal'):
                    print(f"[VLM语义修正] 火灾/烟雾检测：检测到光源/反光干扰表述，强制修正为 False")
                    parsed_json['signal'] = False
                    
            status_icon = "✅" if parsed_json['signal'] else "❌"
            print(f"[VLM分析] {detection_type}: {status_icon} {desc[:40]}...")

        except:
            print(f"[VLM分析] JSON解析警告: {response_text[:40]}...")
            # 如果解析失败，创建一个默认的响应
            parsed_json = {
                'signal': False,
                'behavior_type': detection_type,
                'description': response_text[:100] + "..." if len(response_text) > 100 else response_text
            }

        try:
            # 根据detection_type和source_type确定JSON保存路径
            if source_type == "sequence":
                json_path = os.path.join(image_path, 'vlm_analysis.json')
            else:
                # 单图检测：根据detection_type映射到正确的output子目录
                dtype = detection_type.upper()
                if 'POWER' in dtype or 'FAILURE' in dtype:
                    sub_dir = 'elevator_power_failures'
                elif 'FLOOR' in dtype or 'STUCK' in dtype:
                    sub_dir = 'elevator_floor_stuck'
                elif 'FIRE' in dtype:
                    sub_dir = 'fire_detections'
                elif 'SMOKE' in dtype:
                    sub_dir = 'smoke_detections'
                elif 'CIGARETTE' in dtype:
                    sub_dir = 'cigarette_detections'
                elif 'DANGEROUS' in dtype:
                    sub_dir = 'dangerous_item_detections'
                elif 'FALL' in dtype or 'FIGHT' in dtype or 'JUMP' in dtype:
                    sub_dir = 'behavior_anomalies'
                elif 'OBJECT' in dtype:
                    sub_dir = 'object_detections'
                elif 'EV' in dtype or '电动车' in detection_type:
                    sub_dir = 'ev_violations'
                elif 'MOTOR' in dtype:
                    sub_dir = 'motor_vehicle_violations'
                elif 'NON' in dtype or 'NON-MOTOR' in detection_type:
                    sub_dir = 'non_motor_violations'
                elif 'PET' in dtype or 'ANIMAL' in dtype or '宠物' in detection_type:
                    sub_dir = 'pet_detections'
                else:
                    sub_dir = 'unknown'

                # 从image_path提取文件名，保存到output/sub_dir/
                filename = os.path.basename(image_path)
                json_path = f"output/{sub_dir}/{filename.replace('.jpg', '_vlm_analysis.json')}"

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

# ==============================================================================
# 🛠️ 基础设施
# ==============================================================================

class RealTimeUploader:
    def __init__(self, camera_configs=None):
        # 新接口配置
        self.base_url = "http://umod.me:6969/api/"
        self.file_upload_url = f"{self.base_url}common/upload"
        self.notification_url = f"{self.base_url}notification/task/emergency"

        # 认证头 - 使用 X-App-Id 和 X-App-Secret
        self.app_id = "wuye_app_001"
        self.app_secret = "wuye_secret_123456"
        self.auth_header = {
            "X-App-Id": self.app_id,
            "X-App-Secret": self.app_secret
        }

        # HLS流地址基础URL
        self.hls_base_url = "http://umod.me:6969/hls"
        # 自动加载摄像头配置
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
        self.upload_queue = queue.Queue()
        self.running = False
        self.upload_thread = None

    def start(self):
        self.running = True
        self.upload_thread = threading.Thread(target=self._upload_worker, daemon=True)
        self.upload_thread.start()

    def stop(self):
        self.running = False
        if self.upload_thread: self.upload_thread.join(timeout=1)

    def get_video_stream_url(self, camera_name):
        """生成视频流地址"""
        # 格式: http://umod.me:6969/hls/cam13/index.m3u8
        cam_num = camera_name.replace('camera_', '').lstrip('0') or '1'
        return f"http://umod.me:6969/hls/cam{cam_num}/index.m3u8"

    def upload_detection(self, file_path, camera_name, detection_type, camera_ip="", wait_vlm=False, local_vlm_path=""):
        """上传本地图片文件"""
        try:
            if not os.path.exists(file_path):
                print(f"[上传错误] 文件不存在: {file_path}")
                return
            stream_url = self.get_video_stream_url(camera_name)
            data = {
                'temp_path': file_path,
                'camera_name': camera_name,
                'detection_type': detection_type,
                'is_sequence': False,
                'camera_ip': stream_url,
                'wait_vlm': wait_vlm,
                'local_vlm_path': local_vlm_path
            }
            self.upload_queue.put(data)
        except Exception as e:
            print(f"[上传错误] 准备数据失败: {e}")

    def upload_sequence(self, seq_folder, camera_name, detection_type, camera_ip=""):
        """上传序列帧文件夹"""
        if not seq_folder or not os.path.isdir(seq_folder):
            print(f"[上传错误] 序列文件夹不存在: {seq_folder}")
            return
        try:
            stream_url = self.get_video_stream_url(camera_name)
            jpg_files = sorted(glob.glob(os.path.join(seq_folder, '*.jpg')))
            if not jpg_files:
                print(f"[上传错误] 序列文件夹内无jpg文件: {seq_folder}")
                return
            print(f"[上传序列] {camera_name}: 准备上传 {len(jpg_files)} 张图片")
            for i, file_path in enumerate(jpg_files):
                data = {
                    'temp_path': file_path,
                    'camera_name': camera_name,
                    'detection_type': detection_type,
                    'is_sequence': True,
                    'frame_index': i + 1,
                    'total_frames': len(jpg_files),
                    'camera_ip': stream_url,
                    'wait_vlm': True,
                    'local_vlm_path': seq_folder
                }
                self.upload_queue.put(data)
        except Exception as e:
            print(f"[上传错误] 准备数据失败: {e}")

    def _upload_worker(self):
        """Background worker: wait for VLM, then upload and notify"""
        print(f"[UploadWorker] 启动，running={self.running}")
        while self.running:
            try:
                item = self.upload_queue.get(timeout=1)
                print(f"[UploadWorker] 收到任务: camera={item.get('camera_name')}, type={item.get('detection_type')}")
                file_path = item.get('temp_path')
                camera_name = item['camera_name']
                detection_type = item['detection_type']
                camera_ip = item.get('camera_ip', '')
                wait_vlm = item.get('wait_vlm', False)
                vlm_timeout = item.get('vlm_timeout', 300)  # 增加到5分钟
                local_vlm_path = item.get('local_vlm_path', '')
                is_sequence = item.get('is_sequence', False)

                # 需要VLM验证的检测类型（精确匹配）
                vlm_required_types = ['fire', 'smoke', 'cigarette', 'fall', 'fight', 'dangerous_item', 'power_failure', 'floor_stuck', 'uncivilized_pet', 'object_detections', 'ev_violations', 'motor_vehicle_violations', 'non_motor_violations']

                # 检测类型标准化映射（folder -> 标准类型）
                type_normalize_map = {
                    'fire_detections': 'fire',
                    'smoke_detections': 'smoke',
                    'cigarette_detections': 'cigarette',
                    'fall_detections': 'fall',
                    'fight_detections': 'fight',
                    'dangerous_item_detections': 'dangerous_item',
                    'behavior_anomalies': 'dangerous_item',  # 危险品、摔倒、打架等
                    'object_detections': 'object_detections',
                    'elevator_power_failures': 'power_failure',
                    'elevator_floor_stuck': 'floor_stuck',
                    'pet_detections': 'uncivilized_pet',
                    'motor_vehicle_violations': 'motor_vehicle_violations',
                    'non_motor_violations': 'non_motor_violations',
                }

                # 标准化检测类型
                normalized_type = type_normalize_map.get(detection_type, detection_type)

                # 确定是否需要等待VLM
                need_wait_vlm = wait_vlm or (normalized_type in vlm_required_types and local_vlm_path)

                # 计算VLM JSON路径
                vlm_json_path = ""
                if need_wait_vlm and local_vlm_path:
                    if os.path.isdir(local_vlm_path):
                        vlm_json_path = os.path.join(local_vlm_path, 'vlm_analysis.json')
                    else:
                        # 根据detection_type或folder确定正确的output目录
                        # detection_type可能是behavior_anomalies，需要特殊处理
                        dtype_upper = detection_type.upper()
                        folder_name = ''

                        # 危险行为相关的folder（但危险品、香烟是独立的folder）
                        if 'behavior_anomalies' in detection_type.lower():
                            folder_name = 'behavior_anomalies'
                        elif 'FALL' in dtype_upper or 'FIGHT' in dtype_upper:
                            folder_name = 'behavior_anomalies'
                        elif 'FIRE' in dtype_upper:
                            folder_name = 'fire_detections'
                        elif 'SMOKE' in dtype_upper:
                            folder_name = 'smoke_detections'
                        elif 'CIGARETTE' in dtype_upper:
                            folder_name = 'cigarette_detections'
                        elif 'DANGEROUS' in dtype_upper:
                            folder_name = 'dangerous_item_detections'
                        elif 'OBJECT' in dtype_upper:
                            folder_name = 'object_detections'
                        elif 'EV' in dtype_upper or '电动车' in detection_type:
                            folder_name = 'ev_violations'
                        elif 'NON' in dtype_upper or '非机动车' in detection_type:
                            folder_name = 'non_motor_violations'
                        elif 'MOTOR' in dtype_upper or '机动车' in detection_type:
                            folder_name = 'motor_vehicle_violations'
                        else:
                            folder_name = detection_type  # 尝试直接使用

                        # 从local_vlm_path提取文件名
                        filename = os.path.basename(local_vlm_path)
                        # 在output目录读取JSON
                        vlm_json_path = f"output/{folder_name}/{filename.replace('.jpg', '_vlm_analysis.json')}"

                    # 等待VLM分析完成（最多120秒）
                    max_wait = time.time() + vlm_timeout
                    while not os.path.exists(vlm_json_path) and time.time() < max_wait:
                        time.sleep(0.5)

                    # 超时则继续发送通知（作为误报/未确认）
                    if not os.path.exists(vlm_json_path):
                        print(f"[{camera_name}] ⚠️ VLM分析超时 ({vlm_timeout}s)，发送误报通知")
                        vlm_analysis = "VLM分析超时，无法确认"
                        vlm_status = "未确认"
                        vlm_signal = False  # 视为误报
                    else:
                        print(f"[{camera_name}] ✅ VLM分析完成: {os.path.basename(vlm_json_path)}")

                # 上传文件
                print(f"[UploadWorker] 开始上传: {file_path}")
                upload_start = time.time()
                server_path = self._upload_file(file_path, camera_name, detection_type)
                upload_duration = time.time() - upload_start
                print(f"[UploadWorker] 上传结果: {server_path}")

                # 更新Prometheus上传指标
                try:
                    from prometheus_metrics import upload_success_total, upload_failure_total, upload_duration_seconds, upload_queue_size
                    if server_path:
                        upload_success_total.labels(camera_name=camera_name, detection_type=detection_type).inc()
                        upload_duration_seconds.observe(upload_duration)
                    else:
                        upload_failure_total.labels(camera_name=camera_name, detection_type=detection_type, error_type='upload_failed').inc()
                    upload_queue_size.set(self.upload_queue.qsize())
                except:
                    pass

                if not server_path:
                    print(f"[上传失败] 文件上传失败: {file_path}")
                    self.upload_queue.task_done()
                    continue

                # 序列模式下只发送第一张图片的通知
                frame_index = item.get('frame_index', 1)
                if is_sequence and frame_index > 1:
                    print(f"[UploadWorker] 序列帧{frame_index}跳过通知: {os.path.basename(file_path)}")
                    self.upload_queue.task_done()
                    continue

                # 发送通知（传递已计算的 vlm_json_path）
                print(f"[UploadWorker] 开始发送通知: camera={camera_name}, type={detection_type}")
                self._send_notification(camera_name, detection_type, server_path, camera_ip, vlm_json_path=vlm_json_path)
                self.upload_queue.task_done()
                print(f"[UploadWorker] 任务完成")
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[上传错误] {e}")

    def _upload_file(self, file_path, camera_name='', detection_type=''):
        """上传本地文件到服务器"""
        try:
            # 使用绝对路径
            abs_path = os.path.abspath(file_path)
            print(f"[上传调试] 绝对路径: {abs_path}")
            print(f"[上传调试] 文件存在: {os.path.exists(abs_path)}")
            print(f"[上传调试] 文件大小: {os.path.getsize(abs_path) if os.path.exists(abs_path) else 0}")

            if not os.path.exists(abs_path):
                return None
            filename = os.path.basename(abs_path)
            with open(abs_path, 'rb') as f:
                files = {'file': (filename, f, 'image/jpeg')}
                response = requests.post(self.file_upload_url, files=files, headers=self.auth_header, timeout=30)
            print(f"[上传调试] 响应状态: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"[上传调试] 响应内容: {result}")
                # 新接口返回格式: {"code":0, "url": "http://umod.me:6969/wm-uploads/..."}
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

    def _send_notification(self, camera_name, detection_type, file_path, camera_ip="", vlm_json_path=""):
        """发送通知 - 只在VLM判定为真告警时才发送"""
        try:
            cam_key = camera_name.lower()
            cam_config = self.camera_configs.get(cam_key, {})
            display_name = cam_config.get('display_name', camera_name)

            # 调试日志
            print(f"[通知调试] camera_name={camera_name}, cam_key={cam_key}, display_name={display_name}")
            print(f"[通知调试] available keys: {list(self.camera_configs.keys())[:10]}...")

            type_map = {
                'power_failure': ('电梯停电', f'{display_name} 电梯发生停电故障', 'A'),
                'floor_stuck': ('电梯故障', f'{display_name} 电梯楼层号长时间无变化', 'A'),
                'fire': ('火灾检测', f'{display_name} 检测到火焰', 'A'),
                'smoke': ('烟雾检测', f'{display_name} 检测到烟雾', 'A'),
                'cigarette': ('吸烟检测', f'{display_name} 检测到吸烟行为', 'A'),
                'fall': ('摔倒检测', f'{display_name} 检测到人员摔倒', 'B'),
                'fight': ('打架检测', f'{display_name} 检测到打架行为', 'B'),
                'dangerous_item': ('危险行为', f'{display_name} 检测到危险行为', 'B'),
                'object_detections': ('电梯异物', f'{display_name} 电梯检测到地面异物', 'C'),
                'motor_vehicle_violations': ('机动车违停', f'{display_name} 存在机动车违停', 'C'),
                'non_motor_violations': ('非机动车违停', f'{display_name} 存在非机动车违停', 'C'),
                'ev_violations': ('电动车违停', f'{display_name} 存在电动车违停', 'C'),
                'ev_alert': ('电动车检测', f'{display_name} 检测到电动车', 'A'),
                'car_alert': ('轿车检测', f'{display_name} 检测到轿车', 'A'),
                'uncivilized_pet': ('不文明养宠', f'{display_name} 检测到不文明养宠行为', 'C'),
            }

            # 检测类型标准化映射（folder -> 标准类型）
            type_normalize_map = {
                'fire_detections': 'fire',
                'smoke_detections': 'smoke',
                'cigarette_detections': 'cigarette',
                'fall_detections': 'fall',
                'fight_detections': 'fight',
                'dangerous_item_detections': 'dangerous_item',
                'behavior_anomalies': 'dangerous_item',  # 危险品、摔倒、打架都用这个
                'object_detections': 'object_detections',
                'elevator_power_failures': 'power_failure',
                'elevator_floor_stuck': 'floor_stuck',
                'pet_detections': 'uncivilized_pet',
                'motor_vehicle_violations': 'motor_vehicle_violations',
                'non_motor_violations': 'non_motor_violations',
            }

            # 标准化检测类型
            normalized_type = type_normalize_map.get(detection_type, detection_type)

            # 从文件名提取具体告警类型（fall/fight/elevator_jump）
            filename = os.path.basename(file_path)
            if '_FALL_' in filename.upper():
                normalized_type = 'fall'
            elif '_FIGHT_' in filename.upper():
                normalized_type = 'fight'
            elif '_DANGEROUS_ITEM_' in filename.upper():
                normalized_type = 'dangerous_item'
            elif '_OBJECT_' in filename.upper():
                normalized_type = 'object_detections'

            type_info = type_map.get(normalized_type, ('安全告警', f'{display_name} 发生异常事件', 'C'))
            title_suffix, content_base, risk_level = type_info
            timestamp = datetime.now(BEIJING_TZ).strftime('%Y-%m-%d %H:%M:%S')
            content = f"{timestamp} {display_name} 发生{title_suffix}。"

            # 读取VLM结果
            vlm_analysis = ""
            vlm_result = ""
            vlm_status = ""
            vlm_signal = None

            # 需要VLM验证的检测类型
            vlm_required_types = ['fire', 'smoke', 'cigarette', 'fall', 'fight', 'dangerous_item', 'power_failure', 'floor_stuck', 'uncivilized_pet', 'object_detections', 'ev_violations', 'motor_vehicle_violations', 'non_motor_violations']

            if normalized_type in vlm_required_types and vlm_json_path and os.path.exists(vlm_json_path):
                try:
                    with open(vlm_json_path, 'r', encoding='utf-8') as f:
                        vlm_data = json.load(f)
                        # 提取description，可能需要解析嵌套的JSON字符串
                        desc = vlm_data.get('parsed_result', {}).get('description', '')
                        if isinstance(desc, str):
                            try:
                                desc_obj = json.loads(desc)
                                if isinstance(desc_obj, dict):
                                    desc = desc_obj.get('description', desc)
                            except:
                                pass
                        vlm_analysis = desc
                        # signal在parsed_result嵌套对象中
                        vlm_signal = vlm_data.get('parsed_result', {}).get('signal', None)
                        vlm_status = "警告" if vlm_signal is True else ("误报" if vlm_signal is False else "未完成")
                    print(f"[VLM读取] signal={vlm_signal}, {os.path.basename(vlm_json_path)}")
                except Exception as e:
                    print(f"[VLM读取失败] {e}")
                    vlm_analysis = ""
                    vlm_signal = None

            # 核心逻辑：需要VLM验证的类型，signal=None时跳过，signal=true或false都发送通知
            need_vlm_check = normalized_type in vlm_required_types
            if need_vlm_check and vlm_signal is None:
                print(f"[{camera_name}] ⏭️ VLM分析未完成(signal=None)，跳过通知: {title_suffix}")
                return

            # 误报也发送通知，但调整通知内容
            if vlm_signal is False:
                title_suffix = "误报核实"
                content = f"{timestamp} {display_name} {type_info[0]}经AI核实为误报。"
                print(f"[{camera_name}] 📢 发送误报通知: {title_suffix}")

            # 构建通知payload - 新格式
            related_type_map = {
                'fire': 'fire_detection',
                'smoke': 'smoke_detection',
                'cigarette': 'cigarette_detection',
                'fall': 'fall_detection',
                'fight': 'fight_detection',
                'dangerous_item': 'dangerous_item_detection',
                'power_failure': 'power_failure',
                'floor_stuck': 'floor_stuck',
                'object_detections': 'object_detection',
                'motor_vehicle_violations': 'motor_vehicle_violation',
                'non_motor_violations': 'non_motor_violation',
                'ev_violations': 'ev_violation',
                'uncivilized_pet': 'pet_detection',
            }

            payload = {
                "type": "emergency_alert",
                "notice_type": "emergency_alert",
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
            response = requests.post(self.notification_url, json=payload, headers=self.auth_header, timeout=30)
            if response.status_code == 200:
                result = response.json()
                print(f"[通知调试] 响应内容: {result}")
                # 成功：code == 0 且 msg == '操作成功' 或 message 包含成功
                success = (result.get('code') == 0 and result.get('msg') == '操作成功') or \
                          '成功' in str(result.get('data', {}).get('message', '')) or \
                          'queued' in str(result.get('message', ''))
                if success:
                    print(f"[通知发送成功] {title_suffix} -> {camera_name}")
                else:
                    print(f"[通知发送失败] code={result.get('code')}, message={result.get('message')}")
            else:
                print(f"[通知发送失败] {response.status_code}: {response.text[:100]}")
        except Exception as e:
            print(f"[发送通知异常] {e}")

class RTSPStreamLoader:
    def __init__(self, url, name):
        self.url = url
        self.name = name
        self.frame = None
        self.ret = False
        self.stopped = False
        self.connected = False
        self.lock = threading.Lock()
        self.cap = None
        self.fail_count = 0
        self.reconnect_delay = 120  # 初始重连间隔（秒）
        self.max_delay = 600       # 最大重连间隔（10分钟）
        self.last_reconnect_time = 0
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while not self.stopped:
            if not self.connected:
                now = time.time()
                # 指数退避重连
                if now - self.last_reconnect_time < self.reconnect_delay:
                    time.sleep(1)
                    continue

                try:
                    if self.cap:
                        self.cap.release()
                    self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    if self.cap.isOpened():
                        self.connected = True
                        self.reconnect_delay = 120  # 重置间隔
                        print(f"✅ [连接成功] {self.name}")
                    else:
                        self.reconnect_delay = min(self.reconnect_delay * 1.5, self.max_delay)
                        print(f"⚠️ [连接失败] {self.name} - {self.reconnect_delay}秒后重试")
                except Exception as e:
                    self.reconnect_delay = min(self.reconnect_delay * 1.5, self.max_delay)
                    print(f"⚠️ [连接异常] {self.name} - {e}, {self.reconnect_delay}秒后重试")

                self.last_reconnect_time = time.time()
                continue

            # 关键修改：只取最新帧，清空缓冲区
            # grab() 非常快，只抓取不解码
            for _ in range(int(self.cap.get(cv2.CAP_PROP_BUFFERSIZE)) if self.cap.get(cv2.CAP_PROP_BUFFERSIZE) > 0 else 1):
                self.cap.grab()
            ret, frame = self.cap.retrieve()

            if ret and frame is not None:
                with self.lock:
                    self.frame = frame.copy()
                    self.ret = True
                self.fail_count = 0
            else:
                self.fail_count += 1
                if self.fail_count > 5:
                    self.connected = False
                    self.last_reconnect_time = time.time()  # 重连间隔从这里开始计算
                    if self.cap:
                        self.cap.release()
                        self.cap = None
                    print(f"⚠️ [流中断] {self.name} - {self.reconnect_delay}秒后重连...")
            time.sleep(0.01)

    def read(self):
        with self.lock: return self.ret, self.frame.copy() if self.frame is not None else None

class CameraStatusManager:
    def __init__(self):
        self.status = {}
        self.lock = threading.Lock()
    def update_status(self, name, status):
        with self.lock: self.status[name] = {'status': status, 'time': time.time()}
    def get_summary(self):
        with self.lock:
            return len(self.status), sum(1 for v in self.status.values() if v['status'] == 'connected')


# ==============================================================================
# 🚀 异步任务管理器 (OCR/异物检测解耦)
# ==============================================================================
class AsyncTaskManager:
    """异步任务管理器：处理OCR和异物检测，不阻塞主线程"""
    def __init__(self, system_instance):
        self.system = system_instance
        self.queue = queue.Queue(maxsize=100)  # 队列有界，防止内存爆
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

        # 模型懒加载
        self.ocr_reader = None
        self.yolo_world = None

    def submit(self, task_type, frame, cam_name, **kwargs):
        if self.queue.full():
            return  # 队列满则丢弃
        self.queue.put({
            'type': task_type,
            'frame': frame.copy(),
            'cam': cam_name,
            'kwargs': kwargs
        })

    def _worker(self):
        print("✅ 异步后台线程启动 (OCR/异物检测)")

        # 懒加载 EasyOCR
        if EASYOCR_AVAILABLE:
            import easyocr
            # 使用 EasyOCR-1.7.2/easyocr/model/ 目录下的模型
            os.environ['EASYOCR_MODULE_PATH'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'EasyOCR-1.7.2', 'easyocr')
            self.ocr_reader = easyocr.Reader(['en'], gpu=True)
            print("✅ EasyOCR 加载完成")

        while self.running:
            try:
                task = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                t_type = task['type']
                frame = task['frame']
                cam = task['cam']

                if t_type == 'ocr':
                    self._do_ocr(frame, cam)
                elif t_type == 'object':
                    self._do_object(frame, cam, task['kwargs'].get('person_count', 0))
            except Exception as e:
                print(f"[异步任务异常] {e}")

    def _do_ocr(self, frame, cam):
        """OCR楼层检测 - 异步执行"""
        try:
            cam_status = self.system.elevator_status.camera_status.get(cam, {})
            display_area = cam_status.get('display_area', {})
            if not display_area:
                return

            # 使用系统的 OCR 方法（传入 reader）
            if self.ocr_reader is None:
                return

            # 复用系统的 OCR 逻辑，但传入异步的 reader
            x1, y1, x2, y2 = display_area['x1'], display_area['y1'], display_area['x2'], display_area['y2']
            rotate_angle = display_area.get('angle', 0)

            display_region = frame[y1:y2, x1:x2]
            if display_region.size == 0:
                return

            if rotate_angle and rotate_angle != 0:
                h, w = display_region.shape[:2]
                center = (w / 2, h / 2)
                M = cv2.getRotationMatrix2D(center, rotate_angle, 1.0)
                display_region = cv2.warpAffine(display_region, M, (w, h))

            h, w = display_region.shape[:2]
            enlarged = cv2.resize(display_region, (w*8, h*8), interpolation=cv2.INTER_CUBIC)

            results = self.ocr_reader.readtext(enlarged)
            if not results:
                return

            best_result, best_confidence = None, 0.0
            for (bbox, text, confidence) in results:
                floor_number = self.system._extract_floor_number(text)
                if floor_number is not None and confidence > best_confidence:
                    best_result = floor_number
                    best_confidence = confidence

            if best_result is not None:
                self.system.elevator_status.update_floor_detection(cam, best_result, best_confidence, time.time())
        except Exception as e:
            print(f"[OCR错误] {cam}: {e}")

    def _do_object(self, frame, cam, person_count):
        """异物检测 - 异步执行"""
        try:
            if person_count > 0:
                return  # 有人不检测异物

            if self.yolo_world is None:
                from ultralytics import YOLO
                self.yolo_world = YOLO('yolov8m-worldv2.pt')
                self.yolo_world.set_classes(['person', 'suitcase', 'bag', 'box', 'object'])
                print("✅ YOLO-World 加载完成")

            # 缩小图片加速
            small_frame = cv2.resize(frame, (640, 640))
            results = self.yolo_world.predict(small_frame, verbose=False, classes=[2, 3, 4, 5], conf=0.25)

            if results and len(results[0].boxes) > 0:
                # 检测到异物
                self.system.detect_object_async(self.yolo_world, frame, cam, person_count)
        except Exception as e:
            print(f"[异物检测错误] {cam}: {e}")

# ==============================================================================
# 🧠 检测逻辑
# ==============================================================================

class FallVideoBuffer:
    """Fall detection buffer - optimized to reduce false alarms"""
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
        """Geometry check: shoulders must be below hips to confirm fall"""
        shoulder_y = (pose[5][1] + pose[6][1]) / 2
        hip_y = (pose[11][1] + pose[12][1]) / 2
        return shoulder_y > hip_y  # 肩膀Y > 臀部Y 表示肩膀在臀部下方

    def _check_height_drop_strict(self):
        """严格版高度下降检测 - 下降必须超过40%"""
        if len(self.history) < 20:
            return True

        recent_heights = [h['height'] for h in self.history[-5:] if h['height'] is not None]
        older_heights = [h['height'] for h in self.history[-30:-5] if h['height'] is not None]

        if not recent_heights or not older_heights:
            return True

        avg_recent = sum(recent_heights) / len(recent_heights)
        avg_older = sum(older_heights) / len(older_heights)

        # 宽松阈值：下降超过25%（从40%降低）
        return avg_older > 0 and (avg_older - avg_recent) / avg_older > 0.25

    def check_fall(self, w, h):
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

        # ========== 宽松版摔倒检测 ==========
        # 降低要求：
        # 1. 脊柱角度 > 60°（从70°降低）
        # 2. 身高下降 > 25%（从40%降低）
        # 3. 角度变化率在 10-50°/s 之间（从15降低）
        # 4. 几何验证：肩膀低于臀部

        # 条件1：角度门槛降低
        if curr_angle <= 60:
            self.state = 'standing'
            self.fall_start_time = None
            return False

        # 条件2：身高显著下降
        if not self._check_height_drop_strict():
            self.state = 'standing'
            return False

        # 条件3：变化率范围扩大（宽松）
        if angle_change_rate < 10 or angle_change_rate > 60:
            self.state = 'standing'
            return False

        # 条件4：几何验证 - 肩膀必须低于臀部
        if not self._check_geometry_fall(pose):
            self.state = 'standing'
            return False

        # ========== 状态机 ==========
        if self.state == 'standing':
            self.state = 'falling'
            self.fall_start_time = now
            print(f"[摔倒检测] ID:{self.tid} 检测到潜在摔倒信号")

        elif self.state == 'falling':
            duration = now - self.fall_start_time
            # 宽松：确认时间降到1秒
            if duration >= 1.0:
                self.state = 'fallen'
                print(f"[摔倒检测] ID:{self.tid} 确认摔倒！持续{duration:.1f}秒")
                return True
            # 宽松：如果在0.8秒内恢复站立，重置
            if curr_angle < 50:
                self.state = 'standing'
                self.fall_start_time = None

        elif self.state == 'fallen':
            pass

        return False

    def _check_height_drop(self):
        if len(self.history) < 10:
            return True
        recent_heights = [h['height'] for h in self.history[-10:] if h['height'] is not None]
        older_heights = [h['height'] for h in self.history[-30:-10] if h['height'] is not None]
        if not recent_heights or not older_heights:
            return True
        avg_recent = sum(recent_heights) / len(recent_heights)
        avg_older = sum(older_heights) / len(older_heights)
        if avg_older > 0 and (avg_older - avg_recent) / avg_older > self.height_drop_ratio:
            return True
        return False

    def get_sequence(self):
        return [(d['frame'], f"fall_{self.tid}_{i:03d}.jpg") for i, d in enumerate(self.history[-30:])]

class FightBuffer:
    """Fight detection buffer - optimized to reduce false alarms"""
    def __init__(self, tid, min_frames_required=5, interaction_threshold=40.0):
        self.tid = tid
        self.max_len = 60
        self.history = []
        self.confirmed = False
        self.last_time = time.time()
        self.state = 'normal'
        self.interaction_start_time = None
        self.min_frames_required = min_frames_required
        self.interaction_threshold = interaction_threshold  # 宽松：默认40像素（从50降到40）
        self.prev_poses = {}  # 存储上一帧姿态用于计算运动量

    def update(self, pose, other_poses, frame, fnum):
        """更新状态
        pose: 当前人员姿态关键点
        other_poses: 其他人员姿态列表
        """
        self.last_time = time.time()

        self.history.append({
            'pose': pose,
            'other_poses': other_poses,
            'frame': frame.copy(),
            'time': time.time()
        })
        if len(self.history) > self.max_len:
            self.history.pop(0)

        # 保存当前姿态作为下一帧的参考
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
        """Calculate upper body motion - only hands, arms, torso"""
        if prev_pose is None:
            return 0
        upper_body_indices = [5, 6, 7, 8, 11, 12]  # 肩膀、手、躯干
        motion = sum(
            abs(pose[i][0] - prev_pose[i][0]) + abs(pose[i][1] - prev_pose[i][1])
            for i in upper_body_indices
            if pose[i][0] > 0 and prev_pose[i][0] > 0
        )
        return motion / len([i for i in upper_body_indices if pose[i][0] > 0 and prev_pose[i][0] > 0])

    def _detect_mutual_high_motion(self):
        """检测双方高运动 - 必须双方都在快速运动"""
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
                # 计算双方运动量
                motion_self = self._calc_motion(pose, self.history[i-1]['pose'] if i > 0 else None)
                motion_other = self._calc_motion(op, self.history[i-1].get('other_poses', [None])[0] if i > 0 else None)

                # 宽松：双方运动量都>50像素/帧就算（从80降低）
                if motion_self > 50 and motion_other > 50:
                    mutual_high_motion_frames += 1
                    break

        return mutual_high_motion_frames

    def _detect_close_proximity_strict(self):
        """宽松版近距离检测 - 放宽到<50像素"""
        if len(self.history) < 8:
            return False

        recent = self.history[-8:]
        close_frames = 0

        for f in recent:
            pose = f['pose']
            other_poses = f.get('other_poses', [])

            for op in other_poses:
                dist = self._calculate_distance(pose, op)
                if dist is not None and dist < 50:  # 宽松：放宽到50像素（从30提高到50）
                    close_frames += 1
                    break

        # 宽松：60%的帧满足即可（从80%降低）
        return close_frames >= 5

    def check_fight(self):
        """宽松版打架检测：
        1. distance < 50 pixels（从30放宽）
        2. both motion > 50 pixels/frame（从80降低）
        3. duration > 1.5 seconds（从3秒降低）
        """
        if len(self.history) < 12:
            return False

        # 条件1：距离必须非常近
        if not self._detect_close_proximity_strict():
            self.state = 'normal'
            return False

        # 条件2：双方都必须有高运动
        mutual_motion = self._detect_mutual_high_motion()
        if mutual_motion < 4:  # 宽松：8帧里至少4帧满足双方高运动（从6降低）
            self.state = 'normal'
            return False

        # 状态机
        if self.state == 'normal':
            self.state = 'approaching'
            self.interaction_start_time = time.time()
            print(f"[打架检测] ID:{self.tid} 检测到潜在交互")

        elif self.state == 'approaching':
            duration = time.time() - self.interaction_start_time

            # 宽松：确认时间降到1.5秒
            if duration > 1.5:
                self.state = 'fighting'
                print(f"[打架检测] ID:{self.tid} 确认打架！持续{duration:.1f}秒")
                return True

            # 宽松：1秒内没有持续满足条件则重置
            if duration > 1.0 and mutual_motion < 3:
                self.state = 'normal'
                self.interaction_start_time = None

        elif self.state == 'fighting':
            return True

        return False

    def get_sequence(self):
        return [(d['frame'], f"fight_{self.tid}_{i:03d}.jpg") for i, d in enumerate(self.history[-30:])]

# 危险品类别列表
DANGEROUS_ITEM_CLASSES = ['knife', 'stick', 'rod', 'broken_bottle', 'brick', 'weapon', 'scissors', 'axe', 'hammer']

class DangerousItemBuffer:
    """危险物品检测缓冲区 - 检测人手上是否持有危险物品"""
    def __init__(self, tid, item_type=None, duration_threshold=0.5):
        self.tid = tid  # 关联的人员ID
        self.max_len = 30
        self.history = []
        self.confirmed = False
        self.last_time = time.time()
        self.state = 'normal'  # normal → detected → confirmed
        self.detection_start_time = None
        self.item_type = item_type  # 危险品类型
        self.duration_threshold = duration_threshold  # 0.5秒持续检测

    def update(self, item_detected, item_bbox, person_pose, frame, fnum):
        """更新检测状态

        Args:
            item_detected: 是否检测到危险品
            item_bbox: 危险品边界框 [x1, y1, x2, y2]
            person_pose: 人员姿态关键点
            frame: 当前帧图像
            fnum: 帧号
        """
        self.last_time = time.time()

        self.history.append({
            'item_detected': item_detected,
            'item_bbox': item_bbox,
            'person_pose': person_pose,
            'frame': frame.copy(),
            'time': time.time()
        })
        if len(self.history) > self.max_len:
            self.history.pop(0)

    def _is_item_in_hand(self, item_bbox, pose):
        """判断物品是否在人手上（基于关键点距离）"""
        if item_bbox is None or pose is None:
            return False

        # 获取物品中心点
        item_center = np.array([
            (item_bbox[0] + item_bbox[2]) / 2,
            (item_bbox[1] + item_bbox[3]) / 2
        ])

        # 关键点索引：左手=9, 右手=10
        # YOLO-Pose格式：[0-nose, 1-left_eye, 2-right_eye, 3-left_ear, 4-right_ear,
        #                5-left_shoulder, 6-right_shoulder, 7-left_elbow, 8-right_elbow,
        #                9-left_wrist, 10-right_wrist, 11-left_hip, 12-right_hip,
        #                13-left_knee, 14-right_knee, 15-left_ankle, 16-right_ankle]
        hand_indices = [9, 10]  # 左手腕和右手腕

        for idx in hand_indices:
            if idx < len(pose) and pose[idx][0] > 0 and pose[idx][1] > 0:
                hand_pos = np.array([pose[idx][0], pose[idx][1]])
                distance = np.linalg.norm(item_center - hand_pos)
                # 物品中心在手部60像素范围内认为手持
                if distance < 60:
                    return True

        return False

    def check_dangerous_item(self):
        """检查是否持续检测到危险物品"""
        if len(self.history) < 3:
            return False

        now = time.time()

        # 检查最近几帧是否有检测到危险品且在手上
        recent_detections = 0
        for f in self.history[-6:]:  # 检查最近6帧（约0.5-1秒）
            if f['item_detected'] and self._is_item_in_hand(f['item_bbox'], f['person_pose']):
                recent_detections += 1

        # 至少50%的帧满足条件
        if recent_detections < 3:
            self.state = 'normal'
            self.detection_start_time = None
            return False

        # 状态机
        if self.state == 'normal':
            self.state = 'detected'
            self.detection_start_time = now
            print(f"[危险物品检测] ID:{self.tid} 检测到疑似手持危险物品")

        elif self.state == 'detected':
            duration = now - self.detection_start_time
            # 持续0.5秒确认
            if duration >= self.duration_threshold:
                self.state = 'confirmed'
                print(f"[危险物品检测] ID:{self.tid} 确认手持危险物品！持续{duration:.1f}秒")
                return True
            # 如果1秒后仍不确认，重置
            if duration > 1.0:
                self.state = 'normal'
                self.detection_start_time = None

        elif self.state == 'confirmed':
            return True

        return False

    def get_sequence(self):
        return [(d['frame'], f"dangerous_item_{self.tid}_{i:03d}.jpg") for i, d in enumerate(self.history[-15:])]

class VehicleTracker:
    def __init__(self, tid, cls_name, start_time, cam_name, violation_time_threshold=30.0, stationary_threshold=50):
        self.tid = tid
        self.start_time = start_time
        self.last_pos = None
        self.static_cnt = 0
        self.violation_saved = False
        self.last_seen = time.time()
        self.violation_time_threshold = violation_time_threshold
        self.stationary_threshold = stationary_threshold

    def update(self, bbox, now, is_ev_mode=False, in_zone=False):
        self.last_seen = now
        if is_ev_mode: return True
        if not in_zone:
            self.static_cnt = 0
            self.start_time = now
            return False
        curr = np.array([(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2])
        if self.last_pos is not None:
            if np.linalg.norm(curr - self.last_pos) < self.stationary_threshold: self.static_cnt += 1
            else:
                self.static_cnt = 0
                self.start_time = now
        self.last_pos = curr
        return (now - self.start_time > self.violation_time_threshold) and (self.static_cnt > 10)

class FireSmokeDetector:
    def __init__(self, duration_threshold=1.0):
        self.recs = {}
        self.duration_threshold = duration_threshold

    def process(self, label, cam, conf):
        key = f"{cam}_{label}"
        now = time.time()
        if key not in self.recs: self.recs[key] = {'start': now, 'last': now, 'confirmed': False}
        self.recs[key]['last'] = now
        if now - self.recs[key]['start'] > self.duration_threshold and not self.recs[key]['confirmed']:
            self.recs[key]['confirmed'] = True
            return True
        return False

    def cleanup(self):
        now = time.time()
        self.recs = {k:v for k,v in self.recs.items() if now - v['last'] < 5}

class ElevatorStatusManager:
    """电梯状态管理器 - 基于检测次数"""
    def __init__(self):
        self.camera_status = {}

    def init_camera(self, camera_name, display_area):
        if camera_name not in self.camera_status:
            self.camera_status[camera_name] = {
                'display_area': display_area,
                'last_floor_number': None,
                'people_count': 0,
                'success_count': 0,      # 连续检测到相同楼层号的次数
                'fail_count': 0,         # 连续检测失败的次数
                'power_failure_reported': False,
                'floor_stuck_reported': False
            }

    def update_person_count(self, camera_name, count):
        if camera_name not in self.camera_status:
            return
        status = self.camera_status[camera_name]
        prev_count = status.get('people_count', 0)
        status['people_count'] = count

        # 有人进入时，重置检测次数
        if prev_count == 0 and count > 0:
            status['success_count'] = 0
            status['fail_count'] = 0
            status['power_failure_reported'] = False
            print(f"[电梯状态] {camera_name}: 有人进入，重置检测次数")
        # 无人时，清空状态
        elif prev_count > 0 and count == 0:
            status['last_floor_number'] = None
            status['success_count'] = 0
            status['fail_count'] = 0
            status['power_failure_reported'] = False
            status['floor_stuck_reported'] = False
            print(f"[电梯状态] {camera_name}: 无人，重置状态")

    def update_floor_detection(self, camera_name, floor_number, confidence, current_time):
        if camera_name not in self.camera_status:
            return None

        status = self.camera_status[camera_name]

        # 电梯里没人，重置所有状态
        if status['people_count'] == 0:
            if status['last_floor_number'] is not None or status['success_count'] > 0 or status['fail_count'] > 0:
                status['last_floor_number'] = None
                status['success_count'] = 0
                status['fail_count'] = 0
                status['floor_stuck_reported'] = False
                status['power_failure_reported'] = False
                print(f"[电梯状态] {camera_name}: 无人，重置状态")
            return None

        # OCR成功识别楼层号
        if floor_number is not None and confidence > 0.1:
            status['fail_count'] = 0  # 重置失败计数

            # 楼层号变化，重置所有状态
            if status['last_floor_number'] != floor_number:
                status['last_floor_number'] = floor_number
                status['success_count'] = 1
                status['floor_stuck_reported'] = False
                status['power_failure_reported'] = False
                print(f"[楼层检测] {camera_name}: 楼层号={floor_number}，故障状态已重置")
            else:
                # 楼层号相同，连续成功检测次数+1
                status['success_count'] += 1
                # 连续25次（约25秒）检测到相同楼层号 = 故障
                if status['success_count'] > 25 and not status['floor_stuck_reported']:
                    status['floor_stuck_reported'] = True
                    print(f"[楼层故障] {camera_name}: 楼层号{floor_number}连续{status['success_count']}次无变化")
                    return 'floor_stuck'

            return None

        # OCR识别失败（floor_number == None）
        status['success_count'] = 0  # 失败时重置成功计数
        status['fail_count'] += 1
        status['last_floor_number'] = None
        status['floor_stuck_reported'] = False

        # 电梯有人 + 连续25次（约25秒）检测不到楼层号 = 停电
        if status['fail_count'] > 25 and not status['power_failure_reported']:
            status['power_failure_reported'] = True
            print(f"[停电检测] {camera_name}: 有人但连续{status['fail_count']}次未识别到楼层号，判断为停电")
            return 'power_failure'

        return None

class BatchInferenceEngine:
    """批处理推理引擎 - 纯PyTorch版本"""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_path):
        if hasattr(self, 'initialized'):
            return

        print("=" * 60)
        print("🚀 初始化推理引擎 (PyTorch 纯本地模式)")
        print("=" * 60)

        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO(model_path)
            import torch
            if torch.cuda.is_available():
                self.yolo_model.to('cuda')
                self.mode = "PyTorch GPU"
            else:
                self.mode = "PyTorch CPU"
            print(f"✅ 推理引擎初始化成功 (模式: {self.mode})")
        except Exception as e:
            print(f"⚠️ 推理引擎初始化失败: {e}")
            self.mode = "PyTorch (fallback)"

        self.q_det = queue.Queue()
        self.q_pose = queue.Queue()
        self.pose_model = None
        self.prometheus_mgr = None

        # 添加队列计数器，用于准确跟踪队列大小
        self.q_det_counter = 0
        self.q_det_lock = threading.Lock()

        threading.Thread(target=self._run, args=(self.q_det, self.yolo_model, 640), daemon=True).start()

        self.initialized = True

    def _run(self, q, model, sz):
        batch = []
        last = time.time()
        while True:
            # 先把队列里现有的帧都取出来
            while True:
                try:
                    item = q.get(timeout=0.001)
                    batch.append(item)
                except queue.Empty:
                    break

            # 如果有帧且满足推理条件，则批量推理
            if batch and (len(batch) >= 64 or time.time() - last > 0.05):
                import time as time_module
                start_time = time_module.time()
                frames = [x['img'] for x in batch]
                batch_size = len(frames)

                # 在推理开始时设置批量大小
                if self.prometheus_mgr:
                    try:
                        self.prometheus_mgr.set_inference_batch_size_direct(batch_size)
                    except:
                        pass

                res = model.predict(frames, verbose=False, half=True, imgsz=sz, conf=0.25)

                end_time = time_module.time()
                inference_time = end_time - start_time

                for req, r in zip(batch, res):
                    req['res']['val'] = r
                    req['evt'].set()

                # 更新计数器
                with self.q_det_lock:
                    self.q_det_counter = max(0, self.q_det_counter - len(batch))

                remaining_items = self.q_det_counter
                # print(f"✅ 推理完成: {batch_size} 张, 剩余: {remaining_items} 张, 用时: {inference_time:.3f}s")

                # 更新推理性能统计（如果有performance_mgr）
                if hasattr(self, 'performance_mgr') and self.performance_mgr:
                    self.performance_mgr.update_inference_stats(batch_size, inference_time)

                # 更新Prometheus指标
                if self.prometheus_mgr:
                    try:
                        # 记录队列大小（推理完成后清空批量大小）
                        self.prometheus_mgr.set_inference_batch_size_direct(0)  # 推理完成，清空批量大小
                        self.prometheus_mgr.set_inference_queue_size_direct(remaining_items)

                        # 记录推理耗时（为批次中的每个摄像头记录）
                        # 计算每个摄像头的平均推理时间
                        avg_inference_time = inference_time / batch_size if batch_size > 0 else 0

                        # 统计批次中的摄像头分布
                        camera_counts = {}
                        for item in batch:
                            cam = item.get('camera_name')
                            if cam:
                                camera_counts[cam] = camera_counts.get(cam, 0) + 1

                        # 为每个摄像头记录推理指标
                        for cam, count in camera_counts.items():
                            # 记录推理总次数
                            for _ in range(count):
                                self.prometheus_mgr.record_inference(cam, avg_inference_time)

                    except Exception as e:
                        print(f"[Prometheus] 更新推理指标失败: {e}")

                batch = []
                last = time.time()

            time.sleep(0.001)  # 避免CPU空转

    def infer(self, img, is_pose=False, camera_name=None):
        """推理方法
        Args:
            img: 输入图像
            is_pose: 是否进行姿态检测
            camera_name: 摄像头名称，用于Prometheus监控
        """
        if is_pose:
            try:
                if self.pose_model is None:
                    from ultralytics import YOLO
                    self.pose_model = YOLO('yolo11n-pose.pt')
                result = self.pose_model.predict(img, verbose=False, imgsz=320, conf=0.25)
                return result[0] if result else None
            except Exception as e:
                print(f"[Pose检测失败] {e}")
                return None

        # 普通检测
        evt = threading.Event()
        res = {'val': None}
        with self.q_det_lock:
            self.q_det.put({'img': img, 'evt': evt, 'res': res, 'camera_name': camera_name})
            self.q_det_counter += 1
        if evt.wait(5):
            with self.q_det_lock:
                if self.q_det_counter > 0:
                    self.q_det_counter -= 1
            return res['val']
        return None

# ==============================================================================
# 🎮 主系统逻辑
# ==============================================================================

class UnifiedSystem:
    def __init__(self, config_file='camera_zones_config.json', model_path='best.pt'):
        self.config = json.load(open(config_file, 'r', encoding='utf-8')) if os.path.exists(config_file) else {'cameras':{}, 'detection_settings':{}}

        settings = self.config.get('detection_settings', {})
        self.violation_time_threshold = settings.get('violation_time_threshold', 180.0)
        self.stationary_threshold = settings.get('stationary_threshold', 50)
        self.fire_smoke_duration = settings.get('fire_smoke_duration', 1.0)

        ct = settings.get('confidence_thresholds', {})
        default_thresh = ct.get('default', 0.3)
        cls_list = ['person', 'bicycle', 'car', 'motorcycle', 'fire', 'smoke', 'animal', 'cigarette', 'stick']
        self.conf_thresholds = {k: ct.get(k, default_thresh) for k in cls_list}

        fs = settings.get('fall_detection', {})
        self.min_frames_required = fs.get('min_frames_required', 5)  # 宽松：从8降到5
        self.fall_angle_threshold = fs.get('fall_angle_threshold', 55.0)  # 宽松：从65降到55度
        self.angle_change_threshold = fs.get('angle_change_threshold', 10.0)  # 宽松：从25降到10度/秒
        self.fallen_duration_required = fs.get('fallen_duration_required', 1.0)  # 宽松：从2秒降到1秒
        self.height_drop_ratio = fs.get('height_drop_ratio', 0.20)  # 宽松：从0.25降到0.20

        self.camera_fall_configs = {}
        for k, v in self.config.get('cameras', {}).items():
            fc = v.get('fall_detection', {})
            self.camera_fall_configs[k] = {
                'baseline_angle': fc.get('baseline_angle', 0.0),
                'fall_threshold': fc.get('fall_threshold', 45.0)
            }

        self.zone_detection_enabled = settings.get('zone_detection_enabled', True)
        self.batch = BatchInferenceEngine(model_path)

        # VLM使用Ollama API，无需预加载本地模型
        self.vlm_queue = VLMQueueManager()
        self.async_manager = AsyncTaskManager(self)  # 异步任务管理器

        self.performance_mgr = None
        if PERFORMANCE_MONITOR_AVAILABLE:
            self.performance_mgr = PerformanceManager(update_interval=2.0)
            self.performance_mgr.start()

        # Prometheus监控初始化
        self.prometheus_mgr = None
        if PROMETHEUS_AVAILABLE:
            try:
                self.prometheus_mgr = get_prometheus_manager(port=8000)
                self.prometheus_mgr.start()
            except Exception as e:
                print(f"⚠️ Prometheus启动失败: {e}")

            # 启动活跃摄像头监控线程
            if self.prometheus_mgr:
                threading.Thread(target=self._update_active_cameras_metric, daemon=True).start()

                # 设置VLM队列的Prometheus管理器
                self.vlm_queue.prometheus_mgr = self.prometheus_mgr

                # 设置批处理推理引擎的Prometheus管理器
                self.batch.prometheus_mgr = self.prometheus_mgr

        self.trackers = {}
        self.buffers = {}
        self.fight_buffers = {}  # 打架检测缓冲区
        self.dangerous_item_buffers = {}  # 危险物品检测缓冲区
        self.active_violations = {}
        self.fire_det = FireSmokeDetector(self.fire_smoke_duration)
        self.uploader = RealTimeUploader(self.config.get('cameras', {}))
        self.yolo_world = None  # YOLO-World 模型（危险品检测用）
        self.status_mgr = CameraStatusManager()
        self.stats_lock = threading.Lock()
        self.total_stats = {'motor_vio': 0, 'non_motor_vio': 0, 'fall': 0, 'fire': 0, 'smoke': 0, 'cigarette': 0, 'ev': 0, 'fight': 0, 'dangerous_item': 0, 'object': 0, 'pet': 0}
        self.live_counts = {}

        # 电梯状态管理器
        self.elevator_status = ElevatorStatusManager()

        # EasyOCR初始化
        self.ocr_reader = None
        if EASYOCR_AVAILABLE:
            try:
                # 使用 EasyOCR-1.7.2/easyocr/model/ 目录下的模型
                os.environ['EASYOCR_MODULE_PATH'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'EasyOCR-1.7.2', 'easyocr')
                self.ocr_reader = easyocr.Reader(['en'], gpu=True)
                print("✅ EasyOCR初始化成功 (GPU版本)")
            except Exception as e:
                print(f"❌ EasyOCR初始化失败: {e}")

        # 初始化有显示屏配置的摄像头
        for cam_name, cam_config in self.config.get('cameras', {}).items():
            display_area = cam_config.get('display_area')
            if display_area and display_area.get('enabled', False):
                self.elevator_status.init_camera(cam_name, display_area)
                print(f"✅ 初始化电梯监控: {cam_name}")

        self.object_last_save_time = {}
        self.vehicle_alert_interval = 180  # 同一目标告警间隔（秒）
        self.vehicle_alert_last_time = {}  # 上次告警时间 {key: timestamp}
        self.dangerous_item_alert_interval = 30  # 危险品同一ID告警间隔（秒）
        self.dangerous_item_alert_last_time = {}  # 上次告警时间 {f"{camera}_{tid}": timestamp}
        self.non_motor_counted = set()  # 已统计的非机动车ID {key}
        self.motor_counted = set()  # 已统计的机动车ID {key}

        for d in ['motor_vehicle_violations', 'non_motor_violations', 'behavior_anomalies',
                  'fire_detections', 'smoke_detections', 'ev_violations',
                  'elevator_power_failures', 'elevator_floor_stuck', 'object_detections',
                  'pet_detections']:
            os.makedirs(f"output/{d}", exist_ok=True)
        os.makedirs("dataset", exist_ok=True)

    def _is_elevator_camera(self, camera_name):
        """Check if camera is elevator camera (range camera_49 ~ camera_96)"""
        match = re.match(r'camera_(\d+)', camera_name)
        if match:
            num = int(match.group(1))
            return 49 <= num <= 96
        return False

    def _get_elevator_detection_zone(self, camera_name):
        """获取电梯检测区域，返回 (x1, y1, x2, y2) 或 None"""
        cam_config = self.config.get('cameras', {}).get(camera_name, {})
        zone = cam_config.get('elevator_detection_zone')
        if not zone:
            return None
        # 支持多边形格式(points)和矩形格式(x1,y1,x2,y2)
        if 'points' in zone:
            # 多边形转矩形
            pts = zone['points']
            if len(pts) >= 3:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                return (min(xs), min(ys), max(xs), max(ys))
        elif 'x1' in zone:
            return (zone['x1'], zone['y1'], zone['x2'], zone['y2'])
        return None

    def _is_bbox_in_zone(self, bbox, zone):
        """检查检测框中心是否在区域内"""
        if zone is None:
            return True  # 没有配置区域则全图检测
        x1, y1, x2, y2 = zone
        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        return x1 <= cx <= x2 and y1 <= cy <= y2

    def _record_detection_prometheus(self, camera_name, detection_type, severity='C'):
        """记录检测事件到Prometheus"""
        if self.prometheus_mgr:
            try:
                self.prometheus_mgr.record_detection(camera_name, detection_type, severity)
            except Exception as e:
                pass

    def _update_active_cameras_metric(self):
        """定期更新活跃摄像头数量指标"""
        import time
        while True:
            try:
                if self.prometheus_mgr and hasattr(self, 'status_mgr'):
                    total_cameras, active_cameras = self.status_mgr.get_summary()
                    self.prometheus_mgr.set_active_cameras(active_cameras)
            except Exception as e:
                pass
            time.sleep(10)  # 每10秒更新一次

    def detect_object(self, frame, cam, person_count):
        """
        Object detection: only detect when no person in frame and within elevator_detection_zone
        :param person_count: YOLO raw person count from main loop
        """
        current_time = time.time()

        # 【硬闸】直接使用YOLO原始人数
        if person_count > 0:
            return False  # 当前帧有人，直接终止，不检测异物

        # 获取电梯检测区域
        zone = self._get_elevator_detection_zone(cam)
        h, w = frame.shape[:2]

        # 如果没有配置区域，跳过检测
        if zone is None:
            return False

        # 裁剪到检测区域
        x1, y1, x2, y2 = zone
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return False

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return False

        # 初始化YOLO-World模型（只检测异物，不检测人）
        if not hasattr(self, '_yolo_world_object'):
            from ultralytics import YOLO
            self._yolo_world_object = YOLO('yolov8m-worldv2.pt')
            # 注意：person 只由 best.pt 判断，YOLO-World 只检测异物
            self._yolo_world_object_classes = [
                "garbage", "trash", "rubbish",
                "chair", "stool", "seat",
                "bottle", "plastic bottle",
                "box", "carton", "package",
                "bag", "backpack", "handbag",
                "luggage", "suitcase",
                "umbrella",
                "stick", "rod",
                "tool",
                "cup", "coffee cup", "drink cup",
                "takeout box", "food container", "food packaging",
                "plastic bag",
                "newspaper", "magazine", "paper cup",
                "cigarette", "cigarette butt",
                "can", "tin can"
            ]
            self._yolo_world_object.set_classes(self._yolo_world_object_classes)
            print(f"[YOLO-World] 模型初始化完成，检测类别: {len(self._yolo_world_object_classes)}种（不含person）")

        try:
            # 直接传入原图，YOLO 自动处理
            results = self._yolo_world_object.predict(roi, conf=0.4, imgsz=640, device=0, verbose=False)
        except Exception as e:
            return False

        if not results or not results[0].boxes:
            return False

        # 直接收集异物检测结果（人由 best.pt 判断）
        detections = []
        for box in results[0].boxes.data:
            dx1, dy1, dx2, dy2, conf, cls = box.cpu().numpy()
            if conf < 0.4:
                continue
            # 已经是 roi 坐标，加上偏移 x1, y1 得到原图坐标
            ox1 = int(dx1) + x1
            oy1 = int(dy1) + y1
            ox2 = int(dx2) + x1
            oy2 = int(dy2) + y1
            cls_idx = int(cls)
            cls_name = results[0].names[cls_idx]
            detections.append([ox1, oy1, ox2, oy2, cls_name])

        if not detections:
            return False

        last_save = self.object_last_save_time.get(cam, 0)
        if current_time - last_save < 120:
            return True

        self.object_last_save_time[cam] = current_time
        timestamp = datetime.now(BEIJING_TZ).strftime('%Y%m%d_%H%M%S')
        safe_cam = sanitize_filename(cam)

        result_frame = frame.copy()
        for ox1, oy1, ox2, oy2, cls_name in detections:
            cv2.rectangle(result_frame, (ox1, oy1), (ox2, oy2), (0, 0, 255), 5)

        output_path = f"output/object_detections/{safe_cam}_OBJECT_{timestamp}.jpg"
        cv2.imwrite(output_path, result_frame)

        dataset_path = f"dataset/{safe_cam}_OBJECT_{timestamp}.jpg"
        cv2.imwrite(dataset_path, frame)

        print(f"[🗑️ 异物检测] {cam}: 检测到 {len(detections)} 个异物 -> {output_path}")

        # 提交VLM分析
        print(f"[{cam}] 🤖 提交VLM分析（异物检测）...")
        self.vlm_queue.submit_task(dataset_path, "object")

        with self.stats_lock: self.total_stats['object'] += 1
        self.uploader.upload_detection(output_path, cam, 'object_detections', local_vlm_path=dataset_path)

        return True

    def detect_object_async(self, yolo_world, frame, cam, person_count):
        """异物检测 - 异步执行版本"""
        try:
            if person_count > 0:
                return False

            zone = self._get_elevator_detection_zone(cam)
            if zone is None:
                return False

            h, w = frame.shape[:2]
            x1, y1, x2, y2 = zone
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                return False

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                return False

            # 直接传入原图，YOLO 自动处理
            results = yolo_world.predict(roi, conf=0.4, imgsz=640, device=0, verbose=False)

            if not results or not results[0].boxes:
                return False

            detections = []
            for box in results[0].boxes.data:
                dx1, dy1, dx2, dy2, conf, cls = box.cpu().numpy()
                if conf < 0.4:
                    continue
                # 已经是 roi 坐标，加上偏移 x1, y1 得到原图坐标
                ox1 = int(dx1) + x1
                oy1 = int(dy1) + y1
                ox2 = int(dx2) + x1
                oy2 = int(dy2) + y1
                cls_name = results[0].names[int(cls)]
                detections.append([ox1, oy1, ox2, oy2, cls_name])

            if not detections:
                return False

            current_time = time.time()
            last_save = self.object_last_save_time.get(cam, 0)
            if current_time - last_save < 120:
                return True

            self.object_last_save_time[cam] = current_time
            timestamp = datetime.now(BEIJING_TZ).strftime('%Y%m%d_%H%M%S')
            safe_cam = sanitize_filename(cam)

            result_frame = frame.copy()
            for ox1, oy1, ox2, oy2, cls_name in detections:
                cv2.rectangle(result_frame, (ox1, oy1), (ox2, oy2), (0, 0, 255), 5)

            output_path = f"output/object_detections/{safe_cam}_OBJECT_{timestamp}.jpg"
            cv2.imwrite(output_path, result_frame)

            dataset_path = f"dataset/{safe_cam}_OBJECT_{timestamp}.jpg"
            cv2.imwrite(dataset_path, frame)

            print(f"[🗑️ 异物检测] {cam}: 检测到 {len(detections)} 个异物 -> {output_path}")

            print(f"[{cam}] 🤖 提交VLM分析（异物检测）...")
            self.vlm_queue.submit_task(dataset_path, "object")

            with self.stats_lock: self.total_stats['object'] += 1
            self.uploader.upload_detection(output_path, cam, 'object_detections', local_vlm_path=dataset_path)

            return True
        except Exception as e:
            print(f"[异物检测错误] {cam}: {e}")
            return False

    def _perform_ocr_detection(self, frame, cam):
        """执行OCR楼层号检测 - 使用EasyOCR"""
        if cam not in self.elevator_status.camera_status:
            return None, 0.0

        try:
            display_area = self.elevator_status.camera_status[cam]['display_area']
            x1, y1, x2, y2 = display_area['x1'], display_area['y1'], display_area['x2'], display_area['y2']
            rotate_angle = display_area.get('angle', 0)

            display_region = frame[y1:y2, x1:x2]
            if display_region.size == 0:
                return None, 0.0

            # 旋转
            if rotate_angle and rotate_angle != 0:
                h, w = display_region.shape[:2]
                center = (w / 2, h / 2)
                M = cv2.getRotationMatrix2D(center, rotate_angle, 1.0)
                display_region = cv2.warpAffine(display_region, M, (w, h))

            h, w = display_region.shape[:2]
            # 放大8倍
            enlarged = cv2.resize(display_region, (w*8, h*8), interpolation=cv2.INTER_CUBIC)

            # 保存调试图
            debug_dir = 'output/ocr_debug'
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(f"{debug_dir}/{cam}_ocr_preprocess.jpg", enlarged)

            # EasyOCR识别（检查ocr_reader是否可用）
            if self.ocr_reader is None:
                return None, 0.0
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
            print(f"[OCR错误] {cam}: {e}")
            return None, 0.0

    def _extract_floor_number(self, text):
        """从OCR结果中提取楼层号"""
        if not text:
            return None
        text = text.upper()
        text = text.replace('I', '1').replace('L', '1').replace('|', '1')
        text = text.replace('O', '0').replace('Q', '0')
        text = text.replace('S', '5').replace('Z', '2')
        text = text.replace('B', '8').replace('G', '6')
        text = text.replace('{', '1').replace('}', '1')
        text = text.replace('[', '1').replace(']', '1')

        import re
        numbers = re.findall(r'\d+', text)
        if not numbers:
            return None

        floor_num = int(numbers[0])
        if 0 <= floor_num <= 99:
            return floor_num
        return None

    def _save_queue_stats(self, vlm_stats, yolo_pending):
        try:
            with open('queue_stats.json', 'w', encoding='utf-8') as f:
                json.dump({
                    'timestamp': datetime.now(BEIJING_TZ).strftime('%Y-%m-%d %H:%M:%S'),
                    'vlm_pending': vlm_stats['pending'],
                    'vlm_processing': vlm_stats['processing'],
                    'yolo_pending': yolo_pending
                }, f)
        except: pass

    def _update_active_violations_file(self):
        """Update active violations file for web_viewer.py"""
        try:
            with open('active_violations.json', 'w', encoding='utf-8') as f:
                json.dump(self.active_violations, f, ensure_ascii=False)
        except: pass

    def _save_and_upload(self, frame, bbox, cam, folder, text, is_sequence=False, sequence_data=None):
        timestamp = datetime.now(BEIJING_TZ).strftime('%Y%m%d_%H%M%S')
        safe_cam = sanitize_filename(cam)
        clean_text = re.sub(r'[^a-zA-Z0-9_\-]', '_', text.upper())

        if is_sequence and sequence_data:
            seq_folder = f"output/{folder}/{safe_cam}_{clean_text}_{timestamp}"
            os.makedirs(seq_folder, exist_ok=True)
            print(f"[{cam}] 🆘 摔倒序列 -> {seq_folder}")

            dataset_seq_folder = f"dataset/{safe_cam}_{clean_text}_{timestamp}"
            os.makedirs(dataset_seq_folder, exist_ok=True)

            for img, name in sequence_data:
                cv2.imwrite(os.path.join(seq_folder, name), img)
                cv2.imwrite(os.path.join(dataset_seq_folder, name), img)

            print(f"[{cam}] 🤖 提交VLM序列分析...")
            self.vlm_queue.submit_task(seq_folder, text)
            self.uploader.upload_sequence(seq_folder, cam, folder)

        else:
            filename = f"{safe_cam}_{clean_text}_{timestamp}.jpg"
            path = f"output/{folder}/{filename}"
            dataset_path = f"dataset/{filename}"

            # 确保目录存在
            os.makedirs(f"output/{folder}", exist_ok=True)
            os.makedirs("dataset", exist_ok=True)

            img = frame.copy()
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)
            cv2.putText(img, text, (20, img.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            cv2.imwrite(path, img)
            cv2.imwrite(dataset_path, frame)
            print(f"[{cam}] 📸 截图保存 -> {filename}")

            # 非机动车检测和违停不需要VLM分析（模型已经能准确识别），但需要上传和通知
            text_upper = text.upper()
            skip_vlm = any(x in text_upper for x in ['EV', 'NON-MOTOR', 'NON_MOTOR', '电动车', '非机动车'])

            if skip_vlm:
                # 非机动车/违停：不需要VLM分析，直接上传和通知
                print(f"[{cam}] 📤 上传和通知...")
                self.uploader.upload_detection(path, cam, folder)
            else:
                # 提交VLM分析（其他类型需要）
                print(f"[{cam}] 🤖 提交VLM分析（原始图片）...")
                self.vlm_queue.submit_task(dataset_path, text)
                self.uploader.upload_detection(path, cam, folder, local_vlm_path=dataset_path)

    def process_camera(self, cam, url, dict_key=None):
        if dict_key is None: dict_key = cam
        print(f"[{cam}] 启动本地推理模式")

        self.status_mgr.update_status(cam, 'connecting')
        loader = RTSPStreamLoader(url, cam)
        tracker = None
        if BOT_SORT_AVAILABLE:
            tracker = BOTSORT(args=argparse.Namespace(
                track_buffer=60, match_thresh=0.8, with_reid=False, gmc_method='none',
                track_high_thresh=0.5, track_low_thresh=0.1, new_track_thresh=0.6,
                proximity_thresh=0.5, appearance_thresh=0.25, fuse_score=True, tracker_type='botsort'
            ), frame_rate=10)

        # 检查是否配置了 elevator_detection_zone
        cam_config = self.config.get('cameras', {}).get(dict_key, {})
        zone = cam_config.get('elevator_detection_zone', {})
        is_ev = zone and zone.get('enabled', False)

        f_cnt = 0

        # 异步任务计时器
        last_ocr_time = 0
        last_obj_time = 0

        while True:
            ret, frame = loader.read()
            if not ret:
                # 更新摄像头断开状态
                if self.prometheus_mgr:
                    try:
                        camera_ip = "unknown"
                        if 'rtsp://' in url:
                            import re
                            match = re.search(r'rtsp://([^:/]+)', url)
                            if match:
                                camera_ip = match.group(1)
                        self.prometheus_mgr.set_camera_status(cam, camera_ip, False)
                    except:
                        pass

                self.status_mgr.update_status(cam, 'disconnected')
                time.sleep(0.01)
                continue

            f_cnt += 1
            self.status_mgr.update_status(cam, 'connected')

            # 更新Prometheus摄像头连接状态
            if self.prometheus_mgr:
                try:
                    # 获取摄像头IP（从URL中提取）
                    camera_ip = "unknown"
                    if 'rtsp://' in url:
                        # 从rtsp://ip:port/... 中提取IP
                        import re
                        match = re.search(r'rtsp://([^:/]+)', url)
                        if match:
                            camera_ip = match.group(1)

                    self.prometheus_mgr.set_camera_status(cam, camera_ip, True)
                except Exception as e:
                    pass

            h, w = frame.shape[:2]

            # --- 第一级：高频实时任务 (每帧检测) ---
            # if f_cnt % 5 != 0:
            #     continue

            current_time = time.time()

            # 直接传入原图，YOLO 自动处理
            yolo = self.batch.infer(frame, camera_name=cam)
            if not yolo: continue

            # 计算原始人数（用于后续判断）
            raw_person_count = 0
            self._frame_person_detections = []  # 保存YOLO检测到的人的bbox和track_id
            if yolo.boxes is not None:
                for box in yolo.boxes:
                    if int(box.cls[0]) == 0:
                        raw_person_count += 1
                        # 先保存人的bbox，track_id后面再匹配
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        self._frame_person_detections.append({'bbox': [int(x1), int(y1), int(x2), int(y2)], 'track_id': None})

            # 更新电梯状态（用于停电检测）
            self.elevator_status.update_person_count(dict_key, raw_person_count)

            # Tracker 更新 - YOLO 返回的已经是原图坐标
            tracks = tracker.update(yolo.boxes.cpu(), frame) if tracker and yolo.boxes else []

            # 匹配人的track_id
            for person_info in self._frame_person_detections:
                person_bbox = person_info['bbox']
                px1, py1, px2, py2 = person_bbox
                for t in tracks:
                    if int(t[6]) == 0:  # person class
                        tx1, ty1, tx2, ty2 = t[:4]
                        # 检查是否匹配（有交集）
                        if (px1 < tx2 and px2 > tx1 and py1 < ty2 and py2 > ty1):
                            person_info['track_id'] = int(t[4])  # track_id是t[4]
                            break

            # live_counts
            with self.stats_lock:
                self.live_counts[cam] = {
                    'person': sum(1 for t in tracks if int(t[6]) == 0),
                    'vehicle': sum(1 for t in tracks if int(t[6]) in [2, 3])
                }

            for t in tracks:
                if len(t) < 7: continue
                bbox = t[:4]  # 已经是原图坐标
                tid, conf, cls = int(t[4]), float(t[5]), int(t[6])

                cls_map = {0:'person', 1:'bicycle', 2:'car', 3:'motorcycle', 4:'fire', 5:'smoke', 6:'animal', 7:'cigarette', 8:'stick'}
                thresh = self.conf_thresholds.get(cls_map.get(cls, 'default'), 0.3)

                if is_ev and cls == 3:
                    # 电动车检测
                    zone = self._get_elevator_detection_zone(dict_key)
                    if zone and not self._is_bbox_in_zone(bbox, zone):
                        continue

                    ev_timestamp = datetime.now(BEIJING_TZ).strftime('%Y%m%d_%H%M%S')
                    safe_cam = sanitize_filename(cam)
                    result_frame = frame.copy()
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                    output_path = f"output/ev_violations/{safe_cam}_EV_{ev_timestamp}.jpg"
                    cv2.imwrite(output_path, result_frame)

                    dataset_path = f"dataset/{safe_cam}_EV_{ev_timestamp}.jpg"
                    cv2.imwrite(dataset_path, frame)

                    self.vlm_queue.submit_task(dataset_path, "ev")
                    with self.stats_lock: self.total_stats['ev'] += 1
                    self._record_detection_prometheus(cam, 'ev', 'C')
                    self._save_and_upload(frame, bbox, cam, 'ev_violations', "EV VIOLATION")

                # 宠物检测
                elif cls == 6:
                    if conf < thresh:
                        continue
                    print(f"[{cam}] 🐾 检测到宠物! tid={tid}, conf={conf:.2f}, bbox={bbox}")
                    with self.stats_lock: self.total_stats['pet'] += 1
                    self._record_detection_prometheus(cam, 'pet', 'C')
                    self._save_and_upload(frame, bbox, cam, 'pet_detections', "PET DETECTION")

                # 机动车检测 (cls=2) - 同一ID只统计一次
                elif not is_ev and cls == 2:
                    if conf < thresh:
                        continue
                    # 检查是否配置了违停检测区域，没有配置则不检测
                    zones = self.config['cameras'].get(dict_key, {}).get('violation_zones', {})
                    if zones:
                        key = f"{cam}_{tid}"
                        # 同一ID只统计一次（不重复统计）
                        if key not in self.motor_counted:
                            self.motor_counted.add(key)
                            with self.stats_lock: self.total_stats['motor_vio'] += 1
                            self._record_detection_prometheus(cam, 'motor_vehicle', 'C')
                            self._save_and_upload(frame, bbox, cam, 'motor_vehicle_violations', "MOTOR DETECTED")

                        # 违停检测（持续违停告警）
                        if key not in self.trackers:
                            self.trackers[key] = VehicleTracker(tid, str(cls), time.time(), cam,
                                                                self.violation_time_threshold, self.stationary_threshold)

                        in_zone = True
                        if self.zone_detection_enabled:
                            zones = self.config['cameras'].get(dict_key, {}).get('violation_zones', {})
                            if zones:
                                cx, cy = (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2
                                pts = [np.array(z['points'], np.int32) for z in zones.values() if z.get('enabled', True)]
                                in_zone = any(cv2.pointPolygonTest(p, (cx, cy), False) >= 0 for p in pts)

                        vt = self.trackers[key]
                        if vt.update(bbox, time.time(), False, in_zone):
                            if not vt.violation_saved:
                                vt.violation_saved = True
                                self._save_and_upload(frame, bbox, cam, 'motor_vehicle_violations', "MOTOR VIOLATION")

                # 非机动车检测 (cls=1 bicycle, cls=3 motorcycle) - 同一ID只统计一次，有区域就用区域
                elif not is_ev and cls in [1, 3]:
                    if conf < thresh:
                        continue
                    key = f"{cam}_{tid}"

                    # 检查区域
                    zones = self.config['cameras'].get(dict_key, {}).get('violation_zones', {})
                    in_zone = True
                    if zones and self.zone_detection_enabled:
                        cx, cy = (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2
                        pts = [np.array(z['points'], np.int32) for z in zones.values() if z.get('enabled', True)]
                        in_zone = any(cv2.pointPolygonTest(p, (cx, cy), False) >= 0 for p in pts)

                    # 有区域必须在区域内，无区域就是全屏
                    if in_zone or not zones:
                        if key not in self.non_motor_counted:
                            self.non_motor_counted.add(key)
                            with self.stats_lock: self.total_stats['non_motor_vio'] += 1
                            self._record_detection_prometheus(cam, 'non_motor_vehicle', 'C')
                            self._save_and_upload(frame, bbox, cam, 'non_motor_violations', "NON-MOTOR DETECTED")

                    # 2. 违停检测（仅在配置了区域时才进行）
                    zones = self.config['cameras'].get(dict_key, {}).get('violation_zones', {})
                    if zones:
                        if key not in self.trackers:
                            self.trackers[key] = VehicleTracker(tid, str(cls), time.time(), cam,
                                                                self.violation_time_threshold, self.stationary_threshold)

                        in_zone = True
                        if self.zone_detection_enabled:
                            if zones:
                                cx, cy = (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2
                                pts = [np.array(z['points'], np.int32) for z in zones.values() if z.get('enabled', True)]
                                in_zone = any(cv2.pointPolygonTest(p, (cx, cy), False) >= 0 for p in pts)

                        vt = self.trackers[key]
                        if vt.update(bbox, time.time(), False, in_zone):
                            if not vt.violation_saved:
                                vt.violation_saved = True
                                self._save_and_upload(frame, bbox, cam, 'non_motor_violations', "NON-MOTOR VIOLATION")

                # Pose 检测 (摔倒/打架)
                elif cls == 0 and conf > thresh:
                    x1, y1, x2, y2 = map(int, bbox)
                    if (x2-x1)>20 and (y2-y1)>40:
                        res = self.batch.infer(frame[y1:y2, x1:x2], is_pose=True, camera_name=cam)
                        if res and res[0].keypoints:
                            kpts = res[0].keypoints.xy.cpu().numpy()[0]
                            kpts[:,0] += x1
                            kpts[:,1] += y1

                            key = f"{cam}_{tid}"
                            if key not in self.buffers:
                                cfg = self.camera_fall_configs.get(dict_key, {'baseline_angle':0.0, 'fall_threshold':50.0})
                                self.buffers[key] = FallVideoBuffer(
                                    tid, f_cnt,
                                    fall_angle_threshold=cfg['fall_threshold'],
                                    min_frames_required=self.min_frames_required,
                                    baseline_angle=cfg['baseline_angle'],
                                    angle_change_threshold=self.angle_change_threshold,
                                    fallen_duration_required=self.fallen_duration_required,
                                    height_drop_ratio=self.height_drop_ratio
                                )
                                self.fight_buffers[key] = FightBuffer(tid, min_frames_required=5)

                            buf = self.buffers[key]
                            buf.update(kpts, frame, f_cnt)
                            if buf.check_fall(w, h) and not buf.confirmed:
                                buf.confirmed = True
                                with self.stats_lock: self.total_stats['fall'] += 1
                                self._record_detection_prometheus(cam, 'fall', 'B')
                                self._save_and_upload(frame, bbox, cam, 'behavior_anomalies', "FALL", False, None)

                            if not hasattr(self, '_frame_person_poses'):
                                self._frame_person_poses = {}
                            self._frame_person_poses[key] = {'kpts': kpts, 'bbox': bbox}

            # --- 第二级：中低频异步任务 ---
            has_person_poses = hasattr(self, '_frame_person_poses') and self._frame_person_poses
            if has_person_poses:
                frame_poses_copy = list(self._frame_person_poses.items())
                try:
                    for key, data in frame_poses_copy:
                        kpts = data['kpts']
                        bbox = data['bbox']

                        # 打架检测
                        if key in self.fight_buffers:
                            other_poses = []
                            for other_key, other_data in frame_poses_copy:
                                if other_key != key:
                                    cx1, cy1 = (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2
                                    other_bbox = other_data['bbox']
                                    cx2, cy2 = (other_bbox[0]+other_bbox[2])/2, (other_bbox[1]+other_bbox[3])/2
                                    dist = ((cx1-cx2)**2 + (cy1-cy2)**2)**0.5
                                    if dist < 200:
                                        other_poses.append(other_data['kpts'])

                            fight_buf = self.fight_buffers[key]
                            fight_buf.update(kpts, other_poses, frame, f_cnt)
                            if fight_buf.check_fight() and not fight_buf.confirmed:
                                fight_buf.confirmed = True
                                with self.stats_lock: self.total_stats['fight'] += 1
                                self._record_detection_prometheus(cam, 'fight', 'B')
                                self._save_and_upload(frame, bbox, cam, 'behavior_anomalies', "FIGHT", False, None)
                except Exception as e:
                    pass  # 静默处理错误

            # ==================== 危险物品检测 ====================
            # 已禁用YOLO-World，改用best.pt的cls=8(stick)检测
            # 危险品检测逻辑已移到上面的best.pt检测中（cls=8）

            # 火灾/烟雾检测 - YOLO 返回的已经是原图坐标
            if not is_ev:
                for box in yolo.boxes.data.cpu():
                    cls = int(box[5])
                    if cls in [4, 5]:
                        lbl = 'fire' if cls == 4 else 'smoke'
                        conf = float(box[4])
                        thresh = self.conf_thresholds.get(lbl, 0.3)
                        if conf > thresh:
                            # 已经是原图坐标
                            fx1, fy1, fx2, fy2 = box[:4]
                            print(f"[{cam}] 🔥{'火' if lbl=='fire' else '烟'}检测! conf={conf:.2f}, cls={cls}")
                            with self.stats_lock: self.total_stats['fire' if lbl=='fire' else 'smoke'] += 1
                            self._record_detection_prometheus(cam, lbl, 'A')
                            self._save_and_upload(frame, [fx1, fy1, fx2, fy2], cam, f'{lbl}_detections', lbl.upper())

                    # 香烟检测 (cls=7) - 需要与人有交集，同一ID只告警一次
                    if cls == 7:
                        conf = float(box[4])
                        thresh = self.conf_thresholds.get('cigarette', 0.3)
                        if conf > thresh:
                            # 检查是否与人体有交集
                            cig_bbox = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
                            matched_tid = None
                            if hasattr(self, '_frame_person_detections'):
                                for pinfo in self._frame_person_detections:
                                    pbox = pinfo['bbox']
                                    ix1 = max(cig_bbox[0], pbox[0])
                                    iy1 = max(cig_bbox[1], pbox[1])
                                    ix2 = min(cig_bbox[2], pbox[2])
                                    iy2 = min(cig_bbox[3], pbox[3])
                                    if ix1 < ix2 and iy1 < iy2:
                                        matched_tid = pinfo['track_id']
                                        break

                            if matched_tid is not None:
                                key = f"{cam}_{matched_tid}_cigarette"
                                current_time = time.time()
                                last_time = getattr(self, '_cigarette_alert_last_time', {}).get(key, 0)
                                if current_time - last_time >= 30:
                                    # 设置告警时间
                                    if not hasattr(self, '_cigarette_alert_last_time'):
                                        self._cigarette_alert_last_time = {}
                                    self._cigarette_alert_last_time[key] = current_time
                                    print(f"[{cam}] 🚬 香烟检测! conf={conf:.2f}, tid={matched_tid}")
                                    with self.stats_lock: self.total_stats['cigarette'] += 1
                                    self._save_and_upload(frame, cig_bbox, cam, 'cigarette_detections', 'CIGARETTE')

                    # 危险品检测 (cls=8 stick) - 需要与人有交集，同一ID只告警一次
                    if cls == 8:
                        conf = float(box[4])
                        thresh = self.conf_thresholds.get('stick', 0.3)
                        if conf > thresh:
                            # 检查是否与人体有交集
                            stick_bbox = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
                            matched_tid = None
                            if hasattr(self, '_frame_person_detections'):
                                for pinfo in self._frame_person_detections:
                                    pbox = pinfo['bbox']
                                    ix1 = max(stick_bbox[0], pbox[0])
                                    iy1 = max(stick_bbox[1], pbox[1])
                                    ix2 = min(stick_bbox[2], pbox[2])
                                    iy2 = min(stick_bbox[3], pbox[3])
                                    if ix1 < ix2 and iy1 < iy2:
                                        matched_tid = pinfo['track_id']
                                        break

                            if matched_tid is not None:
                                key = f"{cam}_{matched_tid}_stick"
                                current_time = time.time()
                                last_time = getattr(self, '_stick_alert_last_time', {}).get(key, 0)
                                if current_time - last_time >= 30:
                                    # 设置告警时间
                                    if not hasattr(self, '_stick_alert_last_time'):
                                        self._stick_alert_last_time = {}
                                    self._stick_alert_last_time[key] = current_time
                                    print(f"[{cam}] 🔪 危险品检测! conf={conf:.2f}, tid={matched_tid}")
                                    with self.stats_lock: self.total_stats['dangerous_item'] += 1
                                    self._record_detection_prometheus(cam, 'dangerous_item', 'B')
                                    self._save_and_upload(frame, stick_bbox, cam, 'dangerous_item_detections', 'DANGEROUS_ITEM')

            # 清理本帧的pose数据
            if hasattr(self, '_frame_person_poses'):
                self._frame_person_poses.clear()

            # --- 第三级：异步OCR (每1秒一次) ---
            if dict_key in self.elevator_status.camera_status:
                if current_time - last_ocr_time >= 1.0:
                    # 提交异步任务，不等待
                    self.async_manager.submit('ocr', frame, dict_key)
                    last_ocr_time = current_time

                # 获取当前楼层状态（使用上一次的OCR结果）
                status = self.elevator_status.camera_status.get(dict_key, {})
                floor_number = status.get('last_floor_number')

                # 更新Prometheus电梯状态指标
                if self.prometheus_mgr:
                    try:
                        self.prometheus_mgr.set_elevator_person_count(cam, status.get('people_count', 0))
                        self.prometheus_mgr.set_elevator_floor_number(cam, floor_number)
                        self.prometheus_mgr.set_elevator_floor_stuck_count(cam, status.get('success_count', 0))
                    except:
                        pass

                violation_type = self.elevator_status.update_floor_detection(
                    dict_key, floor_number, 0.0, current_time
                )

                if violation_type:
                    current_status = self.elevator_status.camera_status.get(dict_key, {})
                    if violation_type == 'power_failure':
                        if current_status.get('people_count', 0) == 0:
                            continue
                        self._record_detection_prometheus(cam, 'power_failure', 'A')
                        if self.prometheus_mgr:
                            self.prometheus_mgr.set_elevator_fault(cam, 'power_failure', True)
                        self._save_and_upload(frame, [0, 0, frame.shape[1], frame.shape[0]], cam, 'elevator_power_failures', 'POWER_FAILURE')
                    elif violation_type == 'floor_stuck':
                        if current_status.get('people_count', 0) == 0:
                            continue
                        self._record_detection_prometheus(cam, 'floor_stuck', 'A')
                        if self.prometheus_mgr:
                            self.prometheus_mgr.set_elevator_fault(cam, 'floor_stuck', True)
                        self._save_and_upload(frame, [0, 0, frame.shape[1], frame.shape[0]], cam, 'elevator_floor_stuck', 'FLOOR_STUCK')

            # --- 第四级：异步异物检测 (每5秒一次) ---
            if self._is_elevator_camera(dict_key):
                if current_time - last_obj_time >= 5.0:
                    # 提交异步任务，不等待
                    self.async_manager.submit('object', frame, dict_key, person_count=raw_person_count)
                    last_obj_time = current_time

    def run(self):
        print("🚀 启动检测系统 v5.7 PyTorch版本 (按Ctrl+C停止)")


        self.uploader.start()

        cameras = []
        for k, v in self.config['cameras'].items():
            if not v.get('disabled', False):
                cameras.append((v.get('name', k), v['rtsp_url'], k))

        for args in cameras:
            threading.Thread(target=self.process_camera, args=args, daemon=True).start()
            time.sleep(0.5)

        try:
            while True:
                time.sleep(60)  # 每分钟输出一次统计
                now = time.time()

                # 清理过期的tracker和对应的违停记录
                expired_keys = [k for k, v in self.trackers.items() if now - v.last_seen >= 30]
                for key in expired_keys:
                    # 如果有过期的违停记录，也一并清理
                    if key in self.active_violations:
                        filepath = self.active_violations.pop(key)
                        try:
                            if os.path.exists(filepath):
                                os.remove(filepath)
                                print(f"🕐 清理过期违停记录: {filepath}")
                        except:
                            pass
                        # 更新文件
                        self._update_active_violations_file()

                self.trackers = {k:v for k,v in self.trackers.items() if now - v.last_seen < 30}
                self.buffers = {k:v for k,v in self.buffers.items() if now - v.last_time < 30}
                self.fight_buffers = {k:v for k,v in self.fight_buffers.items() if now - v.last_time < 30}
                self.fire_det.cleanup()

                # 清理超过24小时的违停记录（防止持久化文件堆积）
                if os.path.exists('active_violations.json'):
                    try:
                        with open('active_violations.json', 'r') as f:
                            old_data = json.load(f)
                        cleaned_data = {}
                        for key, filepath in old_data.items():
                            if os.path.exists(filepath) and now - os.path.getmtime(filepath) < 86400:
                                cleaned_data[key] = filepath
                        if len(cleaned_data) != len(old_data):
                            with open('active_violations.json', 'w') as f:
                                json.dump(cleaned_data, f, ensure_ascii=False)
                            print(f"🕐 清理过期违停文件: {len(old_data) - len(cleaned_data)} 条")
                    except:
                        pass

                vlm_s = self.vlm_queue.get_stats()
                # 使用更准确的计数器而不是可能不准确的qsize()
                if hasattr(self.batch, 'q_det_counter'):
                    yolo_q = self.batch.q_det_counter
                else:
                    yolo_q = self.batch.q_det.qsize() if hasattr(self.batch, 'q_det') else 0
                self._save_queue_stats(vlm_s, yolo_q)

                # 更新并输出资源监控
                if self.performance_mgr:
                    self.performance_mgr.update_queue_stats(yolo_q, vlm_s['pending'])
                    print(self.performance_mgr.get_summary())

                with self.stats_lock:
                    s = self.total_stats
                    print(f"[统计] VLM队列:{vlm_s['pending']} | 机动车:{s.get('motor_vio', 0)} | 非机动车:{s.get('non_motor_vio', 0)} | 摔倒:{s.get('fall', 0)} | 打架:{s.get('fight', 0)} | 危险品:{s.get('dangerous_item', 0)} | 火:{s.get('fire', 0)} | 烟:{s.get('smoke', 0)} | 吸烟:{s.get('cigarette', 0)}")

        except KeyboardInterrupt:
            print("\n停止系统...")
        finally:
            self.uploader.stop()

if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base)  # 确保工作目录正确
    if os.path.exists(os.path.join(base, 'best.pt')):
        print("🚀 启动检测系统 (PyTorch 纯本地模式)")
        UnifiedSystem(
            os.path.join(base, 'camera_zones_config.json'),
            os.path.join(base, 'best.pt')
        ).run()
    else:
        print("❌ 缺少 best.pt")
