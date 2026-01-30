"""
员工行为检测工具 - 新流程
使用数学计算方法判断人员身份,解决现有方法的问题:
1. 不对人物放大分析(保留功能接口供后续使用)
2. 使用数学计算而非大模型判断内外左右
3. 识别办公设备(显示器、键盘)和办公桌,拟合直线判断身份
4. 不使用JSON,使用简单文本格式传递数据

检测流程:
- 步骤1: 识别办公设备和工作台(显示器、键盘、办公桌),输出坐标框(简单文本格式)
- 步骤2: 识别人员位置(包括截断人员),输出坐标框(简单文本格式)
- 步骤3: 检测缺岗(无人工位),双重检验
- 步骤4: 拟合直线,使用所有设备和办公桌中心点,通过数学计算判断人员身份(工作人员/顾客)
- 步骤5: 在原图上绘制所有坐标框、拟合直线和身份标签
- 步骤6: 添加缺岗标注
- 步骤7: 检测工作人员违规行为(睡觉、便服、玩手机)
- 步骤8: 检测工作人员聊天行为
- 步骤9: 绘制人员框和违规行为标签
- 最终: 绘制所有结果并保存汇总文件
"""
import os
import base64
import json
import re
import numpy as np
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime

# ==================== 全局配置 ====================
AI_MODEL = "qwen3-vl-235b-a22b-instruct"  # 视觉模型
AI_MODEL_CHAT = "qwen3-32b"  # 文本模型
# =================================================


# ==================== 日志记录器 ====================
class ProcessLogger:
    """处理过程日志记录器 - 记录所有控制台输出和模型调用"""

    def __init__(self):
        self.logs = []
        self.model_calls = []

    def log(self, message):
        """记录日志（同时输出到控制台）"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)
        print(message)  # 同时输出到控制台

    def log_model_call(self, step_name, prompt, response):
        """记录模型调用详情"""
        self.model_calls.append({
            "step": step_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "prompt": prompt,
            "response": response
        })

    def save(self, filepath):
        """保存完整日志到文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("员工行为检测 - 完整处理日志\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")

            # 1. 处理流程日志
            f.write("="*80 + "\n")
            f.write("【处理流程日志】\n")
            f.write("="*80 + "\n\n")
            for log in self.logs:
                f.write(log + "\n")

            # 2. 模型调用详情
            if self.model_calls:
                f.write("\n\n" + "="*80 + "\n")
                f.write("【模型调用详情】\n")
                f.write("="*80 + "\n\n")

                for i, call in enumerate(self.model_calls, 1):
                    f.write(f"{'='*80}\n")
                    f.write(f"模型调用 {i}: {call['step']}\n")
                    f.write(f"时间: {call['timestamp']}\n")
                    f.write(f"{'='*80}\n\n")

                    f.write("【提示词】\n")
                    prompt_text = call['prompt']
                    if len(prompt_text) > 500:
                        f.write(prompt_text[:500])
                        f.write(f"\n... (提示词过长，已截断，完整长度: {len(prompt_text)} 字符)")
                    else:
                        f.write(prompt_text)
                    f.write("\n\n")

                    f.write("【模型响应】\n")
                    response_text = call['response']
                    if len(response_text) > 1000:
                        f.write(response_text[:1000])
                        f.write(f"\n... (响应过长，已截断，完整长度: {len(response_text)} 字符)")
                    else:
                        f.write(response_text)
                    f.write("\n\n")

        print(f"\n✓ 完整日志已保存到: {os.path.basename(filepath)}")


# ==================== 时间跟踪器 ====================
class TimeTracker:
    """时间跟踪器 - 记录模型调用和图像处理的时间统计"""

    def __init__(self):
        self.image_timings = []  # 每张图片的总处理时间
        self.model_calls = []  # 模型调用记录
        self.processing_steps = []  # 图像处理步骤记录
        self.current_image = None
        self.current_image_start = None
        self.image_count = 0

    def start_image(self, image_name):
        """开始处理一张新图片"""
        self.current_image = image_name
        self.current_image_start = datetime.now()
        self.image_count += 1

    def end_image(self):
        """结束处理当前图片"""
        if self.current_image_start:
            elapsed = (datetime.now() - self.current_image_start).total_seconds()
            self.image_timings.append({
                "image": self.current_image,
                "time": elapsed
            })
            print(f"  ⏱ 图片 '{self.current_image}' 处理耗时: {elapsed:.2f}秒")
            self.current_image = None
            self.current_image_start = None

    def record_model_call(self, step_name, model_name):
        """装饰器：记录模型调用时间"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start = datetime.now()
                result = func(*args, **kwargs)
                elapsed = (datetime.now() - start).total_seconds()

                self.model_calls.append({
                    "image": self.current_image,
                    "step": step_name,
                    "model": model_name,
                    "time": elapsed
                })

                print(f"  ⏱ {step_name} 模型调用耗时: {elapsed:.2f}秒 ({model_name})")
                return result
            return wrapper
        return decorator

    def record_processing_step(self, step_name):
        """装饰器：记录图像处理步骤时间"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start = datetime.now()
                result = func(*args, **kwargs)
                elapsed = (datetime.now() - start).total_seconds()

                self.processing_steps.append({
                    "image": self.current_image,
                    "step": step_name,
                    "time": elapsed
                })

                print(f"  ⏱ {step_name} 处理耗时: {elapsed:.2f}秒")
                return result
            return wrapper
        return decorator

    def get_summary(self):
        """计算统计摘要"""
        if not self.image_timings:
            return None

        total_time = sum(t["time"] for t in self.image_timings)
        avg_time = total_time / len(self.image_timings)

        # 按步骤统计模型调用时间
        model_stats = {}
        for call in self.model_calls:
            step = call["step"]
            if step not in model_stats:
                model_stats[step] = {"total": 0, "count": 0, "model": call["model"]}
            model_stats[step]["total"] += call["time"]
            model_stats[step]["count"] += 1

        # 按步骤统计图像处理时间
        processing_stats = {}
        for step in self.processing_steps:
            step_name = step["step"]
            if step_name not in processing_stats:
                processing_stats[step_name] = {"total": 0, "count": 0}
            processing_stats[step_name]["total"] += step["time"]
            processing_stats[step_name]["count"] += 1

        return {
            "total_images": self.image_count,
            "total_time": total_time,
            "avg_time": avg_time,
            "model_stats": model_stats,
            "processing_stats": processing_stats
        }

    def print_summary(self):
        """打印统计摘要"""
        summary = self.get_summary()
        if not summary:
            print("\n⚠ 没有时间统计数据")
            return

        print("\n" + "="*80)
        print("【时间统计摘要】")
        print("="*80)
        print(f"\n总图片数: {summary['total_images']}")
        print(f"总处理时间: {summary['total_time']:.2f}秒")
        print(f"平均每张图片: {summary['avg_time']:.2f}秒")

        print("\n" + "-"*80)
        print("模型调用统计:")
        print("-"*80)
        if summary['model_stats']:
            for step, stats in summary['model_stats'].items():
                avg = stats['total'] / stats['count']
                print(f"  {step}:")
                print(f"    调用次数: {stats['count']}")
                print(f"    总耗时: {stats['total']:.2f}秒")
                print(f"    平均耗时: {avg:.2f}秒")
                print(f"    使用模型: {stats['model']}")
        else:
            print("  无模型调用记录")

        print("\n" + "-"*80)
        print("图像处理步骤统计:")
        print("-"*80)
        if summary['processing_stats']:
            for step, stats in summary['processing_stats'].items():
                avg = stats['total'] / stats['count']
                print(f"  {step}:")
                print(f"    执行次数: {stats['count']}")
                print(f"    总耗时: {stats['total']:.2f}秒")
                print(f"    平均耗时: {avg:.2f}秒")
        else:
            print("  无处理步骤记录")
        print("="*80 + "\n")

    def save_to_file(self, filepath):
        """保存详细时间统计到文件"""
        summary = self.get_summary()
        if not summary:
            return

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("员工行为检测 - 时间统计报告\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")

            # 总体统计
            f.write("【总体统计】\n")
            f.write("-"*80 + "\n")
            f.write(f"总图片数: {summary['total_images']}\n")
            f.write(f"总处理时间: {summary['total_time']:.2f}秒\n")
            f.write(f"平均每张图片: {summary['avg_time']:.2f}秒\n\n")

            # 每张图片详情
            f.write("【每张图片处理时间】\n")
            f.write("-"*80 + "\n")
            for i, timing in enumerate(self.image_timings, 1):
                f.write(f"  {i}. {timing['image']}: {timing['time']:.2f}秒\n")
            f.write("\n")

            # 模型调用统计
            f.write("【模型调用统计】\n")
            f.write("-"*80 + "\n")
            if summary['model_stats']:
                for step, stats in summary['model_stats'].items():
                    avg = stats['total'] / stats['count']
                    f.write(f"\n{step}:\n")
                    f.write(f"  调用次数: {stats['count']}\n")
                    f.write(f"  总耗时: {stats['total']:.2f}秒\n")
                    f.write(f"  平均耗时: {avg:.2f}秒\n")
                    f.write(f"  使用模型: {stats['model']}\n")
            else:
                f.write("无模型调用记录\n")
            f.write("\n")

            # 图像处理步骤统计
            f.write("【图像处理步骤统计】\n")
            f.write("-"*80 + "\n")
            if summary['processing_stats']:
                for step, stats in summary['processing_stats'].items():
                    avg = stats['total'] / stats['count']
                    f.write(f"\n{step}:\n")
                    f.write(f"  执行次数: {stats['count']}\n")
                    f.write(f"  总耗时: {stats['total']:.2f}秒\n")
                    f.write(f"  平均耗时: {avg:.2f}秒\n")
            else:
                f.write("无处理步骤记录\n")
            f.write("\n")

            # 详细记录
            f.write("\n【详细模型调用记录】\n")
            f.write("-"*80 + "\n")
            for call in self.model_calls:
                f.write(f"\n图片: {call['image']}\n")
                f.write(f"步骤: {call['step']}\n")
                f.write(f"模型: {call['model']}\n")
                f.write(f"耗时: {call['time']:.2f}秒\n")

            f.write("\n\n" + "="*80 + "\n")

        print(f"✓ 时间统计已保存到: {os.path.basename(filepath)}")


# 全局时间跟踪器实例
time_tracker = TimeTracker()
# =================================================

# 全局日志记录器实例
process_logger = None
# =================================================


def encode_image(image_path):
    """将本地图像转换为Base64编码,返回(base64_data, mime_type)"""
    try:
        with Image.open(image_path) as img:
            if img.size[0] == 0 or img.size[1] == 0:
                raise ValueError(f"图片尺寸无效: {img.size}")

            format_map = {
                'JPEG': 'image/jpeg',
                'JPG': 'image/jpeg',
                'PNG': 'image/png',
                'BMP': 'image/bmp',
                'GIF': 'image/gif'
            }
            mime_type = format_map.get(img.format.upper(), 'image/jpeg')
    except Exception as e:
        raise ValueError(f"图片验证失败 {image_path}: {e}")

    with open(image_path, "rb") as image_file:
        base64_data = base64.b64encode(image_file.read()).decode("utf-8")

    return base64_data, mime_type


def save_raw_response(text, filepath):
    """保存原始响应文本用于调试"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("原始响应 (用于调试)\n")
            f.write("="*80 + "\n\n")
            f.write(text)
            f.write("\n" + "="*80 + "\n")
    except Exception as e:
        print(f"⚠ 保存原始响应失败: {e}")


def extract_bbox_from_text(text):
    """
    从文本中提取所有bbox坐标框

    返回格式: [[x1, y1, x2, y2], ...]
    """
    bboxes = []

    # 匹配模式: [数字, 数字, 数字, 数字] 或 (数字, 数字, 数字, 数字)
    pattern1 = r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
    pattern2 = r'\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)'

    for pattern in [pattern1, pattern2]:
        matches = re.findall(pattern, text)
        for match in matches:
            x1, y1, x2, y2 = map(int, match)
            bboxes.append([x1, y1, x2, y2])

    return bboxes


def extract_equipment_from_text(text):
    """
    从文本中提取设备信息

    返回格式: [
        {"type": "monitor", "bbox": [x1, y1, x2, y2], "center": [cx, cy]},
        ...
    ]
    """
    equipments = []

    # 按行分割
    lines = text.split('\n')

    for line in lines:
        # 查找设备类型
        device_type = None
        if '显示器' in line or 'monitor' in line.lower():
            device_type = 'monitor'
        elif '键盘' in line or 'keyboard' in line.lower():
            device_type = 'keyboard'
        elif '办公桌' in line or 'desk' in line.lower():
            device_type = 'desk'

        if device_type:
            # 提取bbox
            bboxes = extract_bbox_from_text(line)
            if bboxes:
                bbox = bboxes[0]

                # 计算中心点
                cx = (bbox[0] + bbox[2]) // 2
                cy = (bbox[1] + bbox[3]) // 2

                equipments.append({
                    "type": device_type,
                    "bbox": bbox,
                    "center": [cx, cy]
                })

    return equipments


def extract_persons_from_text(text):
    """
    从文本中提取人员信息

    返回格式: [
        {"bbox": [x1, y1, x2, y2]},
        ...
    ]
    """
    persons = []
    bboxes = extract_bbox_from_text(text)

    for bbox in bboxes:
        persons.append({
            "bbox": bbox
        })

    return persons


@time_tracker.record_processing_step("单人截图处理")
def crop_and_resize_person(image_path, bbox_norm, target_size=640, all_persons=None):
    """
    裁剪人员图像并调整大小

    步骤：
    1. 以人员为中心截取正方形，边长 = 人员框长宽最大值 × 1.5
    2. 如果正方形边长 < 320，则调整为 320
    3. 用灰色框覆盖裁剪区域内的其他人员

    参数:
        image_path: 原始图像路径
        bbox_norm: 归一化坐标 [x1, y1, x2, y2] (0-999)
        target_size: 目标尺寸(最小边长)
        all_persons: 所有人员的列表，用于覆盖其他人员

    返回:
        cropped_image: 裁剪并调整后的PIL Image对象
    """
    # 打开原始图像
    img = Image.open(image_path)
    width, height = img.size

    # 映射到实际坐标
    x1 = int(bbox_norm[0] / 1000 * width)
    y1 = int(bbox_norm[1] / 1000 * height)
    x2 = int(bbox_norm[2] / 1000 * width)
    y2 = int(bbox_norm[3] / 1000 * height)

    # 确保坐标正确
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    # 计算人员框的长宽
    person_width = x2 - x1
    person_height = y2 - y1

    # 计算人员中心点
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    # 步骤1: 正方形边长 = 人员框长宽最大值 × 1.5
    square_size = int(max(person_width, person_height) * 1.5)

    # 步骤2: 如果正方形边长 < 320，则调整为 320
    min_square_size = 320
    if square_size < min_square_size:
        square_size = min_square_size

    # 计算正方形区域的左上角和右下角坐标
    crop_x1 = center_x - square_size // 2
    crop_y1 = center_y - square_size // 2
    crop_x2 = center_x + square_size // 2
    crop_y2 = center_y + square_size // 2

    # 确保不超出图像边界
    if crop_x1 < 0:
        crop_x1 = 0
    if crop_y1 < 0:
        crop_y1 = 0
    if crop_x2 > width:
        crop_x2 = width
    if crop_y2 > height:
        crop_y2 = height

    # 裁剪人员区域
    cropped = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))

    # 步骤3: 用灰色框覆盖其他人员
    if all_persons:
        draw = ImageDraw.Draw(cropped)

        # 计算当前待检测人员在裁剪图像中的bbox（用于检查覆盖冲突）
        current_crop_x1 = x1 - crop_x1
        current_crop_y1 = y1 - crop_y1
        current_crop_x2 = x2 - crop_x1
        current_crop_y2 = y2 - crop_y1

        # 遍历所有其他人员
        covered_count = 0
        skipped_count = 0
        for i, person in enumerate(all_persons):
            # 获取其他人员信息
            other_bbox = person.get('bbox', [0, 0, 0, 0])
            other_identity = person.get('identity', '未知')

            # 跳过当前人员自己（通过bbox元素比较）
            # bbox_norm: 归一化坐标 [x1, y1, x2, y2]
            if (abs(other_bbox[0] - bbox_norm[0]) < 1 and
                abs(other_bbox[1] - bbox_norm[1]) < 1 and
                abs(other_bbox[2] - bbox_norm[2]) < 1 and
                abs(other_bbox[3] - bbox_norm[3]) < 1):
                # 这是当前人员自己，跳过
                continue

            # 映射到实际坐标
            ox1 = int(other_bbox[0] / 1000 * width)
            oy1 = int(other_bbox[1] / 1000 * height)
            ox2 = int(other_bbox[2] / 1000 * width)
            oy2 = int(other_bbox[3] / 1000 * height)

            # 确保坐标正确
            if ox1 > ox2:
                ox1, ox2 = ox2, ox1
            if oy1 > oy2:
                oy1, oy2 = oy2, oy1

            # 转换到裁剪图像的坐标系
            other_crop_x1 = ox1 - crop_x1
            other_crop_y1 = oy1 - crop_y1
            other_crop_x2 = ox2 - crop_x1
            other_crop_y2 = oy2 - crop_y1

            # 检查是否在裁剪区域内（有重叠）
            crop_width = crop_x2 - crop_x1
            crop_height = crop_y2 - crop_y1

            # 如果bbox与裁剪区域有重叠，则覆盖
            if (other_crop_x2 > 0 and other_crop_x1 < crop_width and
                other_crop_y2 > 0 and other_crop_y1 < crop_height):

                # 限制在裁剪区域内
                overlay_x1 = max(0, other_crop_x1)
                overlay_y1 = max(0, other_crop_y1)
                overlay_x2 = min(crop_width, other_crop_x2)
                overlay_y2 = min(crop_height, other_crop_y2)

                # 【重要】检查覆盖框是否会与待检测人员框重叠
                # 如果重叠，说明两个人员靠得太近，覆盖会影响待检测人员
                has_overlap = not (
                    overlay_x2 <= current_crop_x1 or  # 覆盖框在待检测人员左侧
                    overlay_x1 >= current_crop_x2 or  # 覆盖框在待检测人员右侧
                    overlay_y2 <= current_crop_y1 or  # 覆盖框在待检测人员上方
                    overlay_y1 >= current_crop_y2     # 覆盖框在待检测人员下方
                )

                if has_overlap:
                    # 如果会重叠，跳过覆盖
                    skipped_count += 1
                    print(f"    ⚠ 跳过覆盖: {other_identity} (bbox: {other_bbox}) - 与待检测人员重叠")
                    continue

                # 用半透明灰色覆盖
                # 创建一个半透明灰色层
                overlay = Image.new('RGBA', (overlay_x2 - overlay_x1, overlay_y2 - overlay_y1), (128, 128, 128, 200))

                # 将半透明层粘贴到裁剪图像上
                cropped_rgba = cropped.convert('RGBA')
                cropped_rgba.paste(overlay, (overlay_x1, overlay_y1), overlay)
                cropped = cropped_rgba.convert('RGB')

                covered_count += 1
                print(f"    → 已覆盖: {other_identity} (bbox: {other_bbox})")

        if covered_count > 0:
            print(f"    ✓ 共覆盖 {covered_count} 个其他人员")
        if skipped_count > 0:
            print(f"    ⚠ 跳过 {skipped_count} 个人员（与待检测人员重叠）")

    # 获取裁剪后的尺寸
    crop_width, crop_height = cropped.size

    # 计算缩放比例
    if crop_width < crop_height:
        # 以宽度为基准
        if crop_width < target_size:
            scale = target_size / crop_width
        else:
            scale = 1.0
    else:
        # 以高度为基准
        if crop_height < target_size:
            scale = target_size / crop_height
        else:
            scale = 1.0

    # 如果需要缩放
    if scale > 1.0:
        new_width = int(crop_width * scale)
        new_height = int(crop_height * scale)
        cropped = cropped.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # 创建目标尺寸的黑色背景
    final_width = max(cropped.width, target_size)
    final_height = max(cropped.height, target_size)

    # 如果尺寸不足，用黑边补全
    if cropped.width < final_width or cropped.height < final_height:
        black_bg = Image.new('RGB', (final_width, final_height), (0, 0, 0))

        # 计算居中位置
        paste_x = (final_width - cropped.width) // 2
        paste_y = (final_height - cropped.height) // 2

        black_bg.paste(cropped, (paste_x, paste_y))
        cropped = black_bg

    return cropped


def encode_pil_image(pil_image):
    """将PIL Image转换为base64编码"""
    import io
    from PIL import Image

    # 保存到字节流
    img_buffer = io.BytesIO()
    pil_image.save(img_buffer, format='JPEG')
    img_bytes = img_buffer.getvalue()

    # 转换为base64
    base64_data = base64.b64encode(img_bytes).decode('utf-8')

    return base64_data, 'image/jpeg'


@time_tracker.record_model_call("单人违规检测", AI_MODEL)
def detect_person_violations(client, person_image_base64, person_index, output_txt="person_violation.txt"):
    """
    检测单个人员的违规行为(睡觉、便服、玩手机)

    参数:
        client: OpenAI客户端
        person_image_base64: 人员图像的base64编码
        person_index: 人员索引(用于日志)
        output_txt: 输出文本文件路径

    返回:
        violations: 违规行为列表，如 ["玩手机", "便服"] 或 [] 表示无违规
        response_text: AI模型的完整响应文本
    """
    print(f"\n  检测人员{person_index}的违规行为...")

    data_url = f"data:image/jpeg;base64,{person_image_base64}"

    prompt = """请仔细观察图片中的人员，检测是否存在以下违规行为。

【重点关注】
请重点关注图中居中展示的工作人员，忽略背景中的其他人员。

开始回答前，请复述"严格按照下面的标准来判断，不要乱猜，不要用可能之类的语气"

【检测类型】:
1. **玩手机**: 正在使用手机，包括拿在手中看、操作、打电话等
   - **重要判断**: 请先检查桌面上是否有座机（固定电话）
   - 如果有座机，请检查：
     · 座机话筒是否在座机上（与座机连接）？
     · 座机话筒是否被拿在工作人员手中？
   - 以下情况也算玩手机：
     · 工作人员手持手机（正在看、操作或打电话）
     · 手机放在工作人员附近的桌面上（屏幕亮着或在播放）
     · 虽然看不清手部动作，但座机话筒明明在座机上（未在使用座机）
   - 以下情况不算玩手机：
     · 工作人员正在使用座机（话筒有连接线或无线连接座机）
     · 手机在桌面上但屏幕黑着（未使用状态）
     · 座机话筒丢失或不在座机上，无法判断是否在使用座机

2. **便服**: 穿着便服，不穿统一的正式服装
   - 正式服装包括：
     · 黑色、白色、粉色三种纯色衬衫
     · 纯深色上衣（深蓝、深灰、深棕等深色系上衣，单一颜色无图案）
     · 西服（正式西装）
     · 工作马甲（制服背心）
   - 便服特征：
     · 有明显图案装饰的衣服（条纹、格子、卡通、印花等）
     · 颜色鲜艳或花哨的衣服
     · 休闲风格（运动服、卫衣、T恤等）

3. **睡觉**: 明显的睡眠状态，休息不算，必须得趴在桌子上睡觉！
   - 头直接趴在桌子上
   - 或头枕着胳膊，胳膊放在桌子上（头也要枕在在桌子上）

【输出要求】:
1. 先详细描述图片中居中展示的工作人员的行为和状态
2. 在最后一行输出检测结果，格式为：违规：xxx,xxx 或 违规：无

示例输出:
工作人员正在xxxxxx，xxxx
违规：xxx
"""

    try:
        completion = client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
        )

        response_text = completion.choices[0].message.content.strip()

        print(f"    完整响应:\n{response_text}")

        # 解析结果：只读取最后一行
        violations = []
        lines = response_text.strip().split('\n')
        last_line = lines[-1].strip() if lines else ""

        # 从最后一行提取违规行为
        if last_line.startswith("违规："):
            violation_part = last_line[3:].strip()  # 去掉"违规："前缀
            if violation_part and violation_part != "无":
                # 分割逗号
                for v in violation_part.split(','):
                    v = v.strip()
                    if v and v != "无":
                        violations.append(v)

        print(f"    违规行为: {', '.join(violations) if violations else '无'}")

        return violations, response_text

    except Exception as e:
        print(f"    ✗ 检测失败: {e}")
        return [], ""


def find_nearest_customer(person_bbox, persons_data, image_width, image_height):
    """
    找到距离指定人员最近的顾客

    参数:
        person_bbox: 指定人员的bbox (归一化坐标)
        persons_data: 所有人员列表(包含身份信息)
        image_width: 图像宽度
        image_height: 图像高度

    返回:
        nearest_customer: 最近的顾客信息,如果没有顾客则返回None
    """
    # 计算指定人员的中心点
    px1, py1, px2, py2 = person_bbox
    person_center_x = (px1 + px2) / 2
    person_center_y = (py1 + py2) / 2

    # 映射到实际坐标
    person_cx = int(person_center_x / 1000 * image_width)
    person_cy = int(person_center_y / 1000 * image_height)

    nearest_customer = None
    min_distance = float('inf')

    # 遍历所有人员,找顾客
    for person in persons_data:
        identity = person.get('identity', '')
        if identity != '顾客':
            continue

        # 计算顾客的中心点
        customer_bbox = person.get('bbox', [0, 0, 0, 0])
        cx1, cy1, cx2, cy2 = customer_bbox
        customer_center_x = (cx1 + cx2) / 2
        customer_center_y = (cy1 + cy2) / 2

        # 映射到实际坐标
        customer_cx = int(customer_center_x / 1000 * image_width)
        customer_cy = int(customer_center_y / 1000 * image_height)

        # 计算距离(欧氏距离)
        distance = ((person_cx - customer_cx)**2 + (person_cy - customer_cy)**2)**0.5

        if distance < min_distance:
            min_distance = distance
            nearest_customer = person

    return nearest_customer


def crop_persons_with_context(image_path, person_bbox, customer_bbox, padding=100):
    """
    裁剪包含工作人员和顾客的区域

    参数:
        image_path: 原始图像路径
        person_bbox: 工作人员bbox (归一化坐标)
        customer_bbox: 顾客bbox (归一化坐标)
        padding: 扩展边距(像素)

    返回:
        cropped_image: 裁剪后的图像
    """
    img = Image.open(image_path)
    width, height = img.size

    # 映射到实际坐标
    px1 = int(person_bbox[0] / 1000 * width)
    py1 = int(person_bbox[1] / 1000 * height)
    px2 = int(person_bbox[2] / 1000 * width)
    py2 = int(person_bbox[3] / 1000 * height)

    cx1 = int(customer_bbox[0] / 1000 * width)
    cy1 = int(customer_bbox[1] / 1000 * height)
    cx2 = int(customer_bbox[2] / 1000 * width)
    cy2 = int(customer_bbox[3] / 1000 * height)

    # 计算包含两人的区域
    x1 = min(px1, cx1) - padding
    y1 = min(py1, cy1) - padding
    x2 = max(px2, cx2) + padding
    y2 = max(py2, cy2) + padding

    # 确保不超出图像范围
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    # 裁剪
    cropped = img.crop((x1, y1, x2, y2))

    return cropped


@time_tracker.record_model_call("顾客位置验证", AI_MODEL)
def verify_customer_opposite(client, person_customer_image_base64, output_txt="verify_opposite.txt"):
    """
    验证顾客是否位于工作人员对面

    参数:
        client: OpenAI客户端
        person_customer_image_base64: 包含工作人员和顾客的图像base64
        output_txt: 输出文本文件路径

    返回:
        is_opposite: True表示顾客在对面, False表示不在对面
        response_text: AI模型的完整响应文本
    """
    print(f"    二次验证:检查顾客是否在工作人员对面...")

    data_url = f"data:image/jpeg;base64,{person_customer_image_base64}"

    prompt = """请仔细观察图片，判断顾客是否位于玩手机工作人员的正对面。

【场景说明】:
- 工作人员位于柜台内侧（靠近摄像头），正在使用手机
- 顾客位于柜台外侧（远离摄像头）
- 重点是：判断顾客是否在"玩手机工作人员座位"正对面的柜台区域

【判断标准 - 重要】:
顾客只要在工作人员座位对面的区域即可，不一定要与工作人员面对面！

1. **座位对面的区域**
   - 以工作人员座位为参考，判断座位正前方的柜台外侧是否有顾客
   - 顾客应该在工作人员座位朝向的延长线区域内
   - **注意**: 不要求顾客和工作人员必须面对面（可能顾客在看别处，但位置是对的）

2. **柜台隔开**
   - 顾客和工作人员之间有明显的柜台隔开
   - 顾客在柜台外侧，工作人员在柜台内侧

3. **位置关系**
   - 顾客在工作人员座位前方（而不是侧面或后方）
   - 两者之间是服务关系，不是路过或其他关系

【判断示例】:
✓ 算"在对面":
  - 顾客站在工作人员座位正对面的柜台前（即使顾客在看别处）
  - 顾客在工作人员座位前方，在柜台区域内等待
  - 顾客在工作人员座位前方，与柜台有互动

✗ 不算"在对面":
  - 顾客在工作人员座位侧面或后方
  - 顾客在远处，不在工作人员座位前方区域
  - 顾客路过，不是在等待服务

【重要提示】:
- 重点关注：顾客是否在工作人员座位对面的柜台区域
- 不要求：顾客必须看着工作人员或面对面
- 如果画面中有多个顾客，只判断工作人员座位正对面的那位
- 如果工作人员前方没有人，输出"否"

【输出要求】:
只输出"是"或"否"。

- 是:顾客在工作人员座位对面的区域（不需要面对面）
- 否:顾客不在工作人员座位前方区域

请严格按照上述标准判断，只输出一个字。
"""

    try:
        completion = client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
        )

        response_text = completion.choices[0].message.content.strip()

        print(f"      验证结果: {response_text}")

        # 判断结果
        if "是" in response_text:
            return True, response_text
        else:
            return False, response_text

    except Exception as e:
        print(f"      ✗ 验证失败: {e}")
        return False, ""


def parse_json_output(text):
    """
    解析JSON输出,移除markdown标记并清理格式

    处理以下问题:
    1. 移除markdown代码块标记 (```json ... ```)
    2. 修复单引号为双引号
    3. 移除尾随逗号
    4. 移除注释
    5. 清理未转义的换行符
    """
    # 移除markdown标记
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "```json" or line.strip() == "```":
            text = "\n".join(lines[i + 1:])
            text = text.split("```")[0]
            break

    # 清理文本
    text = text.strip()

    # 尝试直接解析
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    # 如果直接解析失败,尝试修复常见问题

    # 1. 移除注释 (// 和 /* */)
    import re
    # 移除单行注释 // ...
    text = re.sub(r'//.*', '', text)
    # 移除多行注释 /* ... */
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)

    # 2. 修复单引号 - 将单引号键值对替换为双引号
    # 这个正则表达式会匹配类似 'key': 'value' 或 'key': "value"
    text = re.sub(r"'([^']+)'(\s*:)", r'"\1"\2', text)  # 键
    text = re.sub(r":\s*'([^']+)'", r': "\1"', text)  # 值

    # 3. 移除尾随逗号 (在 } 和 ] 之前的逗号)
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)

    # 4. 清理未转义的换行符(在字符串值中)
    # 简单处理:移除字符串值内部的换行
    text = re.sub(r'"\s*\n\s*"', '""', text)

    # 再次尝试解析
    try:
        # 验证JSON是否可解析
        json.loads(text)
        return text
    except json.JSONDecodeError as e:
        print(f"\n⚠ JSON解析失败,尝试清理后仍然无效")
        print(f"错误信息: {e}")
        print(f"原始文本前500字:")
        print(text[:500])
        raise


@time_tracker.record_model_call("步骤1设备检测", AI_MODEL)
def step1_detect_equipment(client, image_base64, image_path):
    """
    步骤1: 识别办公设备和工作台位置,输出坐标框

    返回: 设备列表和模型响应
        (
            [
                {
                    "type": "monitor" | "keyboard" | "desk",
                    "bbox": [x1, y1, x2, y2],
                    "center": [x, y]
                },
                ...
            ],
            "模型响应内容"
        )
    """
    print("\n" + "="*80)
    print("步骤1: 识别办公设备和工作台(显示器、键盘、办公桌)")
    print("="*80)

    data_url = f"data:image/jpeg;base64,{image_base64}"

    prompt = """请识别图片中所有办公设备和工作台的位置。

【设备定义】:
- 电脑显示器: 通常放在桌面上,有明显的屏幕
- 键盘: 电脑键盘,放在桌面上
- 办公桌: 工作人员的工作台,通常放有电脑等设备

【要求】:
- 识别图片中所有的办公设备(显示器、键盘)
- 识别所有的办公桌
- 不要遗漏任何可见的设备和办公桌
- 坐标范围: 0-999

【输出格式】:
每行一个设备,格式如下:
显示器 [x1, y1, x2, y2]
键盘 [x1, y1, x2, y2]
办公桌 [x1, y1, x2, y2]

示例:
显示器 [100, 200, 300, 400]
键盘 [120, 380, 280, 420]
办公桌 [100, 400, 500, 700]

注意:
- 请按从左到右的顺序输出
- 只输出坐标,不要有其他描述文字
"""

    completion = client.chat.completions.create(
        model=AI_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ],
    )

    response_text = completion.choices[0].message.content

    print("\n设备识别结果:")
    print("="*80)
    print(response_text[:100] if len(response_text) > 100 else response_text)
    if len(response_text) > 100:
        print("... (完整响应已保存到汇总文件)")
    print("="*80)

    # 从文本中提取设备信息
    equipment_list = extract_equipment_from_text(response_text)

    print(f"\n提取到 {len(equipment_list)} 个设备和办公桌:")
    for i, equip in enumerate(equipment_list):
        type_label = {
            'monitor': '显示器',
            'keyboard': '键盘',
            'desk': '办公桌'
        }.get(equip['type'], equip['type'])
        print(f"  {i+1}. {type_label} - bbox:{equip['bbox']}, center:{equip['center']}")

    return equipment_list, response_text  # 返回设备列表和响应文本


@time_tracker.record_model_call("步骤2人员检测", AI_MODEL)
def step2_detect_persons(client, image_base64, image_path):
    """
    步骤2: 识别人员位置,输出坐标框

    返回: 人员坐标框列表和模型响应
        (
            [
                {
                    "bbox": [x1, y1, x2, y2],
                    "description": "人员位置描述"
                },
                ...
            ],
            "模型响应内容"
        )
    """
    print("\n" + "="*80)
    print("步骤2: 识别人员位置(输出坐标框)")
    print("="*80)

    data_url = f"data:image/jpeg;base64,{image_base64}"

    prompt = """请识别图片中所有人员的位置。

【要求】:
- 识别图片中的所有人员(包括工作人员和顾客)
- 特别注意被严重截断、只露出一小部分的人员
- 如果只看到头部、肩膀或身体的一部分,也要识别并标注
- 不要遗漏任何人员
- 坐标范围: 0-999

【输出格式】:
每行一个人员,格式如下:
人员 [x1, y1, x2, y2]

示例:
人员 [100, 200, 250, 500]
人员 [300, 150, 400, 550]

注意:
- 请按从左到右、从上到下的顺序输出
- 只输出坐标,不要有其他描述文字
- 对于截断人员也要识别
"""

    completion = client.chat.completions.create(
        model=AI_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ],
    )

    response_text = completion.choices[0].message.content

    print("\n人员识别结果:")
    print("="*80)
    print(response_text[:100] if len(response_text) > 100 else response_text)
    if len(response_text) > 100:
        print("... (完整响应已保存到汇总文件)")
    print("="*80)

    # 从文本中提取人员信息
    persons_list = extract_persons_from_text(response_text)

    print(f"\n提取到 {len(persons_list)} 个人员:")
    for i, person in enumerate(persons_list):
        print(f"  人员{i+1}: bbox:{person['bbox']}")

    return persons_list, response_text  # 返回人员列表和响应文本


@time_tracker.record_model_call("步骤3缺岗检测", AI_MODEL)
def step3_detect_vacant_positions(client, image_base64, image_path, equipment_list,
                                  persons_list, image_width, image_height):
    """
    步骤3: 检测缺岗（无人工位）

    判断逻辑:
    1. 识别图中的无人工位
    2. 判断无人工位是否大部分位于设备连线与图片下边缘之间
    3. 双重检验：
       - 检查该位置是否在图像内
       - 检查与步骤2检测到的人体的距离
       - 如果距离大于一个人体长度 → 真的缺岗

    参数:
        client: OpenAI客户端
        image_base64: 图像base64
        image_path: 图像路径
        equipment_list: 设备列表
        persons_list: 人员列表
        image_width: 图像宽度
        image_height: 图像高度
        output_txt: 输出文件路径

    返回:
        vacant_positions: 缺岗位置列表
        [
            {
                "bbox": [x1, y1, x2, y2],
                "reason": "缺岗原因"
            },
            ...
        ]
    """
    print("\n" + "="*80)
    print("步骤3: 检测缺岗（无人工位）")
    print("="*80)

    data_url = f"data:image/jpeg;base64,{image_base64}"

    # 构建设备连线信息
    equipment_info = ""
    if equipment_list:
        equipment_info = "\n\n【参考信息】已识别到的办公设备:\n"
        for i, equip in enumerate(equipment_list):
            center = equip.get('center', [0, 0])
            equipment_info += f"{i+1}. {equip['type']} - 中心点:{center}\n"

    prompt = f"""请仔细观察图片，识别所有空着的工位（无人工位）。

【工位定义 - 重要】
工位是指完整的可供一个人办公的区域，请注意：

1. **与设备识别的关系**
   - 前面的步骤已经识别了独立的办公桌（desk）、显示器等设备
   - 这里的工位是比"办公桌"更大的概念
   - 工位 = 办公桌 + 显示器 + 键盘 + 椅子 + 其他设备的组合区域
   - 不要只识别单个办公桌，要识别整个办公区域

2. **整体性原则**
   - 一个工位可能包含多个桌子和设备
   - 可能是U型布局（多个桌子拼成一个环形或半环形）
   - 需要识别工位的整体区域范围，而不是单个桌子

3. **常见工位类型**
   - 直线型：长条形办公桌
   - L型：两个桌子垂直拼接
   - U型：三个或更多桌子围成环形/半环形
   - 综合型：多种形状组合的完整办公区域

4. **判断标准**
   - 有完整的办公设备组合（显示器、键盘、鼠标、椅子等）
   - 设备围绕形成一个完整的办公区域
   - 该区域当前没有工作人员

【特别提醒 - 避免误判】
- ❌ 不要把单独延伸出的小桌子、小台面识别为工位
- ❌ 不要把边角的小桌子识别为工位
- ❌ 不要把顾客等待区的椅子识别为工位
- ✅ 只识别明显完整的、可供一个人办公的区域

{equipment_info}
【输出格式】
每行一个空工位，格式如下：
空工位 [x1, y1, x2, y2]

示例：
空工位 [100, 200, 300, 400]
空工位 [500, 250, 650, 450]

【注意事项】
- 请按从左到右的顺序输出
- 坐标框要包围整个工位的所有设备
- 只输出坐标，不要有其他描述文字
- 如果没有空工位，输出"无"
"""

    try:
        completion = client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
        )

        response_text = completion.choices[0].message.content.strip()

        print("\n空工位识别结果:")
        print("="*80)
        print(response_text[:100] if len(response_text) > 100 else response_text)
        if len(response_text) > 100:
            print("... (完整响应已保存到汇总文件)")
        print("="*80)

        # 如果没有空工位
        if "无" in response_text or "空工位" not in response_text:
            print("\n未检测到空工位")
            return [], response_text, []  # 返回空列表、响应文本和空判断过程

        # 提取空工位的bbox
        vacant_bboxes = extract_bbox_from_text(response_text)

        if not vacant_bboxes:
            print("\n未能提取到空工位坐标")
            return [], response_text, []  # 返回空列表、响应文本和空判断过程

        print(f"\n提取到 {len(vacant_bboxes)} 个空工位")

        # 获取拟合直线参数（如果有设备）
        fitted_line = None
        if len(equipment_list) >= 2:
            # 收集设备中心点
            equipment_points = []
            for equip in equipment_list:
                center = equip.get('center', [0, 0])
                cx = int(center[0] / 1000 * image_width)
                cy = int(center[1] / 1000 * image_height)
                equipment_points.append([cx, cy])

            # 拟合直线
            equipment_points = np.array(equipment_points)
            x_coords = equipment_points[:, 0]
            y_coords = equipment_points[:, 1]
            coefficients = np.polyfit(x_coords, y_coords, 1)
            k = coefficients[0]
            b = coefficients[1]
            fitted_line = {'slope': k, 'intercept': b}

        # 双重检验每个空工位
        vacant_positions = []
        judgment_process = []  # 记录判断过程

        for i, bbox in enumerate(vacant_bboxes):
            print(f"\n空工位 {i+1}:")
            print(f"  坐标(归一化): {bbox}")

            # 记录这个空工位的判断过程
            process_text = f"\n空工位 {i+1}:\n  归一化坐标: {bbox}\n"

            # 映射到实际坐标
            x1 = int(bbox[0] / 1000 * image_width)
            y1 = int(bbox[1] / 1000 * image_height)
            x2 = int(bbox[2] / 1000 * image_width)
            y2 = int(bbox[3] / 1000 * image_height)

            process_text += f"  实际坐标: [{x1}, {y1}, {x2}, {y2}]\n"

            # 检验1: 判断是否在设备连线与图片下边缘之间
            # 使用采样法计算工位在直线下方的比例（与步骤4保持一致）
            is_work_area = False
            ratio_below = 0.0

            if fitted_line:
                # 在工位框内采样 10×10 = 100 个点
                sample_count_x = 10
                sample_count_y = 10
                points_below_line = 0
                total_points = 0

                k = fitted_line['slope']
                b = fitted_line['intercept']

                for sx in range(sample_count_x):
                    for sy in range(sample_count_y):
                        # 计算采样点的实际坐标
                        sample_x = x1 + (x2 - x1) * sx / (sample_count_x - 1)
                        sample_y = y1 + (y2 - y1) * sy / (sample_count_y - 1)

                        # 计算该x坐标对应的直线y值
                        line_y = k * sample_x + b

                        # 判断采样点是否在直线下方
                        if sample_y > line_y:
                            points_below_line += 1

                        total_points += 1

                # 计算在直线下方的比例
                ratio_below = points_below_line / total_points if total_points > 0 else 0

                process_text += f"  采样点总数: {total_points}\n"
                process_text += f"  直线下方点数: {points_below_line}\n"
                process_text += f"  直线下方比例: {ratio_below:.2%}\n"

                # 判断标准：工位至少30%在直线下方才算工作区域
                # 注意：这里用30%而不是50%，因为工位可能比人员大，且包含上方设备
                threshold = 0.30
                if ratio_below >= threshold:
                    is_work_area = True
                    print(f"  ✓ 位于设备连线与下边缘之间（工作区域）")
                    print(f"     直线下方比例: {ratio_below:.2%} >= {threshold:.0%}")
                    process_text += f"  ✓ 工作区域检验: {ratio_below:.2%} >= {threshold:.0%}，位于工作区域\n"
                else:
                    print(f"  ✗ 不在工作区域（可能是顾客区域）")
                    print(f"     直线下方比例: {ratio_below:.2%} < {threshold:.0%}")
                    process_text += f"  ✗ 工作区域检验: {ratio_below:.2%} < {threshold:.0%}，不在工作区域\n"
                    judgment_process.append(process_text)
                    continue

            # 检验2: 检查是否与已检测到的人员距离足够远
            is_far_enough = True
            min_distance = float('inf')

            for person in persons_list:
                person_bbox = person.get('bbox', [0, 0, 0, 0])
                px1 = int(person_bbox[0] / 1000 * image_width)
                py1 = int(person_bbox[1] / 1000 * image_height)
                px2 = int(person_bbox[2] / 1000 * image_width)
                py2 = int(person_bbox[3] / 1000 * image_height)

                # 计算人员中心点
                person_cx = (px1 + px2) / 2
                person_cy = (py1 + py2) / 2

                # 计算空工位中心点
                vacant_cx = (x1 + x2) / 2
                vacant_cy = (y1 + y2) / 2

                # 计算距离
                distance = ((person_cx - vacant_cx)**2 + (person_cy - vacant_cy)**2)**0.5
                min_distance = min(min_distance, distance)

            # 估算一个人体长度（假设平均身高约占图像高度的30%）
            avg_person_height = image_height * 0.3

            process_text += f"  距离最近人员: {min_distance:.0f}像素\n"
            process_text += f"  估算人体长度: {avg_person_height:.0f}像素 (图像高度的30%)\n"

            if min_distance < avg_person_height:
                is_far_enough = False
                print(f"  ✗ 距离最近人员{min_distance:.0f}像素，小于人体长度({avg_person_height:.0f}像素)，不算缺岗")
                process_text += f"  ✗ 距离检验: {min_distance:.0f} < {avg_person_height:.0f}，不算缺岗\n"
            else:
                print(f"  ✓ 距离最近人员{min_distance:.0f}像素，大于人体长度({avg_person_height:.0f}像素)")
                process_text += f"  ✓ 距离检验: {min_distance:.0f} > {avg_person_height:.0f}，距离足够远\n"

            # 通过双重检验，确认为缺岗
            if is_work_area and is_far_enough:
                vacant_positions.append({
                    "bbox": bbox,
                    "reason": "工作区域内且无人员"
                })
                print(f"  → 确认为缺岗")
                process_text += f"  → 结论: 通过双重检验，确认为缺岗\n"
            else:
                process_text += f"  → 结论: 未通过检验，不算缺岗\n"

            judgment_process.append(process_text)

        # 输出汇总
        print("\n缺岗检测汇总:")
        print("="*80)
        if vacant_positions:
            print(f"检测到 {len(vacant_positions)} 个缺岗:")
            for i, vp in enumerate(vacant_positions):
                print(f"  位置{i+1}: {vp['bbox']}")
        else:
            print("未检测到缺岗")
        print("="*80)

        return vacant_positions, response_text, judgment_process  # 返回缺岗列表、响应文本和判断过程

    except Exception as e:
        print(f"\n✗ 缺岗检测失败: {e}")
        import traceback
        print(traceback.format_exc())
        return [], "", []  # 返回空列表、空响应文本和空判断过程


@time_tracker.record_processing_step("步骤4身份计算")
def step4_calculate_identity(equipment_list, persons_list, image_width, image_height):
    """
    步骤4: 拟合直线,通过数学计算判断人员身份(工作人员/顾客)

    判断逻辑:
    1. 识别所有办公设备(显示器、键盘)和办公桌的中心点
    2. 使用线性回归拟合一条直线: y = kx + b
    3. 对于每个人员框,采样多个点,计算有多少比例的点位于直线下方
    4. 如果人员超过一定比例(如50%)的身体部分位于直线下方,则为工作人员,否则为顾客

    参数:
        equipment_list: 步骤1返回的设备列表
        persons_list: 步骤2返回的人员列表
        image_width: 图像实际宽度
        image_height: 图像实际高度

    返回: 包含身份信息的人员列表和拟合直线参数
    """
    print("\n" + "="*80)
    print("步骤3: 拟合直线判断人员身份")
    print("="*80)

    try:
        equipment_data = equipment_list
        persons_data = persons_list

        print(f"\n图像尺寸: {image_width} x {image_height}")
        print(f"\n检测到 {len(equipment_data)} 个设备和办公桌")

        # 收集用于拟合的设备中心点
        # 优先级: 显示器/键盘 > 桌子
        equipment_points = []
        fitting_equipment_type = None  # 记录拟合使用的设备类型

        # 先检查是否有显示器或键盘
        has_monitor_or_keyboard = False
        for equip in equipment_data:
            equip_type = equip.get('type', '')
            if equip_type in ['monitor', 'keyboard']:
                has_monitor_or_keyboard = True
                break

        # 根据是否有显示器/键盘来决定收集哪些设备
        if has_monitor_or_keyboard:
            # 有显示器或键盘: 只收集显示器和键盘，不收集桌子
            fitting_equipment_type = 'monitor_keyboard'
            print("\n检测到显示器或键盘，仅使用显示器和键盘进行拟合")
            for i, equip in enumerate(equipment_data):
                equip_type = equip.get('type', '')
                if equip_type in ['monitor', 'keyboard']:
                    center_norm = equip.get('center', [0, 0])
                    cx_norm, cy_norm = center_norm

                    # 映射到实际坐标
                    cx = int(cx_norm / 1000 * image_width)
                    cy = int(cy_norm / 1000 * image_height)

                    equipment_points.append([cx, cy])

                    type_label = {
                        'monitor': '显示器',
                        'keyboard': '键盘'
                    }.get(equip_type, equip_type)
                    print(f"  {len(equipment_points)}. {type_label}: 中心点({cx}, {cy})")
        else:
            # 没有显示器和键盘: 只收集桌子
            fitting_equipment_type = 'desk'
            print("\n未检测到显示器或键盘，使用桌子进行拟合")
            for i, equip in enumerate(equipment_data):
                equip_type = equip.get('type', '')
                if equip_type == 'desk':
                    center_norm = equip.get('center', [0, 0])
                    cx_norm, cy_norm = center_norm

                    # 映射到实际坐标
                    cx = int(cx_norm / 1000 * image_width)
                    cy = int(cy_norm / 1000 * image_height)

                    equipment_points.append([cx, cy])

                    type_label = '办公桌'
                    print(f"  {len(equipment_points)}. {type_label}: 中心点({cx}, {cy})")

        # 根据拟合设备类型确定阈值
        if fitting_equipment_type == 'monitor_keyboard':
            threshold = 0.5  # 显示器/键盘: 50%
            print(f"\n使用显示器/键盘拟合，判断阈值: {threshold:.0%}")
        elif fitting_equipment_type == 'desk':
            threshold = 0.3  # 桌子: 30%
            print(f"\n使用桌子拟合，判断阈值: {threshold:.0%}")
        else:
            threshold = 0.5  # 默认50%
            print(f"\n使用默认拟合，判断阈值: {threshold:.0%}")

        if len(equipment_points) < 2:
            print("\n⚠ 设备/办公桌数量不足,使用所有人员的bbox中点作为拟合点")
            # 使用所有人员的bbox中点作为拟合点
            for person in persons_data:
                bbox = person.get('bbox', [0, 0, 0, 0])
                x1, y1, x2, y2 = bbox

                # 映射到实际坐标
                px1 = int(x1 / 1000 * image_width)
                py1 = int(y1 / 1000 * image_height)
                px2 = int(x2 / 1000 * image_width)
                py2 = int(y2 / 1000 * image_height)

                # 计算中心点
                center_x = (px1 + px2) // 2
                center_y = (py1 + py2) // 2

                equipment_points.append([center_x, center_y])
                print(f"  人员中点: ({center_x}, {center_y})")

            if len(equipment_points) < 2:
                print("\n✗ 错误: 人员数量也不足,无法拟合直线")
                return {"fitted_line": {}, "persons": []}

        # 使用numpy进行线性回归拟合直线
        equipment_points = np.array(equipment_points)
        x_coords = equipment_points[:, 0]
        y_coords = equipment_points[:, 1]

        # 线性回归: y = kx + b
        # 使用numpy的polyfit进行一次多项式拟合
        coefficients = np.polyfit(x_coords, y_coords, 1)
        k = coefficients[0]  # 斜率
        b = coefficients[1]  # 截距

        print(f"\n拟合直线方程: y = {k:.4f}x + {b:.4f}")

        # 计算拟合误差(R²)
        y_pred = k * x_coords + b
        ss_res = np.sum((y_coords - y_pred) ** 2)  # 残差平方和
        ss_tot = np.sum((y_coords - np.mean(y_coords)) ** 2)  # 总平方和
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        print(f"拟合优度(R²): {r_squared:.4f}")

        # 判断每个人员的身份
        results = []
        judgment_process = []  # 记录判断过程

        for i, person in enumerate(persons_data):
            person_bbox_norm = person.get('bbox', [0, 0, 0, 0])
            px1_norm, py1_norm, px2_norm, py2_norm = person_bbox_norm

            # 映射到实际图像坐标
            px1 = int(px1_norm / 1000 * image_width)
            py1 = int(py1_norm / 1000 * image_height)
            px2 = int(px2_norm / 1000 * image_width)
            py2 = int(py2_norm / 1000 * image_height)

            print(f"\n人员 {i+1}:")
            print(f"  坐标(归一化): {person_bbox_norm}")
            print(f"  坐标(实际): [{px1}, {py1}, {px2}, {py2}]")

            # 在人员框内采样多个点,判断它们相对于直线的位置
            # 采样策略:在人员框内均匀采样10x10=100个点
            sample_count_x = 10
            sample_count_y = 10
            total_points = 0
            points_below_line = 0

            for sx in range(sample_count_x):
                for sy in range(sample_count_y):
                    # 计算采样点的实际坐标
                    sample_x = px1 + (px2 - px1) * sx / (sample_count_x - 1)
                    sample_y = py1 + (py2 - py1) * sy / (sample_count_y - 1)

                    # 计算该x坐标对应的直线y值
                    line_y = k * sample_x + b

                    # 判断采样点是否在直线下方
                    # 注意:图像坐标系y轴向下为正,所以y值越大表示越靠下
                    if sample_y > line_y:
                        points_below_line += 1

                    total_points += 1

            # 计算在直线下方的比例
            ratio_below = points_below_line / total_points if total_points > 0 else 0

            print(f"  采样点总数: {total_points}")
            print(f"  直线下方点数: {points_below_line}")
            print(f"  直线下方比例: {ratio_below:.2%}")
            print(f"  判断阈值: {threshold:.0%}")

            # 判断身份:使用动态阈值
            if ratio_below > threshold:
                identity = "工作人员"
                identity_en = "staff"
            else:
                identity = "顾客"
                identity_en = "customer"

            print(f"  判断结果: {identity}")

            # 记录判断过程
            process_text = (
                f"人员 {i+1}:\n"
                f"  - 归一化坐标: {person_bbox_norm}\n"
                f"  - 实际坐标: [{px1}, {py1}, {px2}, {py2}]\n"
                f"  - 在人员框内采样10x10=100个点\n"
                f"  - 直线下方点数: {points_below_line}/{total_points}\n"
                f"  - 直线下方比例: {ratio_below:.2%}\n"
                f"  - 判断标准: 比例>{threshold:.0%}为工作人员，否则为顾客\n"
                f"  - 判断结果: {identity}\n"
            )
            judgment_process.append(process_text)

            # 构建结果
            result = {
                "bbox": person_bbox_norm,  # 保持归一化坐标
                "identity": identity,
                "identity_en": identity_en,
                "description": person.get('description', ''),
                "ratio_below_line": round(ratio_below, 4),
                "reasoning": f"人员{ratio_below:.1%}的身体部分位于拟合直线下方,超过{threshold:.0%}阈值,判定为{identity}"
            }
            results.append(result)

        # 生成输出数据,包含直线参数和人员信息
        output_data = {
            "fitted_line": {
                "equation": f"y = {k:.4f}x + {b:.4f}",
                "slope": round(k, 4),
                "intercept": round(b, 4),
                "r_squared": round(r_squared, 4)
            },
            "persons": results,
            "judgment_process": judgment_process  # 添加判断过程
        }

        print("\n身份判断结果:")
        print("="*80)
        print(f"拟合直线: y = {k:.4f}x + {b:.4f}")
        print(f"拟合优度(R²): {r_squared:.4f}")
        for i, result in enumerate(results):
            print(f"人员{i+1}: {result['identity']} - 比例:{result['ratio_below_line']:.2%}")
        print("="*80)

        return output_data, judgment_process  # 返回数据和判断过程

    except Exception as e:
        print(f"\n✗ 身份判断失败: {e}")
        import traceback
        print(traceback.format_exc())
        return {"fitted_line": {}, "persons": [], "judgment_process": []}, []


@time_tracker.record_processing_step("步骤5绘制结果")
def step5_plot_results(image_path, equipment_list, identity_data,
                      output_path):
    """
    步骤5: 在原图上绘制所有坐标框、拟合直线和身份标签

    绘制内容:
    1. 设备框(显示器、键盘、办公桌)
       - 显示器: 紫色
       - 键盘: 不绘制（太小，避免拥挤）
       - 办公桌: 绿色
    2. 拟合直线(红色虚线)
    3. 人员矩形框,根据身份使用不同颜色:
       - 工作人员: 蓝色
       - 顾客: 橙色
    4. 标注身份文本

    参数:
        image_path: 原始图片路径
        equipment_list: 步骤1返回的设备列表
        identity_data: 步骤3返回的身份判断数据(包含fitted_line和persons)
        output_path: 输出图片路径
    """
    print("\n" + "="*80)
    print("步骤5: 在原图上绘制设备、拟合直线和人员框")
    print("="*80)

    try:
        # 加载图像
        img = Image.open(image_path)
        width, height = img.size
        print(f"图像尺寸: {img.size}")

        draw = ImageDraw.Draw(img)

        # 尝试加载中文字体
        try:
            font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=25)
            font_small = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=20)
            font_violation = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=35)  # 违规标签字体（更大）
        except:
            try:
                font = ImageFont.truetype("msyh.ttc", size=25)  # Windows微软雅黑
                font_small = ImageFont.truetype("msyh.ttc", size=20)
                font_violation = ImageFont.truetype("msyh.ttc", size=35)  # 违规标签字体（更大）
            except:
                font = ImageFont.load_default()
                font_small = ImageFont.load_default()
                font_violation = ImageFont.load_default()

        # 获取设备和人员数据
        equipment_data = equipment_list
        fitted_line = identity_data.get('fitted_line', {})
        persons_data = identity_data.get('persons', [])
        vacant_positions = identity_data.get('vacant_positions', [])

        # 定义设备类型对应的颜色和标签
        device_config = {
            "monitor": {
                "color": "purple",
                "label": "显示器"
            },
            "keyboard": {
                "color": "cyan",  # 青色（虽然配置了，但后面会跳过不绘制）
                "label": "键盘"
            },
            "desk": {
                "color": "green",
                "label": "办公桌"
            }
        }

        print(f"\n绘制{len(equipment_data)}个设备框:")

        # 绘制每个设备框
        for i, equipment in enumerate(equipment_data):
            device_type = equipment.get('type', 'unknown')
            bbox = equipment.get('bbox', [0, 0, 0, 0])
            x1, y1, x2, y2 = bbox

            # 跳过键盘（太小，绘制会让画面拥挤）
            if device_type == 'keyboard':
                continue

            # 映射到实际坐标
            abs_x1 = int(x1 / 1000 * width)
            abs_y1 = int(y1 / 1000 * height)
            abs_x2 = int(x2 / 1000 * width)
            abs_y2 = int(y2 / 1000 * height)

            # 获取配置
            config = device_config.get(device_type, {"color": "gray", "label": "设备"})
            color = config["color"]
            label = config["label"]

            print(f"\n设备 {i+1} ({label}):")
            print(f"  坐标(归一化): {bbox}")
            print(f"  坐标(实际): [{abs_x1}, {abs_y1}, {abs_x2}, {abs_y2}]")

            # 绘制设备框
            draw.rectangle(
                ((abs_x1, abs_y1), (abs_x2, abs_y2)),
                outline=color,
                width=3
            )

            # 添加设备标签
            draw.text((abs_x1 + 5, abs_y1 + 5), label, fill=color, font=font_small)

        # 绘制拟合直线
        if fitted_line:
            slope = fitted_line.get('slope', 0)
            intercept = fitted_line.get('intercept', 0)

            print(f"\n绘制拟合直线:")
            print(f"  方程: y = {slope}x + {intercept}")

            # 计算直线的两个端点(从图像左边缘到右边缘)
            # 左端点 (x=0)
            y_left = slope * 0 + intercept
            # 右端点 (x=width)
            y_right = slope * width + intercept

            # 确保端点在图像范围内
            points = []
            if 0 <= y_left <= height:
                points.append((0, int(y_left)))
            if 0 <= y_right <= height:
                points.append((width, int(y_right)))

            # 如果只有一个点在范围内,计算上边缘或下边缘的交点
            if len(points) == 1:
                if y_left < 0:  # 直线从上方进入图像
                    x_top = -intercept / slope if slope != 0 else 0
                    if 0 <= x_top <= width:
                        points.append((int(x_top), 0))
                elif y_left > height:  # 直线从下方进入图像
                    x_bottom = (height - intercept) / slope if slope != 0 else 0
                    if 0 <= x_bottom <= width:
                        points.append((int(x_bottom), height))

            # 绘制直线(使用点划线模拟)
            if len(points) >= 2:
                # 绘制多条线段模拟虚线效果
                for i in range(0, 100, 2):
                    t1 = i / 100
                    t2 = (i + 1) / 100
                    x1_line = points[0][0] + (points[1][0] - points[0][0]) * t1
                    y1_line = points[0][1] + (points[1][1] - points[0][1]) * t1
                    x2_line = points[0][0] + (points[1][0] - points[0][0]) * t2
                    y2_line = points[0][1] + (points[1][1] - points[0][1]) * t2

                    draw.line(
                        [(x1_line, y1_line), (x2_line, y2_line)],
                        fill="red",
                        width=3
                    )

                # 添加直线标签(放在直线中间)
                mid_x = (points[0][0] + points[1][0]) // 2
                mid_y = (points[0][1] + points[1][1]) // 2
                line_label = "拟合直线"
                draw.text((mid_x + 10, mid_y - 30), line_label, fill="red", font=font_small)

                print(f"  直线端点: {points}")

        print(f"\n绘制所有人员框（工作人员和顾客）:")

        # 定义身份对应的颜色
        identity_colors = {
            "工作人员": "blue",
            "顾客": "orange"
        }

        # 绘制所有人员的框
        staff_count = 0
        customer_count = 0
        for i, person in enumerate(persons_data):
            bbox = person.get('bbox', [0, 0, 0, 0])
            identity = person.get('identity', '未知')
            px1, py1, px2, py2 = bbox

            # 统计
            if identity == '工作人员':
                staff_count += 1
                person_label = f"工作人员{staff_count}"
            elif identity == '顾客':
                customer_count += 1
                person_label = f"顾客{customer_count}"
            else:
                person_label = f"未知{i+1}"

            # 映射到实际坐标
            abs_px1 = int(px1 / 1000 * width)
            abs_py1 = int(py1 / 1000 * height)
            abs_px2 = int(px2 / 1000 * width)
            abs_py2 = int(py2 / 1000 * height)

            # 确保坐标正确
            if abs_px1 > abs_px2:
                abs_px1, abs_px2 = abs_px2, abs_px1
            if abs_py1 > abs_py2:
                abs_py1, abs_py2 = abs_py2, abs_py1

            # 获取颜色
            color = identity_colors.get(identity, "gray")

            print(f"\n人员 {i+1} ({person_label}):")
            print(f"  坐标(归一化): {bbox}")
            print(f"  坐标(实际): [{abs_px1}, {abs_py1}, {abs_px2}, {abs_py2}]")
            print(f"  身份: {identity}")
            print(f"  颜色: {color}")

            # 绘制人员框
            draw.rectangle(
                ((abs_px1, abs_py1), (abs_px2, abs_py2)),
                outline=color,
                width=4
            )

            # 绘制身份标签（在框上方）
            label_text = person_label
            text_bbox = draw.textbbox((0, 0), label_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # 标签位置:框上方居中
            label_x = abs_px1 + (abs_px2 - abs_px1) // 2 - text_width // 2
            label_y = abs_py1 - text_height - 8

            # 绘制标签背景
            draw.rectangle(
                [(label_x - 4, label_y - 2),
                 (label_x + text_width + 4, label_y + text_height + 2)],
                fill=color,
                outline="white",
                width=2
            )

            # 绘制标签文字
            draw.text((label_x, label_y), label_text, fill="white", font=font)

        # 绘制缺岗位置
        if vacant_positions:
            print(f"\n绘制{len(vacant_positions)}个缺岗位置:")

            for i, vacant in enumerate(vacant_positions):
                bbox = vacant.get('bbox', [0, 0, 0, 0])
                vx1, vy1, vx2, vy2 = bbox

                # 映射到实际坐标
                abs_vx1 = int(vx1 / 1000 * width)
                abs_vy1 = int(vy1 / 1000 * height)
                abs_vx2 = int(vx2 / 1000 * width)
                abs_vy2 = int(vy2 / 1000 * height)

                # 确保坐标正确
                if abs_vx1 > abs_vx2:
                    abs_vx1, abs_vx2 = abs_vx2, abs_vx1
                if abs_vy1 > abs_vy2:
                    abs_vy1, abs_vy2 = abs_vy2, abs_vy1

                print(f"\n缺岗位置 {i+1}:")
                print(f"  坐标(归一化): {bbox}")
                print(f"  坐标(实际): [{abs_vx1}, {abs_vy1}, {abs_vx2}, {abs_vy2}]")

                # 绘制缺岗框（红色虚线效果）
                # 使用多条线段模拟虚线
                dash_length = 20
                gap_length = 10

                # 上边
                for x in range(abs_vx1, abs_vx2, dash_length + gap_length):
                    x_end = min(x + dash_length, abs_vx2)
                    draw.line([(x, abs_vy1), (x_end, abs_vy1)], fill="red", width=3)

                # 下边
                for x in range(abs_vx1, abs_vx2, dash_length + gap_length):
                    x_end = min(x + dash_length, abs_vx2)
                    draw.line([(x, abs_vy2), (x_end, abs_vy2)], fill="red", width=3)

                # 左边
                for y in range(abs_vy1, abs_vy2, dash_length + gap_length):
                    y_end = min(y + dash_length, abs_vy2)
                    draw.line([(abs_vx1, y), (abs_vx1, y_end)], fill="red", width=3)

                # 右边
                for y in range(abs_vy1, abs_vy2, dash_length + gap_length):
                    y_end = min(y + dash_length, abs_vy2)
                    draw.line([(abs_vx2, y), (abs_vx2, y_end)], fill="red", width=3)

                # 添加缺岗标签
                label_text = "缺岗"
                text_bbox = draw.textbbox((0, 0), label_text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]

                # 标签位置:框上方居中
                label_x = abs_vx1 + (abs_vx2 - abs_vx1) // 2 - text_width // 2
                label_y = abs_vy1 - text_height - 8

                # 绘制标签背景
                draw.rectangle(
                    [(label_x - 4, label_y - 2),
                     (label_x + text_width + 4, label_y + text_height + 2)],
                    fill="red",
                    outline="white",
                    width=2
                )

                # 绘制标签文字
                draw.text((label_x, label_y), label_text, fill="white", font=font)

        # 保存结果图像
        img.save(output_path)
        print(f"\n✓ 检测结果已保存到: {output_path}")

        return True

    except Exception as e:
        print(f"\n✗ 绘制失败: {e}")
        import traceback
        print(traceback.format_exc())
        return False


@time_tracker.record_processing_step("步骤9绘制违规")
def step9_plot_violations(image_path, equipment_list, identity_data, output_path):
    """
    步骤9: 绘制人员框和违规行为标签（在原图基础上）

    参数:
        image_path: 原始图片路径
        equipment_list: 步骤1返回的设备列表
        identity_data: 包含违规信息的数据
        output_path: 输出图片路径

    绘制流程:
        1. 加载原始图像
        2. 绘制工作人员框
        3. 绘制违规行为标签
        4. 保存图像
    """
    try:
        # 加载原始图像
        img = Image.open(image_path)
        width, height = img.size
        print(f"图像尺寸: {img.size}")

        draw = ImageDraw.Draw(img)

        # 打印图像尺寸信息
        print(f"图像宽度: {width}, 高度: {height}")

        # 尝试加载中文字体（字体改小）
        try:
            font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=25)
            font_small = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=20)
            font_violation = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=20)  # 违规标签字体（改小）
        except:
            try:
                font = ImageFont.truetype("msyh.ttc", size=25)
                font_small = ImageFont.truetype("msyh.ttc", size=20)
                font_violation = ImageFont.truetype("msyh.ttc", size=20)  # 违规标签字体（改小）
            except:
                font = ImageFont.load_default()
                font_small = ImageFont.load_default()
                font_violation = ImageFont.load_default()

        # 获取违规数据和人员数据
        violations = identity_data.get('violations', [])

        # 定义违规类型对应的颜色
        violation_colors = {
            "玩手机": "red",
            "便服": "yellow",
            "不穿工服": "yellow",
            "睡觉": "orange",
            "聊天": "purple"
        }

        # 绘制违规行为（只绘制有违规的人员）
        print(f"\n绘制违规行为:")

        if not violations:
            print("  ⚠ 没有违规行为需要绘制")
        else:
            for violation in violations:
                person_index = violation.get('person_index', 0)
                bbox = violation.get('bbox', [0, 0, 0, 0])
                violation_types = violation.get('violations', [])

                if not violation_types:
                    continue

                # 映射到实际坐标（与步骤5保持一致）
                x1 = int(bbox[0] / 1000 * width)
                y1 = int(bbox[1] / 1000 * height)
                x2 = int(bbox[2] / 1000 * width)
                y2 = int(bbox[3] / 1000 * height)

                # 确保坐标正确
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1

                print(f"\n违规人员 {person_index}:")
                print(f"  bbox(归一化): {bbox}")
                print(f"  坐标(实际): [{x1}, {y1}, {x2}, {y2}]")
                print(f"  违规行为: {', '.join(violation_types)}")

                # 绘制违规人员框（蓝色）
                draw.rectangle(
                    ((x1, y1), (x2, y2)),
                    outline="blue",
                    width=3
                )
                print(f"  ✓ 绘制违规人员框: [{x1}, {y1}, {x2}, {y2}]")

                # 对每个违规类型绘制标签
                for i, v_type in enumerate(violation_types):
                    label_text = v_type
                    text_bbox = draw.textbbox((0, 0), label_text, font=font_violation)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]

                    # 标签绘制在人员框内（左上角），确保不重叠
                    label_x = x1 + 5
                    label_y = y1 + 10 + i * (text_height + 12)  # 间距适当

                    print(f"    标签{i+1}: '{v_type}' 在 ({label_x}, {label_y})")

                    # 获取违规类型对应的颜色
                    color = violation_colors.get(v_type, "red")

                    # 绘制标签背景和文字
                    draw.rectangle([(label_x - 2, label_y - 2), (label_x + text_width + 2, label_y + text_height + 2)],
                                 fill=color, outline="white", width=1)
                    draw.text((label_x, label_y), label_text, fill="white", font=font_violation)

        # 绘制聊天连线
        chat_pairs = identity_data.get('chat_pairs', [])
        persons_data = identity_data.get('persons', [])

        if chat_pairs:
            print(f"\n绘制聊天连线:")
            for pair in chat_pairs:
                p1_idx = pair.get('person1_index', 0) - 1
                p2_idx = pair.get('person2_index', 0) - 1

                if 0 <= p1_idx < len(persons_data) and 0 <= p2_idx < len(persons_data):
                    p1_bbox = persons_data[p1_idx].get('bbox', [0, 0, 0, 0])
                    p2_bbox = persons_data[p2_idx].get('bbox', [0, 0, 0, 0])

                    # 计算中心点
                    p1_cx = int((p1_bbox[0] + p1_bbox[2]) / 2000 * width)
                    p1_cy = int((p1_bbox[1] + p1_bbox[3]) / 2000 * height)
                    p2_cx = int((p2_bbox[0] + p2_bbox[2]) / 2000 * width)
                    p2_cy = int((p2_bbox[1] + p2_bbox[3]) / 2000 * height)

                    print(f"  工作人员{pair.get('person1_index', 0)} <-> 工作人员{pair.get('person2_index', 0)}")

                    # 绘制紫色连线
                    draw.line([(p1_cx, p1_cy), (p2_cx, p2_cy)], fill="purple", width=3)

                    # 在连线中点绘制"聊天"标签
                    mid_x = (p1_cx + p2_cx) // 2
                    mid_y = (p1_cy + p2_cy) // 2

                    label_text = "聊天"
                    text_bbox = draw.textbbox((0, 0), label_text, font=font_violation)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]

                    label_x = mid_x - text_width // 2
                    label_y = mid_y - text_height // 2

                    # 绘制标签背景
                    draw.rectangle([(label_x - 3, label_y - 2), (label_x + text_width + 3, label_y + text_height + 2)],
                                 fill="purple", outline="white", width=2)
                    draw.text((label_x, label_y), label_text, fill="white", font=font_violation)

        # 保存图像
        print(f"\n=== 保存图像 ===")
        print(f"保存路径: {output_path}")
        img.save(output_path)
        print(f"✓ 步骤9图像已保存: {os.path.basename(output_path)}")
        print(f"  包含: 违规人员框 + 违规行为标签 + 聊天连线")
        print(f"  违规标签字体大小: 20px")
        print("="*80)

        return True

    except Exception as e:
        print(f"✗ 绘制违规标签失败: {e}")
        import traceback
        print(traceback.format_exc())
        return False


@time_tracker.record_processing_step("步骤6绘制含违规")
def step6_plot_results_with_violations(image_path, equipment_list, identity_data, output_path):
    """
    步骤6的增强版: 绘制设备、拟合直线、人员框和违规行为

    参数:
        image_path: 原始图片路径
        equipment_list: 步骤1返回的设备列表
        identity_data: 包含身份、违规、聊天信息的数据
        output_path: 输出图片路径

    绘制流程:
        1. 绘制基础内容（设备、拟合直线、人员框、缺岗）
        2. 添加违规行为标签
        3. 添加聊天连线
        4. 保存最终图像
    """
    try:
        # 加载原始图像
        img = Image.open(image_path)
        width, height = img.size
        draw = ImageDraw.Draw(img)

        # 尝试加载中文字体
        try:
            font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=25)
            font_small = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=20)
            font_violation = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=35)  # 违规标签字体（更大）
        except:
            try:
                font = ImageFont.truetype("msyh.ttc", size=25)
                font_small = ImageFont.truetype("msyh.ttc", size=20)
                font_violation = ImageFont.truetype("msyh.ttc", size=35)  # 违规标签字体（更大）
            except:
                font = ImageFont.load_default()
                font_small = ImageFont.load_default()
                font_violation = ImageFont.load_default()

        # ========== 第1部分：绘制基础内容 ==========
        print(f"\n绘制基础内容（设备、拟合直线、人员框、缺岗）:")

        # 获取数据
        equipment_data = equipment_list
        fitted_line = identity_data.get('fitted_line', {})
        persons_data = identity_data.get('persons', [])
        vacant_positions = identity_data.get('vacant_positions', [])

        # 定义设备类型对应的颜色和标签
        device_config = {
            "monitor": {"color": "purple", "label": "显示器"},
            "keyboard": {"color": "cyan", "label": "键盘"},
            "desk": {"color": "green", "label": "办公桌"}
        }

        # 1. 绘制设备框
        print(f"\n绘制{len(equipment_data)}个设备框:")
        for i, equipment in enumerate(equipment_data):
            device_type = equipment.get('type', 'unknown')
            bbox = equipment.get('bbox', [0, 0, 0, 0])
            x1, y1, x2, y2 = bbox

            # 映射到实际坐标
            abs_x1 = int(x1 / 1000 * width)
            abs_y1 = int(y1 / 1000 * height)
            abs_x2 = int(x2 / 1000 * width)
            abs_y2 = int(y2 / 1000 * height)

            config = device_config.get(device_type, {"color": "gray", "label": "设备"})
            color = config["color"]
            label = config["label"]

            draw.rectangle(((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=3)
            draw.text((abs_x1 + 5, abs_y1 + 5), label, fill=color, font=font_small)

        # 2. 绘制拟合直线
        if fitted_line:
            slope = fitted_line.get('slope', 0)
            intercept = fitted_line.get('intercept', 0)

            y_left = slope * 0 + intercept
            y_right = slope * width + intercept

            points = []
            if 0 <= y_left <= height:
                points.append((0, int(y_left)))
            if 0 <= y_right <= height:
                points.append((width, int(y_right)))

            if len(points) == 1:
                if y_left < 0:
                    x_top = -intercept / slope if slope != 0 else 0
                    if 0 <= x_top <= width:
                        points.append((int(x_top), 0))
                elif y_left > height:
                    x_bottom = (height - intercept) / slope if slope != 0 else 0
                    if 0 <= x_bottom <= width:
                        points.append((int(x_bottom), height))

            if len(points) >= 2:
                for i in range(0, 100, 2):
                    t1 = i / 100
                    t2 = (i + 1) / 100
                    x1_line = points[0][0] + (points[1][0] - points[0][0]) * t1
                    y1_line = points[0][1] + (points[1][1] - points[0][1]) * t1
                    x2_line = points[0][0] + (points[1][0] - points[0][0]) * t2
                    y2_line = points[0][1] + (points[1][1] - points[0][1]) * t2

                    draw.line([(x1_line, y1_line), (x2_line, y2_line)], fill="red", width=3)

                mid_x = (points[0][0] + points[1][0]) // 2
                mid_y = (points[0][1] + points[1][1]) // 2
                draw.text((mid_x + 10, mid_y - 30), "拟合直线", fill="red", font=font_small)

        # 3. 绘制人员框（只绘制工作人员）
        print(f"\n绘制人员框（仅工作人员）:")
        staff_count = 0
        for i, person in enumerate(persons_data):
            bbox = person.get('bbox', [0, 0, 0, 0])
            identity = person.get('identity', '未知')

            if identity != '工作人员':
                continue

            staff_count += 1

            px1, py1, px2, py2 = bbox
            abs_px1 = int(px1 / 1000 * width)
            abs_py1 = int(py1 / 1000 * height)
            abs_px2 = int(px2 / 1000 * width)
            abs_py2 = int(py2 / 1000 * height)

            if abs_px1 > abs_px2:
                abs_px1, abs_px2 = abs_px2, abs_px1
            if abs_py1 > abs_py2:
                abs_py1, abs_py2 = abs_py2, abs_py1

            draw.rectangle(((abs_px1, abs_py1), (abs_px2, abs_py2)), outline="blue", width=4)
            # 不绘制"工作人员"文字

        # 4. 绘制缺岗位置
        if vacant_positions:
            print(f"\n绘制{len(vacant_positions)}个缺岗位置:")
            for i, vacant in enumerate(vacant_positions):
                bbox = vacant.get('bbox', [0, 0, 0, 0])
                vx1, vy1, vx2, vy2 = bbox

                abs_vx1 = int(vx1 / 1000 * width)
                abs_vy1 = int(vy1 / 1000 * height)
                abs_vx2 = int(vx2 / 1000 * width)
                abs_vy2 = int(vy2 / 1000 * height)

                if abs_vx1 > abs_vx2:
                    abs_vx1, abs_vx2 = abs_vx2, abs_vx1
                if abs_vy1 > abs_vy2:
                    abs_vy1, abs_vy2 = abs_vy2, abs_vy1

                # 绘制红色虚线框
                dash_length = 20
                gap_length = 10

                for x in range(abs_vx1, abs_vx2, dash_length + gap_length):
                    x_end = min(x + dash_length, abs_vx2)
                    draw.line([(x, abs_vy1), (x_end, abs_vy1)], fill="red", width=3)

                for x in range(abs_vx1, abs_vx2, dash_length + gap_length):
                    x_end = min(x + dash_length, abs_vx2)
                    draw.line([(x, abs_vy2), (x_end, abs_vy2)], fill="red", width=3)

                for y in range(abs_vy1, abs_vy2, dash_length + gap_length):
                    y_end = min(y + dash_length, abs_vy2)
                    draw.line([(abs_vx1, y), (abs_vx1, y_end)], fill="red", width=3)

                for y in range(abs_vy1, abs_vy2, dash_length + gap_length):
                    y_end = min(y + dash_length, abs_vy2)
                    draw.line([(abs_vx2, y), (abs_vx2, y_end)], fill="red", width=3)

                # 绘制"缺岗"标签
                label_text = "缺岗"
                text_bbox = draw.textbbox((0, 0), label_text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]

                label_x = abs_vx1 + (abs_vx2 - abs_vx1) // 2 - text_width // 2
                label_y = abs_vy1 - text_height - 8

                draw.rectangle([(label_x - 4, label_y - 2), (label_x + text_width + 4, label_y + text_height + 2)],
                             fill="red", outline="white", width=2)
                draw.text((label_x, label_y), label_text, fill="white", font=font)

        # ========== 第2部分：绘制违规行为标签 ==========
        violations = identity_data.get('violations', [])
        chat_pairs = identity_data.get('chat_pairs', [])

        # 准备调试信息（用于保存到txt）
        debug_log = []

        debug_log.append("=== 违规行为绘制调试信息 ===")
        debug_log.append(f"违规结果数量: {len(violations)}")
        debug_log.append(f"聊天对数量: {len(chat_pairs)}")

        # 定义违规类型对应的颜色
        violation_colors = {
            "玩手机": "red",
            "便服": "yellow",
            "不穿工服": "yellow",
            "睡觉": "orange",
            "聊天": "purple"
        }

        # 绘制违规标签
        print(f"\n绘制{len(violations)}个违规行为标签:")

        debug_log.append("\n绘制违规行为标注:")
        if not violations:
            debug_log.append("  ⚠ 没有违规行为需要绘制")
            print("  ⚠ 没有违规行为需要绘制")
        else:
            debug_log.append(f"  ✓ 准备绘制 {len(violations)} 个人员的违规行为")

        for violation in violations:
            person_index = violation.get('person_index', 0)
            bbox = violation.get('bbox', [0, 0, 0, 0])
            violation_types = violation.get('violations', [])

            if not violation_types:
                continue

            # 映射到实际坐标
            x1 = int(bbox[0] / 1000 * width)
            y1 = int(bbox[1] / 1000 * height)
            x2 = int(bbox[2] / 1000 * width)
            y2 = int(bbox[3] / 1000 * height)

            # 确保坐标正确
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1

            print(f"\n违规人员 {person_index}:")
            print(f"  坐标(归一化): {bbox}")
            print(f"  坐标(实际): [{x1}, {y1}, {x2}, {y2}]")
            print(f"  违规行为: {', '.join(violation_types)}")

            debug_log.append(f"\n  处理人员 {person_index}:")
            debug_log.append(f"    bbox: {bbox}")
            debug_log.append(f"    violations: {violation_types}")
            debug_log.append(f"    实际坐标: ({x1}, {y1}) -> ({x2}, {y2})")

            # 对每个违规类型绘制标签（使用更大的字体）
            for i, v_type in enumerate(violation_types):
                color = violation_colors.get(v_type, "red")

                label_text = v_type
                text_bbox = draw.textbbox((0, 0), label_text, font=font_violation)  # 使用更大的字体
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]

                label_x = x1 + 5
                label_y = y2 + 10 + i * (text_height + 8)

                print(f"    标签{i+1}: '{v_type}' 在 ({label_x}, {label_y}), 颜色: {color}")

                debug_log.append(f"      绘制标签 '{v_type}' 在 ({label_x}, {label_y}), 颜色: {color}, 字体大小: 35")

                # 绘制标签背景和文字
                draw.rectangle([(label_x - 3, label_y - 2), (label_x + text_width + 3, label_y + text_height + 2)],
                             fill=color, outline="white", width=2)
                draw.text((label_x, label_y), label_text, fill="white", font=font_violation)  # 使用更大的字体

        # ========== 第3部分：绘制聊天连线 ==========
        debug_log.append("\n绘制聊天行为标注:")
        for pair in chat_pairs:
            p1_idx = pair.get('person1_index', 0) - 1
            p2_idx = pair.get('person2_index', 0) - 1

            if 0 <= p1_idx < len(persons_data) and 0 <= p2_idx < len(persons_data):
                p1_bbox = persons_data[p1_idx].get('bbox', [0, 0, 0, 0])
                p2_bbox = persons_data[p2_idx].get('bbox', [0, 0, 0, 0])

                p1_cx = int((p1_bbox[0] + p1_bbox[2]) / 2000 * width)
                p1_cy = int((p1_bbox[1] + p1_bbox[3]) / 2000 * height)
                p2_cx = int((p2_bbox[0] + p2_bbox[2]) / 2000 * width)
                p2_cy = int((p2_bbox[1] + p2_bbox[3]) / 2000 * height)

                debug_log.append(f"  工作人员{pair.get('person1_index', 0)} <-> 工作人员{pair.get('person2_index', 0)}")

                draw.line([(p1_cx, p1_cy), (p2_cx, p2_cy)], fill="purple", width=3)

                mid_x = (p1_cx + p2_cx) // 2
                mid_y = (p1_cy + p2_cy) // 2

                label_text = "聊天"
                text_bbox = draw.textbbox((0, 0), label_text, font=font_violation)  # 使用更大的字体
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]

                label_x = mid_x - text_width // 2
                label_y = mid_y - text_height // 2

                draw.rectangle([(label_x - 3, label_y - 2), (label_x + text_width + 3, label_y + text_height + 2)],
                             fill="purple", outline="white", width=2)
                draw.text((label_x, label_y), label_text, fill="white", font=font_violation)  # 使用更大的字体

        if not violations and not chat_pairs:
            debug_log.append("  无违规行为和聊天行为")

        # ========== 第4部分：保存调试日志到txt ==========
        debug_log_path = output_path.replace(os.path.splitext(output_path)[1], "_调试日志.txt")
        try:
            with open(debug_log_path, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("违规行为绘制调试日志\n")
                f.write("="*80 + "\n\n")
                for line in debug_log:
                    f.write(line + "\n")
            print(f"✓ 调试日志已保存: {os.path.basename(debug_log_path)}")
        except Exception as e:
            print(f"⚠ 保存调试日志失败: {e}")

        # ========== 第5部分：保存最终图像 ==========
        print(f"\n=== 保存最终图像 ===")
        print(f"保存路径: {output_path}")
        img.save(output_path)
        print(f"✓ 最终图像已保存: {os.path.basename(output_path)}")
        print(f"  包含: 设备框 + 拟合直线 + 人员框 + 缺岗 + 违规标签 + 聊天连线")
        print(f"  违规标签字体大小: 35px")
        print("="*80)

        return True

    except Exception as e:
        print(f"✗ 绘制违规标注失败: {e}")
        import traceback
        print(traceback.format_exc())
        return False


def save_summary_results(filepath, equipment_list, persons_list, identity_data,
                        violation_results, chat_pairs, vacant_positions, img_width, img_height,
                        all_model_responses=None, all_judgment_processes=None):
    """
    保存总的检测结果到一个txt文件（包含完整的模型响应和判断过程）

    参数:
        filepath: 输出文件路径
        equipment_list: 设备列表
        persons_list: 人员列表
        identity_data: 身份判断数据
        violation_results: 违规结果
        chat_pairs: 聊天对
        vacant_positions: 缺岗位置列表
        img_width: 图像宽度
        img_height: 图像高度
        all_model_responses: 所有模型响应的字典
        all_judgment_processes: 所有判断过程的字典
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            # ═══════════════════════════════════════════════════════════════════════════════
            # 文件头部
            # ═══════════════════════════════════════════════════════════════════════════════
            f.write("="*80 + "\n")
            f.write("员工行为检测结果汇总报告\n")
            f.write("="*80 + "\n\n")

            # 基本信息
            f.write("【基本信息】\n")
            f.write(f"图像尺寸: {img_width} x {img_height}\n")
            from datetime import datetime
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # ═══════════════════════════════════════════════════════════════════════════════
            # 第一部分：检测摘要
            # ═══════════════════════════════════════════════════════════════════════════════
            f.write("\n" + "="*80 + "\n")
            f.write("第一部分：检测结果摘要\n")
            f.write("="*80 + "\n")
            f.write("说明：本部分汇总了所有检测步骤的核心结果\n\n")

            # 1. 设备识别结果
            f.write("┌─ 设备和工作台识别\n")
            f.write("├" + "─"*78 + "\n")
            f.write(f"│ 检测到 {len(equipment_list)} 个办公设备和工作台:\n")
            for i, equip in enumerate(equipment_list):
                type_label = {
                    'monitor': '显示器',
                    'keyboard': '键盘',
                    'desk': '办公桌'
                }.get(equip['type'], equip['type'])
                bbox_str = str(equip['bbox'])
                f.write(f"│   {i+1}. {type_label} - 位置:{bbox_str}\n")
            f.write("└" + "─"*78 + "\n\n")

            # 2. 人员识别与身份判断结果
            f.write("┌─ 人员识别与身份判断\n")
            f.write("├" + "─"*78 + "\n")
            persons = identity_data.get('persons', [])

            # 统计工作人员和顾客
            staff_count = sum(1 for p in persons if p.get('identity') == '工作人员')
            customer_count = sum(1 for p in persons if p.get('identity') == '顾客')

            f.write(f"│ 总人数: {len(persons)} (工作人员: {staff_count}, 顾客: {customer_count})\n")
            f.write("│\n")

            for i, person in enumerate(persons):
                identity = person.get('identity', '未知')
                bbox = person.get('bbox', [0, 0, 0, 0])
                ratio = person.get('ratio_below_line', 0)

                # 统一编号
                if identity == '工作人员':
                    person_num = sum(1 for p in persons[:i] if p.get('identity') == '工作人员') + 1
                    label = f"工作人员{person_num}"
                elif identity == '顾客':
                    person_num = sum(1 for p in persons[:i] if p.get('identity') == '顾客') + 1
                    label = f"顾客{person_num}"
                else:
                    label = f"未知{i+1}"

                f.write(f"│ {i+1}. {label:<15} - 位置:{str(bbox):<40} (线下比例:{ratio:.1%})\n")

            f.write("└" + "─"*78 + "\n\n")

            # 3. 拟合直线
            fitted_line = identity_data.get('fitted_line', {})
            if fitted_line:
                f.write("┌─ 拟合直线信息\n")
                f.write("├" + "─"*78 + "\n")
                f.write(f"│ 方程: {fitted_line.get('equation', 'N/A')}\n")
                f.write(f"│ 拟合优度(R²): {fitted_line.get('r_squared', 'N/A')}\n")
                f.write(f"│ 使用的中心点数量: {fitted_line.get('num_points', 'N/A')}\n")
                f.write("└" + "─"*78 + "\n\n")

            # 4. 缺岗检测结果
            f.write("┌─ 缺岗检测\n")
            f.write("├" + "─"*78 + "\n")
            if not vacant_positions:
                f.write("│ 未检测到缺岗\n")
            else:
                f.write(f"│ 检测到 {len(vacant_positions)} 个缺岗位置:\n")
                for i, vp in enumerate(vacant_positions):
                    bbox_str = str(vp['bbox'])
                    reason = vp['reason']
                    f.write(f"│   {i+1}. 位置:{bbox_str:<45} - 原因:{reason}\n")
            f.write("└" + "─"*78 + "\n\n")

            # 5. 违规检测结果
            f.write("┌─ 违规行为检测\n")
            f.write("├" + "─"*78 + "\n")
            if not violation_results:
                f.write("│ 未检测到违规行为\n")
            else:
                # 统计违规类型
                violation_count = 0
                for result in violation_results:
                    violations = result.get('violations', [])
                    violation_count += len(violations)

                f.write(f"│ 检测到 {len(violation_results)} 名工作人员, 共 {violation_count} 项违规:\n")
                f.write("│\n")

                for result in violation_results:
                    person_idx = result.get('person_index', 0)
                    bbox = result.get('bbox', [0, 0, 0, 0])
                    violations = result.get('violations', [])

                    if violations:
                        for v in violations:
                            f.write(f"│   ⚠ 工作人员{person_idx} - {v}\n")
                    else:
                        f.write(f"│   ✓ 工作人员{person_idx} - 无违规\n")
            f.write("└" + "─"*78 + "\n\n")

            # 6. 聊天检测结果
            f.write("┌─ 聊天行为检测\n")
            f.write("├" + "─"*78 + "\n")
            if not chat_pairs:
                f.write("│ 未检测到聊天行为\n")
            else:
                f.write(f"│ 检测到 {len(chat_pairs)} 组聊天:\n")
                for pair in chat_pairs:
                    p1 = pair.get('person1_index', 0)
                    p2 = pair.get('person2_index', 0)
                    f.write(f"│   • 工作人员{p1} ↔ 工作人员{p2}\n")
            f.write("└" + "─"*78 + "\n\n")

            # ═══════════════════════════════════════════════════════════════════════════════
            # 第二部分：详细数据
            # ═══════════════════════════════════════════════════════════════════════════════
            f.write("\n" + "="*80 + "\n")
            f.write("第二部分：详细检测数据\n")
            f.write("="*80 + "\n")
            f.write("说明：本部分记录了检测过程中的详细数据，便于数据分析和追溯\n\n")

            f.write("• 人员原始识别数据（步骤2）:\n")
            for i, person in enumerate(persons_list):
                f.write(f"  {i+1}. bbox:{person['bbox']}\n")
            f.write("\n")

            # ═══════════════════════════════════════════════════════════════════════════════
            # 第三部分：AI模型响应
            # ═══════════════════════════════════════════════════════════════════════════════
            if all_model_responses:
                f.write("\n" + "="*80 + "\n")
                f.write("第三部分：AI模型响应\n")
                f.write("="*80 + "\n")
                f.write("说明：以下记录了所有步骤的AI模型原始响应，便于调试和审查\n\n")

                # 按步骤顺序组织
                step_order = ['步骤1_设备识别', '步骤2_人员识别', '步骤3_缺岗检测',
                             '步骤7_违规检测', '步骤8_聊天检测']

                for step_name in step_order:
                    if step_name in all_model_responses:
                        f.write(f"\n{'─'*80}\n")
                        f.write(f"【{step_name}】\n")
                        f.write(f"{'─'*80}\n\n")
                        response_text = all_model_responses[step_name]
                        if response_text:
                            # 限制输出长度，避免文件过大
                            if len(response_text) > 3000:
                                f.write(response_text[:3000])
                                f.write(f"\n\n... (响应过长，已截断。完整长度: {len(response_text)} 字符) ...\n")
                            else:
                                f.write(response_text)
                        else:
                            f.write("(无响应)\n")
                        f.write("\n")

            # ═══════════════════════════════════════════════════════════════════════════════
            # 第四部分：判断过程
            # ═══════════════════════════════════════════════════════════════════════════════
            if all_judgment_processes:
                f.write("\n" + "="*80 + "\n")
                f.write("第四部分：详细判断过程\n")
                f.write("="*80 + "\n")
                f.write("说明：以下记录了每个步骤的详细判断过程，便于理解逻辑和追踪问题\n\n")

                # 步骤3: 缺岗检测判断过程
                if '步骤3_缺岗检测' in all_judgment_processes:
                    f.write(f"\n{'─'*80}\n")
                    f.write("【步骤3：缺岗检测】判断过程\n")
                    f.write(f"{'─'*80}\n\n")
                    step3_process = all_judgment_processes['步骤3_缺岗检测']
                    if step3_process:
                        # 如果是列表，逐行输出
                        if isinstance(step3_process, list):
                            for process_text in step3_process:
                                f.write(f"  {process_text}")
                        else:
                            f.write(f"  {step3_process}")
                    else:
                        f.write("  (无判断过程记录)\n")
                    f.write("\n")

                # 步骤4: 身份判断过程
                if '步骤4_身份判断' in all_judgment_processes:
                    f.write(f"\n{'─'*80}\n")
                    f.write("【步骤4：人员身份判断】判断过程\n")
                    f.write(f"{'─'*80}\n\n")

                    # 先输出拟合直线信息
                    fitted_line = identity_data.get('fitted_line', {})
                    if fitted_line:
                        f.write(f"  拟合直线方程: {fitted_line.get('equation', 'N/A')}\n")
                        f.write(f"  拟合优度(R²): {fitted_line.get('r_squared', 'N/A')}\n")
                        f.write(f"  使用的中心点数量: {fitted_line.get('num_points', 'N/A')}\n")
                        f.write("\n")

                    step4_process = all_judgment_processes['步骤4_身份判断']
                    if step4_process:
                        if isinstance(step4_process, list):
                            for process_text in step4_process:
                                f.write(f"  {process_text}")
                        else:
                            f.write(f"  {step4_process}")
                    else:
                        f.write("  (无判断过程记录)\n")
                    f.write("\n")

                # 步骤7: 违规检测判断过程
                if '步骤7_违规检测' in all_judgment_processes:
                    f.write(f"\n{'─'*80}\n")
                    f.write("【步骤7：违规行为检测】判断过程\n")
                    f.write(f"{'─'*80}\n")
                    step7_process = all_judgment_processes['步骤7_违规检测']
                    if step7_process:
                        if isinstance(step7_process, list):
                            for process_text in step7_process:
                                f.write(f"  {process_text}")
                        else:
                            f.write(f"  {step7_process}")
                    else:
                        f.write("  (无判断过程记录)\n")
                    f.write("\n")

                # 步骤8: 聊天检测判断过程
                if '步骤8_聊天检测' in all_judgment_processes:
                    f.write(f"\n{'─'*80}\n")
                    f.write("【步骤8：聊天行为检测】判断过程\n")
                    f.write(f"{'─'*80}\n")
                    step8_process = all_judgment_processes['步骤8_聊天检测']
                    if step8_process:
                        if isinstance(step8_process, list):
                            for process_text in step8_process:
                                f.write(f"  {process_text}")
                        else:
                            f.write(f"  {step8_process}")
                    else:
                        f.write("  (无判断过程记录)\n")
                    f.write("\n")

            f.write("="*80 + "\n")

        print(f"✓ 检测结果汇总已保存: {os.path.basename(filepath)}")

    except Exception as e:
        print(f"✗ 保存检测结果汇总失败: {e}")


def step7_detect_staff_violations(client, image_path, identity_data, image_width, image_height,
                                  output_folder, base_name):
    """
    步骤7: 检测工作人员违规行为(睡觉、便服、玩手机)

    参数:
        client: OpenAI客户端
        image_path: 原始图像路径
        identity_data: 步骤3返回的身份判断数据
        image_width: 图像宽度
        image_height: 图像高度
        output_folder: 输出文件夹
        base_name: 文件基础名

    返回:
        violation_results: 每个工作人员的违规结果
        [
            {
                "person_index": 1,
                "bbox": [x1, y1, x2, y2],
                "violations": ["玩手机", "便服"]
            },
            ...
        ]
        model_responses: AI模型响应列表
    """
    print("\n" + "="*80)
    print("步骤7: 检测工作人员违规行为")
    print("="*80)

    persons_data = identity_data.get('persons', [])

    # 用于收集所有模型响应和判断过程
    model_responses = []
    judgment_processes = []

    # 筛选出工作人员
    staff_list = []
    for i, person in enumerate(persons_data):
        if person.get('identity') == '工作人员':
            staff_list.append({
                'index': i + 1,
                'bbox': person.get('bbox', [0, 0, 0, 0]),
                'identity': '工作人员'
            })

    if not staff_list:
        print("\n没有检测到工作人员,跳过违规检测")
        judgment_processes.append("没有检测到工作人员,跳过违规检测\n")
        return [], [], judgment_processes

    print(f"\n检测到 {len(staff_list)} 个工作人员,开始逐个检测违规行为...")

    violation_results = []

    # 对每个工作人员进行检测
    for staff in staff_list:
        person_index = staff['index']
        person_bbox = staff['bbox']

        print(f"\n{'='*60}")
        print(f"工作人员 {person_index}")
        print(f"{'='*60}")

        # 裁剪并调整人员图像（传入所有人员列表，用于覆盖其他人员）
        person_image = crop_and_resize_person(image_path, person_bbox, target_size=640, all_persons=persons_data)

        # 编码为base64
        person_image_base64, _ = encode_pil_image(person_image)

        # 保存裁剪后的人员图像(用于调试)
        person_crop_path = os.path.join(output_folder, f"{base_name}_人员{person_index}_裁剪.jpg")
        person_image.save(person_crop_path)
        print(f"  ✓ 裁剪图像已保存: {os.path.basename(person_crop_path)}")

        # 检测违规行为
        violations, response_text = detect_person_violations(client, person_image_base64, person_index)

        # 收集模型响应和判断过程
        if response_text:
            model_responses.append(f"人员{person_index}违规检测:\n{response_text}\n")
            judgment_processes.append(f"人员{person_index} - 违规检测结果: {response_text}\n")

        # 如果检测到玩手机,进行二次验证
        if '玩手机' in violations:
            print(f"    → 检测到玩手机,检查是否有顾客...")

            # 记录判断过程
            judgment_processes.append(f"人员{person_index} - 检测到玩手机,开始验证...\n")

            # 检查是否有顾客
            has_customer = any(p.get('identity') == '顾客' for p in persons_data)

            if not has_customer:
                print(f"    → 画面中无顾客,玩手机违规成立")
                judgment_processes.append(f"人员{person_index} - 画面中无顾客,玩手机违规成立\n")
            else:
                print(f"    → 画面中有顾客,需要验证是否在对面...")
                judgment_processes.append(f"人员{person_index} - 画面中有顾客,需要验证位置关系\n")

                # 找最近的顾客
                nearest_customer = find_nearest_customer(person_bbox, persons_data, image_width, image_height)

                if nearest_customer:
                    customer_bbox = nearest_customer.get('bbox', [0, 0, 0, 0])
                    print(f"    → 找到最近的顾客")
                    judgment_processes.append(f"人员{person_index} - 找到最近的顾客,开始验证是否在对面\n")

                    # 裁剪包含工作人员和顾客的区域
                    person_customer_image = crop_persons_with_context(image_path, person_bbox, customer_bbox)

                    # 编码为base64
                    person_customer_base64, _ = encode_pil_image(person_customer_image)

                    # 保存裁剪后的双人图像(用于调试)
                    dual_crop_path = os.path.join(output_folder, f"{base_name}_人员{person_index}_与顾客_裁剪.jpg")
                    person_customer_image.save(dual_crop_path)
                    print(f"    ✓ 双人裁剪图像已保存: {os.path.basename(dual_crop_path)}")

                    # 验证顾客是否在对面
                    is_opposite, verify_response = verify_customer_opposite(client, person_customer_base64)

                    # 收集模型响应和判断过程
                    if verify_response:
                        model_responses.append(f"人员{person_index}顾客位置验证:\n{verify_response}\n")
                        judgment_processes.append(f"人员{person_index} - 顾客位置验证结果: {verify_response}\n")

                    if is_opposite:
                        print(f"    → 顾客在对面,工作人员不算玩手机!")
                        judgment_processes.append(f"人员{person_index} - 顾客在对面,玩手机违规不成立\n")
                        # 移除玩手机违规
                        violations = [v for v in violations if v != '玩手机']
                    else:
                        print(f"    → 顾客不在对面,玩手机违规成立")
                        judgment_processes.append(f"人员{person_index} - 顾客不在对面,玩手机违规成立\n")
                else:
                    print(f"    → 未找到顾客,玩手机违规成立")
                    judgment_processes.append(f"人员{person_index} - 未找到顾客,玩手机违规成立\n")

        # 记录最终结果
        result = {
            "person_index": person_index,
            "bbox": person_bbox,
            "violations": violations
        }
        violation_results.append(result)

        # 记录最终判断
        if violations:
            judgment_processes.append(f"人员{person_index} - 最终结果: 检测到违规 - {', '.join(violations)}\n")
        else:
            judgment_processes.append(f"人员{person_index} - 最终结果: 未检测到违规行为\n")

        # 输出结果
        if violations:
            print(f"\n  最终结果: 检测到违规 - {', '.join(violations)}")
        else:
            print(f"\n  最终结果: 未检测到违规行为")

    # 输出汇总
    print("\n" + "="*80)
    print("违规检测汇总:")
    print("="*80)

    violation_count = 0
    for result in violation_results:
        if result['violations']:
            print(f"  工作人员{result['person_index']}: {', '.join(result['violations'])}")
            violation_count += len(result['violations'])

    if violation_count == 0:
        print("  未检测到任何违规行为")
    else:
        print(f"\n总计: {violation_count} 项违规")

    print("="*80)

    return violation_results, model_responses, judgment_processes


@time_tracker.record_model_call("步骤8聊天检测", AI_MODEL)
def step8_detect_staff_chat(client, image_path, identity_data, image_width, image_height,
                            output_folder, base_name):
    """
    步骤8: 检测工作人员之间是否存在聊天行为

    参数:
        client: OpenAI客户端
        image_path: 原始图像路径
        identity_data: 步骤3返回的身份判断数据
        image_width: 图像宽度
        image_height: 图像高度
        output_folder: 输出文件夹
        base_name: 文件基础名

    返回:
        chat_pairs: 聊天的人员对列表
        [
            {
                "person1_index": 1,
                "person2_index": 2,
                "bbox": [x1, y1, x2, y2]  # 包含两人的区域
            },
            ...
        ]
        model_response: AI模型的完整响应文本
        judgment_process: 判断推理过程
    """
    print("\n" + "="*80)
    print("步骤8: 检测工作人员聊天行为")
    print("="*80)

    persons_data = identity_data.get('persons', [])

    # 用于收集判断过程
    judgment_process = []

    # 筛选出工作人员
    staff_list = []
    for i, person in enumerate(persons_data):
        if person.get('identity') == '工作人员':
            staff_list.append({
                'index': i + 1,
                'bbox': person.get('bbox', [0, 0, 0, 0])
            })

    if len(staff_list) < 2:
        print("\n工作人员数量少于2人,无法检测聊天行为")
        judgment_process.append("工作人员数量少于2人,跳过聊天检测\n")
        return [], "", '\n'.join(judgment_process)

    print(f"\n检测到 {len(staff_list)} 个工作人员,准备裁剪组合图像...")

    # 计算包含所有工作人员的区域
    all_bboxes = [s['bbox'] for s in staff_list]

    # 找到所有bbox的边界
    min_x = min(bbox[0] for bbox in all_bboxes)
    min_y = min(bbox[1] for bbox in all_bboxes)
    max_x = max(bbox[2] for bbox in all_bboxes)
    max_y = max(bbox[3] for bbox in all_bboxes)

    # 映射到实际坐标
    actual_min_x = int(min_x / 1000 * image_width)
    actual_min_y = int(min_y / 1000 * image_height)
    actual_max_x = int(max_x / 1000 * image_width)
    actual_max_y = int(max_y / 1000 * image_height)

    # 裁剪包含所有工作人员的区域
    img = Image.open(image_path)
    staff_combined_image = img.crop((actual_min_x, actual_min_y, actual_max_x, actual_max_y))

    # 获取裁剪后的尺寸
    crop_width, crop_height = staff_combined_image.size

    # 计算缩放比例，保持长宽比，最小边补齐到640
    target_size = 640
    if crop_width < crop_height:
        if crop_width < target_size:
            scale = target_size / crop_width
        else:
            scale = 1.0
    else:
        if crop_height < target_size:
            scale = target_size / crop_height
        else:
            scale = 1.0

    # 如果需要缩放
    if scale > 1.0:
        new_width = int(crop_width * scale)
        new_height = int(crop_height * scale)
        staff_combined_image = staff_combined_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # 如果尺寸不是正方形，用黑边补全为正方形
    final_size = max(staff_combined_image.width, staff_combined_image.height, target_size)
    if staff_combined_image.width != final_size or staff_combined_image.height != final_size:
        black_bg = Image.new('RGB', (final_size, final_size), (0, 0, 0))

        # 居中粘贴
        paste_x = (final_size - staff_combined_image.width) // 2
        paste_y = (final_size - staff_combined_image.height) // 2

        black_bg.paste(staff_combined_image, (paste_x, paste_y))
        staff_combined_image = black_bg

    # 保存组合图像
    staff_combined_path = os.path.join(output_folder, f"{base_name}_工作人员组合_聊天检测.jpg")
    staff_combined_image.save(staff_combined_path)
    print(f"✓ 工作人员组合图像已保存: {os.path.basename(staff_combined_path)}")
    print(f"  图像尺寸: {staff_combined_image.size}")

    # 编码为base64
    staff_combined_base64, _ = encode_pil_image(staff_combined_image)

    # 调用模型检测聊天
    print(f"\n检测工作人员之间的聊天行为...")

    data_url = f"data:image/jpeg;base64,{staff_combined_base64}"

    prompt = """请仔细观察图片中的工作人员,检测是否存在聊天行为。

【聊天行为特征】:
1. 两个或多个工作人员面对面或侧面对面
2. 有眼神交流或口型动作
3. 身体朝向对方
4. 明显的交流姿态

【输出要求】:
只输出检测到的聊天人员对，格式如下：
人员1和人员2在聊天
人员3和人员4在聊天

如果没有检测到聊天行为，输出"无"。

请严格按照上述格式输出。
"""

    try:
        completion = client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
        )

        response_text = completion.choices[0].message.content.strip()

        print(f"检测结果: {response_text}")

        # 记录判断过程
        judgment_process.append(f"AI模型响应: {response_text}\n")

        # 解析结果
        chat_pairs = []
        if "无" not in response_text and "聊天" in response_text:
            # 使用正则表达式提取人员编号
            import re
            pattern = r'人员(\d+).*?人员(\d+).*?聊天'
            matches = re.findall(pattern, response_text)

            for match in matches:
                p1 = int(match[0])
                p2 = int(match[1])

                # 验证这两个人员都是工作人员
                if p1 <= len(staff_list) and p2 <= len(staff_list):
                    chat_pairs.append({
                        "person1_index": p1,
                        "person2_index": p2
                    })
                    judgment_process.append(f"检测到聊天: 工作人员{p1} <-> 工作人员{p2}\n")

            if not matches:
                judgment_process.append("未检测到明确的聊天对\n")
        else:
            judgment_process.append("未检测到聊天行为\n")

        # 输出结果
        if chat_pairs:
            print(f"\n检测到 {len(chat_pairs)} 组聊天:")
            for pair in chat_pairs:
                print(f"  工作人员{pair['person1_index']} <-> 工作人员{pair['person2_index']}")
        else:
            print(f"\n未检测到聊天行为")

        return chat_pairs, response_text, '\n'.join(judgment_process)

    except Exception as e:
        print(f"\n✗ 检测聊天失败: {e}")
        return [], "", ""


def process_image_new(image_path, output_folder, client):
    """
    处理单张图片的完整流程(新方法)

    参数:
        image_path: 输入图片路径
        output_folder: 输出文件夹
        client: OpenAI客户端

    返回:
        success: 是否成功
    """
    print("\n" + "╔" + "="*78 + "╗")
    image_filename = os.path.basename(image_path)
    print(f"║  开始处理: {image_filename}")
    print("╚" + "="*78 + "╝")

    # 开始计时
    time_tracker.start_image(image_filename)

    try:
        # 编码图像
        print(f"\n正在读取图像...")
        image_base64, mime_type = encode_image(image_path)
        print(f"图像已编码: {len(image_base64)} 字符, 类型: {mime_type}")

        # 获取图像尺寸
        with Image.open(image_path) as img:
            img_width, img_height = img.size
        print(f"图像尺寸: {img_width} x {img_height}")

        # 构造输出文件名
        base_name = os.path.splitext(image_filename)[0]

        # 收集所有模型响应和判断过程
        all_model_responses = {}
        all_judgment_processes = {}  # 存储判断过程

        # 步骤1: 识别办公设备和工作台
        print("\n" + "█"*80)
        print("步骤1: 识别办公设备和工作台(显示器、键盘、办公桌)")
        print("█"*80)
        equipment_list, step1_response = step1_detect_equipment(client, image_base64, image_path)
        all_model_responses['步骤1_设备识别'] = step1_response

        # 步骤2: 识别人员位置
        print("\n" + "█"*80)
        print("步骤2: 识别人员位置")
        print("█"*80)
        persons_list, step2_response = step2_detect_persons(client, image_base64, image_path)
        all_model_responses['步骤2_人员识别'] = step2_response

        # 步骤3: 检测缺岗（无人工位）
        print("\n" + "█"*80)
        print("步骤3: 检测缺岗（无人工位）")
        print("█"*80)
        vacant_positions, step3_response, step3_judgment = step3_detect_vacant_positions(
            client, image_base64, image_path, equipment_list,
            persons_list, img_width, img_height
        )
        all_model_responses['步骤3_缺岗检测'] = step3_response
        all_judgment_processes['步骤3_缺岗检测'] = step3_judgment

        # 步骤4: 拟合直线判断人员身份
        print("\n" + "█"*80)
        print("步骤4: 拟合直线判断人员身份")
        print("█"*80)
        identity_data, step4_judgment = step4_calculate_identity(
            equipment_list, persons_list, img_width, img_height
        )
        all_judgment_processes['步骤4_身份判断'] = step4_judgment

        # 步骤5: 绘制设备、拟合直线和人员框
        print("\n" + "█"*80)
        print("步骤5: 绘制设备、拟合直线和人员框")
        print("█"*80)
        ext = os.path.splitext(image_filename)[1].lower()
        output_filename = f"{base_name}_结果_基础标注.{ext.split('.')[1]}"
        output_path = os.path.join(output_folder, output_filename)

        # 将缺岗结果添加到identity_data中
        identity_data_with_vacant = identity_data.copy()
        identity_data_with_vacant['vacant_positions'] = vacant_positions

        success = step5_plot_results(
            image_path, equipment_list, identity_data_with_vacant, output_path
        )

        if not success:
            print(f"\n✗ {image_filename} 处理失败(绘制步骤)")
            return False

        # 步骤6: 添加缺岗标注（已在步骤5中完成）
        print("\n" + "█"*80)
        print("步骤6: 缺岗标注已添加")
        print("█"*80)

        # 步骤7: 检测工作人员违规行为
        print("\n" + "█"*80)
        print("步骤7: 检测工作人员违规行为")
        print("█"*80)
        violation_results, step7_responses, step7_judgments = step7_detect_staff_violations(
            client, image_path, identity_data, img_width, img_height,
            output_folder, base_name
        )

        # 收集步骤7的模型响应和判断过程
        if step7_responses:
            all_model_responses['步骤7_违规检测'] = '\n'.join(step7_responses)
        if step7_judgments:
            all_judgment_processes['步骤7_违规检测'] = step7_judgments

        # 调试：打印违规检测结果
        print(f"\n=== 步骤7完成，违规检测结果汇总 ===")
        print(f"返回的违规结果数量: {len(violation_results)}")
        for i, vr in enumerate(violation_results):
            print(f"  结果{i+1}:")
            print(f"    人员索引: {vr.get('person_index', 'N/A')}")
            print(f"    bbox: {vr.get('bbox', 'N/A')}")
            print(f"    违规行为: {vr.get('violations', [])}")
        print("="*50)

        # 将违规结果添加到identity_data中
        identity_data_with_violations = identity_data.copy()
        identity_data_with_violations['violations'] = violation_results

        # 步骤8: 检测工作人员聊天行为
        print("\n" + "█"*80)
        print("步骤8: 检测工作人员聊天行为")
        print("█"*80)
        chat_pairs, step8_response, step8_judgment = step8_detect_staff_chat(
            client, image_path, identity_data, img_width, img_height,
            output_folder, base_name
        )

        # 收集步骤8的模型响应和判断过程
        if step8_response:
            all_model_responses['步骤8_聊天检测'] = step8_response
        if step8_judgment:
            all_judgment_processes['步骤8_聊天检测'] = [step8_judgment]

        # 调试：打印聊天检测结果
        print(f"\n=== 步骤8完成，聊天检测结果汇总 ===")
        print(f"返回的聊天对数量: {len(chat_pairs)}")
        for i, cp in enumerate(chat_pairs):
            print(f"  聊天对{i+1}: 人员{cp.get('person1_index', 'N/A')} <-> 人员{cp.get('person2_index', 'N/A')}")
        print("="*50)

        # 将聊天检测结果添加到identity_data_with_violations中，以便绘制
        identity_data_with_violations['chat_pairs'] = chat_pairs

        # 步骤9: 绘制人员框和违规行为标签
        print("\n" + "█"*80)
        print("步骤9: 绘制人员框和违规行为标签")
        print("█"*80)
        output_filename_step9 = f"{base_name}_结果_人员框和标签.{ext.split('.')[1]}"
        output_path_step9 = os.path.join(output_folder, output_filename_step9)

        success = step9_plot_violations(
            image_path, equipment_list, identity_data_with_violations, output_path_step9
        )

        if not success:
            print(f"\n✗ 步骤9绘制失败")
            return False

        # 移除原来的步骤8调用（已经提前执行）

        # 保存总的检测结果到一个txt文件（包含完整的模型响应和判断过程）
        summary_txt_path = os.path.join(output_folder, f"{base_name}_检测结果汇总.txt")
        save_summary_results(summary_txt_path, equipment_list, persons_list, identity_data,
                            violation_results, chat_pairs, vacant_positions, img_width, img_height,
                            all_model_responses, all_judgment_processes)

        print(f"\n✓ {image_filename} 处理成功")

        # 结束计时
        time_tracker.end_image()

        return True

    except Exception as e:
        print(f"\n✗ 处理 {image_filename} 时出错: {e}")
        import traceback
        print(traceback.format_exc())

        # 结束计时
        time_tracker.end_image()

        return False


def main():
    """主函数"""
    # 配置
    input_folder = r"E:xxxx" #可以放一个父文件夹地址，父文件夹下可以是各个营业厅的子文件夹 
    api_key = "sk-xxxx"
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    print("="*80)
    print("员工行为检测工具 - 新流程")
    print("使用数学计算判断人员身份(基于电脑显示器拟合直线)")
    print("="*80)
    print(f"输入文件夹: {input_folder}")
    print("\n处理流程:")
    print("  步骤1: 识别办公设备和工作台(显示器、键盘、办公桌)")
    print("  步骤2: 识别人员位置(包括截断人员)")
    print("  步骤3: 检测缺岗(无人工位)")
    print("  步骤4: 拟合直线判断人员身份(使用所有设备和办公桌中心)")
    print("  步骤5: 在原图上绘制设备、拟合直线、人员框和缺岗标注")
    print("  步骤6: 缺岗标注已添加")
    print("  步骤7: 裁剪并检测工作人员违规行为(睡觉、便服、玩手机)")
    print("  步骤8: 将所有工作人员组合检测聊天行为")
    print("  步骤9: 绘制人员框和违规行为标签")
    print("  最终: 绘制所有标注(设备+人员+缺岗+违规+聊天)并保存汇总结果")
    print("="*80)

    # 创建输出文件夹
    output_folder = input_folder + "_result_new"
    os.makedirs(output_folder, exist_ok=True)

    # 初始化客户端
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP'}

    # 递归获取所有图片文件（包括子文件夹）
    all_image_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if os.path.splitext(file)[1] in image_extensions:
                full_path = os.path.join(root, file)
                # 计算相对路径，用于保持输出目录结构
                rel_path = os.path.relpath(full_path, input_folder)
                all_image_files.append((full_path, rel_path))

    if not all_image_files:
        print(f"\n在 {input_folder} 及其子文件夹中没有找到图片文件")
        return

    print(f"\n找到 {len(all_image_files)} 张图片（包括子文件夹）")
    print("="*80)

    # 处理每张图片
    success_count = 0
    fail_count = 0

    for i, (input_path, rel_path) in enumerate(all_image_files):
        print(f"\n处理图片 {i+1}/{len(all_image_files)}: {rel_path}")

        # 计算输出路径，保持子文件夹结构
        output_subfolder = os.path.join(output_folder, os.path.dirname(rel_path))
        os.makedirs(output_subfolder, exist_ok=True)

        success = process_image_new(input_path, output_subfolder, client)

        if success:
            success_count += 1
        else:
            fail_count += 1

    # 输出统计
    print("\n" + "="*80)
    print("处理完成!")
    print("="*80)
    print(f"总计: {len(all_image_files)} 张图片")
    print(f"成功: {success_count} 张")
    print(f"失败: {fail_count} 张")
    print(f"输出文件夹: {output_folder}")
    print("="*80)

    # 打印并保存时间统计
    time_tracker.print_summary()
    time_stats_path = os.path.join(output_folder, "时间统计报告.txt")
    time_tracker.save_to_file(time_stats_path)


if __name__ == "__main__":
    main()
