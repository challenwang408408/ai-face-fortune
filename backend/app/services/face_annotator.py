"""
面部特征圈注服务
使用 MediaPipe Tasks FaceLandmarker 获取 478 个面部关键点，
然后用 Pillow 在图片上绘制五个面相维度的区域标注。
"""

import base64
import io
import logging
from pathlib import Path
from typing import Optional

import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger("ai_fortune")

# ── 模型路径 ──
MODEL_PATH = str(Path(__file__).resolve().parent.parent.parent / "models" / "face_landmarker.task")

# ── 五维面相区域对应的 FaceLandmarker 关键点索引 ──
REGION_LANDMARKS = {
    "印堂": {
        "label": "印堂",
        "color": (255, 80, 80),       # 红色
        "indices": [9, 10, 151, 108, 337, 168, 6, 197, 195, 5],
        "pad": 12,
    },
    "眉眼神态": {
        "label": "眉眼",
        "color": (80, 200, 80),       # 绿色
        "indices": [
            # 左眉
            70, 63, 105, 66, 107,
            # 右眉
            300, 293, 334, 296, 336,
            # 左眼
            33, 133, 160, 159, 158, 144, 145, 153,
            # 右眼
            362, 263, 387, 386, 385, 373, 374, 380,
        ],
        "pad": 10,
    },
    "鼻翼鼻头": {
        "label": "鼻",
        "color": (80, 120, 255),      # 蓝色
        "indices": [1, 2, 98, 327, 168, 6, 197, 195, 5, 4, 45, 275],
        "pad": 8,
    },
    "嘴角下巴": {
        "label": "嘴/下巴",
        "color": (255, 180, 40),      # 橙色
        "indices": [
            # 嘴唇
            61, 291, 0, 17, 78, 308, 191, 415,
            # 下巴
            152, 377, 148, 176, 400, 175,
        ],
        "pad": 12,
    },
    "整体气色": {
        "label": "气色",
        "color": (200, 80, 255),      # 紫色
        # 面部轮廓采样点
        "indices": [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
            361, 288, 397, 365, 379, 378, 400, 377, 152,
            148, 176, 149, 150, 136, 172, 58, 132, 93, 234,
            127, 162, 21, 54, 103, 67, 109,
        ],
        "pad": 0,
    },
}


def _create_landmarker():
    """创建 FaceLandmarker 实例"""
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return FaceLandmarker.create_from_options(options)


def _get_face_landmarks(image_rgb: np.ndarray) -> Optional[list]:
    """使用 MediaPipe FaceLandmarker 获取面部关键点"""
    landmarker = _create_landmarker()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = landmarker.detect(mp_image)
    landmarker.close()

    if result.face_landmarks and len(result.face_landmarks) > 0:
        return result.face_landmarks[0]
    return None


def _get_font(size: int = 22):
    """获取支持中文的字体"""
    font_paths = [
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    for fp in font_paths:
        try:
            return ImageFont.truetype(fp, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def annotate_face(image_base64: str, features: dict) -> str:
    """
    在人脸图片上标注五维面相区域。

    参数:
        image_base64: 图片的 base64 data URI (data:image/png;base64,...)
        features: AI 提取的五维特征 dict

    返回:
        标注后图片的 base64 data URI
    """
    # 解码图片
    if "," in image_base64:
        header, b64data = image_base64.split(",", 1)
    else:
        header = "data:image/png;base64"
        b64data = image_base64

    img_bytes = base64.b64decode(b64data)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_np = np.array(img)

    # 获取面部关键点
    landmarks = _get_face_landmarks(img_np)
    if landmarks is None:
        logger.warning("[圈注] 未检测到人脸，返回原图")
        return image_base64

    h, w = img_np.shape[:2]

    # 创建半透明覆盖层
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)

    # 在原图上绘制
    img_rgba = img.convert("RGBA")
    draw = ImageDraw.Draw(img_rgba)
    font = _get_font(20)
    font_small = _get_font(14)

    # 逐维度绘制标注
    annotation_info = []
    for dim_key, region in REGION_LANDMARKS.items():
        indices = region["indices"]
        color = region["color"]
        label = region["label"]
        pad = region["pad"]

        # 获取该区域的关键点坐标
        points = []
        for idx in indices:
            if idx < len(landmarks):
                lm = landmarks[idx]
                px, py = int(lm.x * w), int(lm.y * h)
                points.append((px, py))

        if not points:
            continue

        # 计算包围矩形
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x_min = max(0, min(xs) - pad)
        y_min = max(0, min(ys) - pad)
        x_max = min(w, max(xs) + pad)
        y_max = min(h, max(ys) + pad)

        # 绘制半透明填充区域
        fill_color = (*color, 35)
        draw_overlay.rectangle([x_min, y_min, x_max, y_max], fill=fill_color)

        # 绘制边框
        border_color = (*color, 200)
        draw.rectangle([x_min, y_min, x_max, y_max], outline=border_color, width=2)

        # 绘制标签背景
        label_text = f"[{label}]"
        bbox = font.getbbox(label_text)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        label_x = x_min
        label_y = max(0, y_min - text_h - 6)

        draw.rectangle(
            [label_x, label_y, label_x + text_w + 8, label_y + text_h + 4],
            fill=(*color, 180),
        )
        draw.text((label_x + 4, label_y + 1), label_text, fill=(255, 255, 255, 255), font=font)

        # 在右侧添加特征描述
        feature_text = features.get(dim_key, "")
        if feature_text and len(feature_text) > 20:
            feature_text = feature_text[:18] + "…"
        if feature_text:
            desc_y = y_min + 2
            desc_x = x_max + 5
            if desc_x + 200 > w:
                desc_x = max(0, x_min - 200)
            draw.text((desc_x, desc_y), feature_text, fill=(*color, 230), font=font_small)

        annotation_info.append({
            "dimension": dim_key,
            "bbox": [x_min, y_min, x_max, y_max],
            "label": label,
        })

    # 合成覆盖层
    img_final = Image.alpha_composite(img_rgba, overlay)
    img_final = img_final.convert("RGB")

    # 编码为 base64
    buf = io.BytesIO()
    img_final.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode()

    logger.info("[圈注] 完成标注，标注区域: %s", [a["dimension"] for a in annotation_info])

    return f"data:image/png;base64,{encoded}"
