#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step3_crop_margin1088_then_synth.py

1) 将 BACKGROUND_ROOT 下的背景图按固定边距裁成 1088x1088:
   left_trim=296, right_trim=296, top_trim=50, bottom_trim=50
   -> crop region: x = [296, 296+1088), y = [50, 50+1088)

2) 在裁剪结果上合成（不缩放抠图），每张合成图只含单一类别，密度：
示例：
   Coin_H:8, Coin_T:8, Markers_Damage010:18, Markers_Damage050:12,
   Markers_Damage100:10, Markers_Poisoned:5, Markers_Burned:5,
   Markers_GX_Avaiable:3, Markers_GX_Used:3

输出：
 - crops -> SYNTH_ROOT/crops_1088_margin
 - 合成图 -> SYNTH_ROOT/<class>/images/
 - 可视化 -> SYNTH_ROOT/<class>/viz/
 - 标注 NDJSON -> SYNTH_ROOT/placements_cropped1088_margins.ndjson
 - 摘要 -> SYNTH_ROOT/synth_summary_cropped1088_margins.json

--mixed-classes: 开启混合类别模式，在一张图上放置不同类型的指示物,默认为False
--bg-repeat: 设置背景图片重复使用次数(默认1),通过 bg_repeat 参数控制背景图片重复次数,重复的背景图片会添加序号后缀（如 _r01, _r02）

用法（示例）：
混合类别模式（推荐用于实战场景）:
python /mnt/afs/xiemingjin/sam_project/tool/step3_synthesize.py \
  --cutout-root /mnt/afs/xiemingjin/sam_project/saved/resized/20250909_cutouts_0.2 \
  --background-root /mnt/afs/share_data/Pokemon/v0.2/data/raw_data/det_markers/20250826_det_maskers_train_material_20250813_pokemon_reid_traindata_csvE1C_Po_v1.0_DIY02_cam1_aruco_unocc/img_h \
  --synth-root /mnt/afs/xiemingjin/sam_project/saved/training_data/20250910_sam_step2_synth_crop1088_margins_v2 \
  --mixed-classes \
  --bg-repeat 5
单类别模式（原有模式）：不用加 --mixed-classes
  python /mnt/afs/xiemingjin/sam_project/tool/step3_synthesize.py \
  --cutout-root /mnt/afs/xiemingjin/sam_project/saved/resized/20250919_cutouts_0.1_v1 \
  --background-root /mnt/afs/share_data/Pokemon/v0.2/data/raw_data/det_markers/20250826_det_maskers_train_material_20250813_pokemon_reid_traindata_csvE1C_Po_v1.0_DIY02_cam1_aruco_unocc/img_h \
  --synth-root /mnt/afs/xiemingjin/sam_project/saved/training_data/20250919_sam_step2_synth_crop1088_margins_0.1_v2 \
  --bg-repeat 5
"""
import os
import cv2
import json
import random
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

# -------------------- 默认配置（按需改） --------------------
TARGET_W = 1088
TARGET_H = 1088

# 基于照片分析的卡牌有效区域（相对于1088x1088图像）
CARD_AREA_CONFIG = {
    "x_start_ratio": 0.02,    # 左边缘留5%边距
    "x_end_ratio": 0.99,     # 右边缘到90%（避开右侧无卡牌区域）
    "y_start_ratio": 0.01,   # 上边缘留15%边距（避开上方无卡牌区域）
    "y_end_ratio": 0.99,     # 下边缘到95%
}

CUTOUT_ROOT_DEFAULT = "/mnt/afs/xiemingjin/sam_project/saved/20250905_sam_step1_out/cutouts"
BACKGROUND_ROOT_DEFAULT = "/mnt/afs/share_data/Pokemon/v0.2/data/raw_data/det_markers/20250826_det_maskers_train_material_20250813_pokemon_reid_traindata_csvE1C_Po_v1.0_DIY02_cam1_aruco_unocc/img_h"
SYNTH_ROOT_DEFAULT = "/mnt/afs/xiemingjin/sam_project/saved/_sam_step2_synth_crop1088_margins"

# 裁剪边距（你指定的值）
LEFT_TRIM = 296
RIGHT_TRIM = 296
TOP_TRIM = 50
BOTTOM_TRIM = 50

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

CLASS_NAME_TO_ID = {
    "Coin_H": 1, "Coin_T": 2,
    "Markers_Damage010": 3, "Markers_Damage050": 4, "Markers_Damage100": 5,
    "Markers_Poisoned": 6, "Markers_Burned": 7,
    "Markers_GX_Avaiable": 8, "Markers_GX_Used": 9
}

# 新密度，权重分配
DENSITY_PER_IMAGE = {
    "Coin_H": 12,
    "Coin_T": 12,
    "Markers_Damage010": 60,
    "Markers_Damage050": 50,
    "Markers_Damage100": 40,
    "Markers_Poisoned": 30,
    "Markers_Burned": 30,
    "Markers_GX_Avaiable": 12,
    "Markers_GX_Used": 12,
}

# 混合模式下每张图片的总对象数量范围 (最小值, 最大值)
MIXED_TOTAL_OBJECTS_RANGE = (40, 60)

# 放置策略（可调）
BORDER_MARGIN = 8
MAX_OVERLAP_IOU = 0.05
MAX_TRY_PER_OBJECT = 200
GLOBAL_SEED = 20250826
SAVE_JPG_QUALITY = 95

# 添加alpha阈值常量，用于过滤边缘光晕
ALPHA_THRESHOLD = 8  # 忽略透明度低于此值的像素，避免边缘光晕影响bbox计算

DRAW_VIZ = True
VIZ_BOX_THICK = 2
VIZ_FONT = cv2.FONT_HERSHEY_SIMPLEX
VIZ_FONT_SCALE = 0.6
VIZ_TEXT_THICK = 1

# -------------------- 工具函数 --------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))

def is_image_file(p: str):
    return os.path.splitext(p)[1].lower() in IMG_EXTS

def list_all_images(root: str):
    files = []
    for p in glob(os.path.join(root, "**", "*"), recursive=True):
        if is_image_file(p):
            files.append(p)
    return sorted(files)

def list_cutouts_by_class(cutout_root: str, class_name: str):
    cls_dir = os.path.join(cutout_root, class_name)
    if not os.path.isdir(cls_dir):
        return []
    files = [p for p in glob(os.path.join(cls_dir, "*")) if is_image_file(p)]
    return sorted(files)

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

#增进版本，自适应，可以裁剪成你想要的任何格式，只需要修改target_w和target_h
def crop_with_margins(img, left_trim, right_trim, top_trim, bottom_trim, target_w=TARGET_W, target_h=TARGET_H):
    H, W = img.shape[:2]
    
    # 如果图像已经是目标尺寸，直接返回
    if W == target_w and H == target_h:
        return img, (0, 0, W, H)

    # 如果图像尺寸大于目标尺寸，按比例裁剪
    if W > target_w and H > target_h:
        # 计算裁剪比例
        crop_w = (W - target_w) // 2
        crop_h = (H - target_h) // 2
        return img[crop_h:crop_h + target_h, crop_w:crop_w + target_w], (crop_w, crop_h, crop_w + target_w, crop_h + target_h)

    # 如果图像尺寸小于目标尺寸，进行反射填充
    pad_left = pad_right = pad_top = pad_bottom = 0
    if W < target_w:
        pad_left = (target_w - W) // 2
        pad_right = target_w - W - pad_left
    if H < target_h:
        pad_top = (target_h - H) // 2
        pad_bottom = target_h - H - pad_top

    # 填充图像
    padded_img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT)
    H, W = padded_img.shape[:2]

    # 进行裁剪，确保图像大小为目标尺寸
    crop_x1 = max(0, (W - target_w) // 2)
    crop_y1 = max(0, (H - target_h) // 2)
    crop_x2 = crop_x1 + target_w
    crop_y2 = crop_y1 + target_h

    return padded_img[crop_y1:crop_y2, crop_x1:crop_x2], (crop_x1, crop_y1, crop_x2, crop_y2)

# 问题：混合使用浮点数和整数
def compute_bbox_from_alpha(img_rgba, alpha_threshold=ALPHA_THRESHOLD):
    """计算基于alpha通道的紧致边界框，使用阈值过滤边缘光晕，保持浮点精度
    
    Args:
        img_rgba: RGBA图像
        alpha_threshold: alpha阈值，低于此值的像素被忽略
    
    Returns:
        [x, y, w, h] 浮点数坐标的边界框，保留四位小数
    """
    if img_rgba.shape[2] < 4:
        # 无alpha通道，返回整个图像尺寸
        h, w = img_rgba.shape[:2]
        return [0.0, 0.0, float(w), float(h)]
    
    alpha = img_rgba[:, :, 3]
    # 使用alpha阈值过滤边缘光晕，避免极低透明度像素影响bbox
    coords = np.column_stack(np.where(alpha > alpha_threshold))
    if len(coords) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    
    ys, xs = coords[:, 0], coords[:, 1]
    # 保持浮点精度，四舍五入到四位小数
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    
    # 返回浮点数bbox，保留四位小数精度
    bbox = [x_min, y_min, x_max - x_min + 1.0, y_max - y_min + 1.0]
    return [round(coord, 4) for coord in bbox]

def alpha_paste(bg_bgr, fg_bgra, x0, y0, alpha_threshold=ALPHA_THRESHOLD):
    """Alpha混合粘贴，支持浮点位置输入但使用整数粘贴
    
    Args:
        bg_bgr: 背景图像
        fg_bgra: 前景RGBA图像
        x0, y0: 粘贴位置（浮点数输入，内部转为整数）
        alpha_threshold: alpha阈值
    """
    # 四舍五入到最近整数像素进行实际粘贴
    x0_int, y0_int = round(x0), round(y0)
    
    h, w = fg_bgra.shape[:2]
    H, W = bg_bgr.shape[:2]
    
    # 边界检查
    if x0_int < 0 or y0_int < 0 or x0_int + w > W or y0_int + h > H:
        x1 = max(0, x0_int); y1 = max(0, y0_int)
        x2 = min(W, x0_int + w); y2 = min(H, y0_int + h)
        if x2 <= x1 or y2 <= y1:
            return
        fx1 = x1 - x0_int; fy1 = y1 - y0_int; fx2 = fx1 + (x2 - x1); fy2 = fy1 + (y2 - y1)
        roi = bg_bgr[y1:y2, x1:x2]
        if fg_bgra.shape[2] == 4:
            b,g,r,a = cv2.split(fg_bgra[fy1:fy2, fx1:fx2])
            # 应用alpha阈值，过滤边缘光晕
            a = a.astype(np.float32) / 255.0
            a = np.where(fg_bgra[fy1:fy2, fx1:fx2, 3] > alpha_threshold, a, 0)
            a3 = np.stack([a,a,a], axis=-1)
            fg = np.stack([b,g,r], axis=-1).astype(np.float32)
            bg = roi.astype(np.float32)
            out = fg * a3 + bg * (1 - a3)
            bg_bgr[y1:y2, x1:x2] = np.clip(out, 0, 255).astype(np.uint8)
        else:
            bg_bgr[y1:y2, x1:x2] = fg_bgra[fy1:fy2, fx1:fx2, :3]
        return

    roi = bg_bgr[y0_int:y0_int+h, x0_int:x0_int+w]
    if fg_bgra.shape[2] == 4:
        b,g,r,a = cv2.split(fg_bgra)
        a = a.astype(np.float32) / 255.0
        # 应用alpha阈值，过滤边缘光晕
        a = np.where(fg_bgra[:, :, 3] > alpha_threshold, a, 0)
        a3 = np.stack([a,a,a], axis=-1)
        fg = np.stack([b,g,r], axis=-1).astype(np.float32)
        bg = roi.astype(np.float32)
        out = fg * a3 + bg * (1 - a3)
        bg_bgr[y0_int:y0_int+h, x0_int:x0_int+w] = np.clip(out, 0, 255).astype(np.uint8)
    else:
        bg_bgr[y0_int:y0_int+h, x0_int:x0_int+w] = fg_bgra[:, :, :3]

def iou_xywh(b1, b2):
    x1, y1, w1, h1 = b1; x2, y2, w2, h2 = b2
    xa1, ya1, xa2, ya2 = x1, y1, x1 + w1, y1 + h1
    xb1, yb1, xb2, yb2 = x2, y2, x2 + w2, y2 + h2
    inter = max(0, min(xa2, xb2) - max(xa1, xb1)) * max(0, min(ya2, yb2) - max(ya1, yb1))
    union = max(1e-9, w1*h1 + w2*h2 - inter)
    return inter / union

def can_place(candidate_bbox, placed_bboxes, max_iou=MAX_OVERLAP_IOU):
    for b in placed_bboxes:
        if iou_xywh(candidate_bbox, b) > max_iou:
            return False
    return True

def get_card_placement_area(img_width, img_height):
    """根据图像尺寸计算卡牌放置的有效区域"""
    x_start = int(img_width * CARD_AREA_CONFIG["x_start_ratio"])
    x_end = int(img_width * CARD_AREA_CONFIG["x_end_ratio"])
    y_start = int(img_height * CARD_AREA_CONFIG["y_start_ratio"])
    y_end = int(img_height * CARD_AREA_CONFIG["y_end_ratio"])
    
    return {
        "x_range": (x_start, x_end),
        "y_range": (y_start, y_end),
        "width": x_end - x_start,
        "height": y_end - y_start
    }

def try_place_object(bg_img, obj_img, existing_bboxes, max_attempts=100):
    """尝试在背景图中放置对象，确保坐标一致性"""
    bg_h, bg_w = bg_img.shape[:2]
    obj_h, obj_w = obj_img.shape[:2]
    
    for _ in range(max_attempts):
        # 生成浮点数随机位置
        x0_float = random.uniform(0, max(0, bg_w - obj_w))
        y0_float = random.uniform(0, max(0, bg_h - obj_h))
        
        # 四舍五入到整数位置（与粘贴时保持一致）
        x0_int, y0_int = round(x0_float), round(y0_float)
        
        tight_bbox = compute_bbox_from_alpha(obj_img)
        if tight_bbox[2] <= 0 or tight_bbox[3] <= 0:
            continue
            
        # 计算在背景图中的实际bbox位置（使用整数坐标）
        tight_bbox_bg = [
            x0_int + tight_bbox[0],
            y0_int + tight_bbox[1], 
            tight_bbox[2],
            tight_bbox[3]
        ]
        
        if can_place(tight_bbox_bg, existing_bboxes):
            return x0_int, y0_int, tight_bbox_bg
    
    return None, None, None
    
    for _ in range(max_attempts):
        # 在卡牌区域内随机选择位置
        x0_float = random.uniform(x_min, effective_x_max)
        y0_float = random.uniform(y_min, effective_y_max)
        
        # 四舍五入到整数位置
        x0_int, y0_int = round(x0_float), round(y0_float)
        
        tight_bbox = compute_bbox_from_alpha(obj_img)
        if tight_bbox[2] <= 0 or tight_bbox[3] <= 0:
            continue
            
        # 计算在背景图中的实际bbox位置（使用整数坐标）
        tight_bbox_bg = [
            x0_int + tight_bbox[0],
            y0_int + tight_bbox[1], 
            tight_bbox[2],
            tight_bbox[3]
        ]
        
        # 确保bbox完全在卡牌区域内
        bbox_x_end = tight_bbox_bg[0] + tight_bbox_bg[2]
        bbox_y_end = tight_bbox_bg[1] + tight_bbox_bg[3]
        
        if (tight_bbox_bg[0] >= x_min and bbox_x_end <= x_max and 
            tight_bbox_bg[1] >= y_min and bbox_y_end <= y_max):
            
            if can_place(tight_bbox_bg, existing_bboxes):
                return x0_int, y0_int, tight_bbox_bg
    
    return None, None, None

def try_place_object_in_card_area(bg_img, obj_img, existing_bboxes, max_attempts=100):
    """在卡牌区域内尝试放置对象，保持坐标精度"""
    bg_h, bg_w = bg_img.shape[:2]
    obj_h, obj_w = obj_img.shape[:2]
    
    # 获取卡牌放置区域
    card_area = get_card_placement_area(bg_w, bg_h)
    x_min, x_max = card_area["x_range"]
    y_min, y_max = card_area["y_range"]
    
    # 确保对象能完全放置在区域内
    effective_x_max = max(x_min, x_max - obj_w)
    effective_y_max = max(y_min, y_max - obj_h)
    
    if effective_x_max < x_min or effective_y_max < y_min:
        return None, None, None
    
    for _ in range(max_attempts):
        # 在卡牌区域内随机选择位置，保持浮点精度
        x0_float = random.uniform(x_min, effective_x_max)
        y0_float = random.uniform(y_min, effective_y_max)
        
        tight_bbox = compute_bbox_from_alpha(obj_img)
        if tight_bbox[2] <= 0 or tight_bbox[3] <= 0:
            continue
            
        # 计算在背景图中的实际bbox位置（保持浮点精度）
        tight_bbox_bg = [
            x0_float + tight_bbox[0],
            y0_float + tight_bbox[1], 
            tight_bbox[2],
            tight_bbox[3]
        ]
        
        # 四舍五入到四位小数
        tight_bbox_bg = [round(coord, 4) for coord in tight_bbox_bg]
        
        # 确保bbox完全在卡牌区域内
        bbox_x_end = tight_bbox_bg[0] + tight_bbox_bg[2]
        bbox_y_end = tight_bbox_bg[1] + tight_bbox_bg[3]
        
        if (tight_bbox_bg[0] >= x_min and bbox_x_end <= x_max and 
            tight_bbox_bg[1] >= y_min and bbox_y_end <= y_max):
            
            if can_place(tight_bbox_bg, existing_bboxes):
                return x0_float, y0_float, tight_bbox_bg
    
    return None, None, None

def try_place_object(bg_img, obj_img, existing_bboxes, max_attempts=100):
    bg_h, bg_w = bg_img.shape[:2]
    obj_h, obj_w = obj_img.shape[:2]
    
    for _ in range(max_attempts):
        # 保持浮点数精度
        x0 = random.uniform(0, max(0, bg_w - obj_w))
        y0 = random.uniform(0, max(0, bg_h - obj_h))
        
        tight_bbox = compute_bbox_from_alpha(obj_img)
        if tight_bbox[2] <= 0 or tight_bbox[3] <= 0:
            continue
            
        # 计算在背景图中的实际bbox位置（保持浮点数）
        tight_bbox_bg = [
            x0 + tight_bbox[0],
            y0 + tight_bbox[1], 
            tight_bbox[2],
            tight_bbox[3]
        ]
        
        if can_place(tight_bbox_bg, existing_bboxes):
            return x0, y0, tight_bbox_bg
    
    return None, None, None

# def try_place_object_optimized(bg_w, bg_h, fg_bgra, tight_bbox, placed_bboxes, border=BORDER_MARGIN, max_try=MAX_TRY_PER_OBJECT):
#     """优化的物体放置算法，使用网格采样 + 随机扰动"""
#     fh, fw = fg_bgra.shape[:2]
#     tx, ty, tw, th = tight_bbox
#     min_x0 = border - tx
#     min_y0 = border - ty
#     max_x0 = (bg_w - border) - (tx + tw)
#     max_y0 = (bg_h - border) - (ty + th)
    
#     if max_x0 < min_x0 or max_y0 < min_y0:
#         return None
    
#     # 网格采样策略
#     grid_size = 20
#     x_step = max(1, (max_x0 - min_x0) // grid_size)
#     y_step = max(1, (max_y0 - min_y0) // grid_size)
    
#     candidates = []
#     for x in range(int(min_x0), int(max_x0), x_step):
#         for y in range(int(min_y0), int(max_y0), y_step):
#             # 添加随机扰动
#             x0 = x + random.uniform(-x_step//2, x_step//2)
#             y0 = y + random.uniform(-y_step//2, y_step//2)
#             x0 = max(min_x0, min(max_x0, x0))
#             y0 = max(min_y0, min(max_y0, y0))
#             candidates.append((x0, y0))
    
#     # 随机打乱候选位置
#     random.shuffle(candidates)
    
#     for x0, y0 in candidates[:max_try]:
#         cand_bbox = [x0 + tx, y0 + ty, tw, th]
#         if can_place(cand_bbox, placed_bboxes):
#             return (x0, y0), cand_bbox
    
#     return None

def draw_viz(vis, bbox, label):
    x, y, w, h = [int(v) for v in bbox]
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), VIZ_BOX_THICK)
    cv2.putText(vis, label, (x, max(12, y - 6)), VIZ_FONT, VIZ_FONT_SCALE, (0, 255, 0), VIZ_TEXT_THICK, cv2.LINE_AA)

# -------------------- 主流程 --------------------
def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--cutout-root", default=CUTOUT_ROOT_DEFAULT)
    p.add_argument("--background-root", default=BACKGROUND_ROOT_DEFAULT)
    p.add_argument("--synth-root", default=SYNTH_ROOT_DEFAULT)
    p.add_argument("--left-trim", type=int, default=LEFT_TRIM)
    p.add_argument("--right-trim", type=int, default=RIGHT_TRIM)
    p.add_argument("--top-trim", type=int, default=TOP_TRIM)
    p.add_argument("--bottom-trim", type=int, default=BOTTOM_TRIM)
    p.add_argument("--only-crop", action="store_true", help="只做裁剪（生成 crops），不做合成")
    p.add_argument("--mixed-classes", action="store_true", help="在一张图上放置多种类别的指示物")
    p.add_argument("--bg-repeat", type=int, default=1, help="背景图片重复使用次数（默认1）")
    p.add_argument("--seed", type=int, default=GLOBAL_SEED)
    p.add_argument("--verbose", action="store_true")
    return p

# 在main函数开始处添加配置验证
def validate_config(args):
    """验证配置参数的合理性"""
    if not os.path.exists(args.cutout_root):
        raise FileNotFoundError(f"抠图根目录不存在: {args.cutout_root}")
    if not os.path.exists(args.background_root):
        raise FileNotFoundError(f"背景图根目录不存在: {args.background_root}")
    
    # 验证密度配置
    for cls_name, density in DENSITY_PER_IMAGE.items():
        if density <= 0 or density > 250:
            print(f"[WARN] {cls_name} 密度配置异常: {density}")
    
    # 验证目标尺寸
    if TARGET_W <= 0 or TARGET_H <= 0:
        raise ValueError(f"目标尺寸无效: {TARGET_W}x{TARGET_H}")

def generate_mixed_object_distribution(total_objects):
    """根据密度配置生成混合对象分布"""
    # 计算总权重
    total_weight = sum(DENSITY_PER_IMAGE.values())
    
    # 如果总权重为0，返回空分布
    if total_weight == 0:
        return {cls_name: 0 for cls_name in DENSITY_PER_IMAGE.keys()}
    
    # 按权重分配对象数量
    distribution = {}
    remaining_objects = total_objects
    
    for cls_name, weight in DENSITY_PER_IMAGE.items():
        if remaining_objects <= 0:
            distribution[cls_name] = 0
            continue
            
        # 按权重比例分配，但至少保证有一定随机性
        ratio = weight / total_weight
        expected_count = int(total_objects * ratio)
        
        # 添加随机波动 ±20%
        variation = int(expected_count * 0.2)
        actual_count = max(0, expected_count + random.randint(-variation, variation))
        actual_count = min(actual_count, remaining_objects)
        
        distribution[cls_name] = actual_count
        remaining_objects -= actual_count
    
    # 将剩余对象随机分配给**有权重的**类别
    valid_class_names = [cls_name for cls_name, weight in DENSITY_PER_IMAGE.items() if weight > 0]
    while remaining_objects > 0 and valid_class_names:
        cls_name = random.choice(valid_class_names)
        distribution[cls_name] += 1
        remaining_objects -= 1
    
    return distribution

def run_step3_synthesize(
    cutout_root: str,
    background_root: str,
    output_root: str,
    config,
    **kwargs
) -> dict:
    """
    Step3: 图像合成主函数 - 供pipeline调用
    
    Args:
        cutout_root: step2输出的缩放后cutouts目录路径
        background_root: 背景图像目录路径
        output_root: 输出目录路径
        config: 配置对象
        **kwargs: 其他参数
        
    Returns:
        dict: 处理结果统计
    """
    from pathlib import Path
    
    # 从配置获取参数
    step3_config = config.step3
    class_name_to_id = config.config_manager.get_category_mapping()
    density_per_image = config.config_manager.get_density_per_image()
    
    # 设置随机种子
    set_seed(step3_config.seed)
    
    # 获取配置参数
    left_trim = step3_config.left_trim
    right_trim = step3_config.right_trim
    top_trim = step3_config.top_trim
    bottom_trim = step3_config.bottom_trim
    mixed_classes = step3_config.mixed_classes
    bg_repeat = max(1, step3_config.bg_repeat)
    verbose = config.global_config.verbose
    
    synth_root = output_root
    ensure_dir(synth_root)
    crops_dir = Path(synth_root) / "crops_1088_margin"
    ensure_dir(crops_dir)
    
    # 统计信息
    stats = {
        "total_backgrounds": 0,
        "total_cropped": 0,
        "total_synthesized": 0,
        "classes_processed": [],
        "mixed_mode": mixed_classes,
        "bg_repeat": bg_repeat
    }
    
    if verbose:
        print(f"🔄 开始处理图像合成任务")
        print(f"📁 抠图目录: {cutout_root}")
        print(f"📁 背景目录: {background_root}")
        print(f"📁 输出目录: {synth_root}")
        print(f"🎯 混合类别模式: {mixed_classes}")
        print(f"🔄 背景重复次数: {bg_repeat}")
    
    # 1) 列出背景，做裁剪
    bg_files = list_all_images(background_root)
    if len(bg_files) == 0:
        raise RuntimeError("未找到背景图片，请检查背景目录")
    
    stats["total_backgrounds"] = len(bg_files)
    if verbose:
        print(f"Found {len(bg_files)} background images, repeat {bg_repeat} times = {len(bg_files) * bg_repeat} total")
    
    crop_records = []  # list of dicts {crop_path, src_bg}
    
    # 根据重复次数扩展背景文件列表
    expanded_bg_files = bg_files * bg_repeat
    
    for idx, bg_path in enumerate(tqdm(expanded_bg_files, desc="Crop backgrounds")):
        img = cv2.imread(bg_path)
        if img is None:
            continue
        H, W = img.shape[:2]
        # preferred crop coords from user margins
        crop_img, (x1, y1, x2, y2) = crop_with_margins(img, left_trim, right_trim, top_trim, bottom_trim, TARGET_W, TARGET_H)
        # save with mirrored dir structure
        rel = os.path.relpath(bg_path, background_root)
        rel_dir = os.path.dirname(rel)
        out_dir = Path(crops_dir) / rel_dir
        ensure_dir(out_dir)
        
        # 为重复的背景添加序号后缀
        base_name = Path(bg_path).stem
        ext = Path(bg_path).suffix
        if bg_repeat > 1:
            repeat_idx = idx // len(bg_files)
            if repeat_idx > 0:
                base_name = f"{base_name}_r{repeat_idx:02d}"
        
        crop_name = f"{base_name}_crop{ext}"
        crop_path = out_dir / crop_name
        cv2.imwrite(str(crop_path), crop_img, [cv2.IMWRITE_JPEG_QUALITY, SAVE_JPG_QUALITY])
        
        crop_records.append({
            "crop_path": str(crop_path),
            "src_bg": bg_path,
            "crop_region": (x1, y1, x2, y2)
        })
    
    stats["total_cropped"] = len(crop_records)
    if verbose:
        print(f"✅ 完成背景裁剪: {len(crop_records)} 张")
    
    # 2) 列出所有类别的抠图
    all_cutouts = {}  # class_name -> [cutout_paths]
    for class_name in class_name_to_id.keys():
        cutouts = list_cutouts_by_class(cutout_root, class_name)
        if cutouts:
            all_cutouts[class_name] = cutouts
            if verbose:
                print(f"Found {len(cutouts)} cutouts for class '{class_name}'")
    
    if not all_cutouts:
        raise RuntimeError("未找到任何抠图文件，请检查抠图目录")
    
    stats["classes_processed"] = list(all_cutouts.keys())
    
    # 3) 合成逻辑
    all_placements = []  # 所有placement记录
    synth_count = 0
    
    if mixed_classes:
        # 混合类别模式：在一张图上放置不同类型的指示物
        if verbose:
            print("🎯 使用混合类别模式")
        
        synth_images_dir = Path(synth_root) / "mixed" / "images"
        synth_viz_dir = Path(synth_root) / "mixed" / "viz"
        ensure_dir(synth_images_dir)
        if DRAW_VIZ:
            ensure_dir(synth_viz_dir)
        
        for crop_record in tqdm(crop_records, desc="Synthesize mixed"):
            crop_path = crop_record["crop_path"]
            bg_img = cv2.imread(crop_path)
            if bg_img is None:
                continue
            
            # 生成混合对象分布
            total_objects = random.randint(*MIXED_TOTAL_OBJECTS_RANGE)
            object_distribution = generate_mixed_object_distribution(total_objects)
            
            # 在背景上合成对象
            placements = []
            composite = bg_img.copy()
            
            for class_name, count in object_distribution.items():
                if class_name not in all_cutouts or count == 0:
                    continue
                
                class_cutouts = all_cutouts[class_name]
                for _ in range(count):
                    # 随机选择一个抠图
                    cutout_path = random.choice(class_cutouts)
                    cutout_img = cv2.imread(cutout_path, cv2.IMREAD_UNCHANGED)
                    if cutout_img is None:
                        continue
                    
                    # 尝试放置对象
                    success = False
                    for attempt in range(MAX_TRY_PER_OBJECT):
                        # 随机位置
                        card_area = CARD_AREA_CONFIG
                        x_min = int(TARGET_W * card_area["x_start_ratio"])
                        x_max = int(TARGET_W * card_area["x_end_ratio"]) - cutout_img.shape[1]
                        y_min = int(TARGET_H * card_area["y_start_ratio"])
                        y_max = int(TARGET_H * card_area["y_end_ratio"]) - cutout_img.shape[0]
                        
                        if x_max <= x_min or y_max <= y_min:
                            break
                        
                        x = random.randint(x_min, x_max)
                        y = random.randint(y_min, y_max)
                        
                        # 检查重叠
                        new_bbox = (x, y, x + cutout_img.shape[1], y + cutout_img.shape[0])
                        overlap = False
                        for existing in placements:
                            if compute_iou_xywh(new_bbox, existing["bbox"]) > MAX_OVERLAP_IOU:
                                overlap = True
                                break
                        
                        if not overlap:
                            # 合成到背景上
                            composite = blend_cutout_onto_bg(composite, cutout_img, x, y)
                            
                            # 记录placement
                            placement = {
                                "bbox": new_bbox,
                                "class_name": class_name,
                                "class_id": class_name_to_id[class_name],
                                "cutout_path": cutout_path
                            }
                            placements.append(placement)
                            success = True
                            break
                    
                    if not success and verbose:
                        print(f"⚠️  无法放置 {class_name} 对象")
            
            # 保存合成图像
            crop_stem = Path(crop_path).stem
            synth_name = f"{crop_stem}_synth.jpg"
            synth_path = synth_images_dir / synth_name
            cv2.imwrite(str(synth_path), composite, [cv2.IMWRITE_JPEG_QUALITY, SAVE_JPG_QUALITY])
            
            # 保存可视化
            if DRAW_VIZ and placements:
                viz_img = composite.copy()
                for placement in placements:
                    draw_viz(viz_img, placement["bbox"], placement["class_name"])
                viz_path = synth_viz_dir / f"{crop_stem}_viz.jpg"
                cv2.imwrite(str(viz_path), viz_img, [cv2.IMWRITE_JPEG_QUALITY, SAVE_JPG_QUALITY])
            
            # 记录所有placements
            for placement in placements:
                placement.update({
                    "synth_image": str(synth_path),
                    "src_bg": crop_record["src_bg"],
                    "crop_region": crop_record["crop_region"]
                })
                all_placements.append(placement)
            
            synth_count += 1
    
    else:
        # 单类别模式：每张图只含单一类别
        if verbose:
            print("🎯 使用单类别模式")
        
        for class_name in all_cutouts.keys():
            if verbose:
                print(f"🏷️  处理类别: {class_name}")
            
            synth_images_dir = Path(synth_root) / class_name / "images"
            synth_viz_dir = Path(synth_root) / class_name / "viz"
            ensure_dir(synth_images_dir)
            if DRAW_VIZ:
                ensure_dir(synth_viz_dir)
            
            class_cutouts = all_cutouts[class_name]
            objects_per_image = density_per_image.get(class_name, 10)
            
            for crop_record in tqdm(crop_records, desc=f"Synthesize {class_name}"):
                crop_path = crop_record["crop_path"]
                bg_img = cv2.imread(crop_path)
                if bg_img is None:
                    continue
                
                # 在背景上合成对象
                placements = []
                composite = bg_img.copy()
                
                for _ in range(objects_per_image):
                    # 随机选择一个抠图
                    cutout_path = random.choice(class_cutouts)
                    cutout_img = cv2.imread(cutout_path, cv2.IMREAD_UNCHANGED)
                    if cutout_img is None:
                        continue
                    
                    # 尝试放置对象
                    success = False
                    for attempt in range(MAX_TRY_PER_OBJECT):
                        # 随机位置
                        card_area = CARD_AREA_CONFIG
                        x_min = int(TARGET_W * card_area["x_start_ratio"])
                        x_max = int(TARGET_W * card_area["x_end_ratio"]) - cutout_img.shape[1]
                        y_min = int(TARGET_H * card_area["y_start_ratio"])
                        y_max = int(TARGET_H * card_area["y_end_ratio"]) - cutout_img.shape[0]
                        
                        if x_max <= x_min or y_max <= y_min:
                            break
                        
                        x = random.randint(x_min, x_max)
                        y = random.randint(y_min, y_max)
                        
                        # 检查重叠
                        new_bbox = (x, y, x + cutout_img.shape[1], y + cutout_img.shape[0])
                        overlap = False
                        for existing in placements:
                            if compute_iou_xywh(new_bbox, existing["bbox"]) > MAX_OVERLAP_IOU:
                                overlap = True
                                break
                        
                        if not overlap:
                            # 合成到背景上
                            composite = blend_cutout_onto_bg(composite, cutout_img, x, y)
                            
                            # 记录placement
                            placement = {
                                "bbox": new_bbox,
                                "class_name": class_name,
                                "class_id": class_name_to_id[class_name],
                                "cutout_path": cutout_path
                            }
                            placements.append(placement)
                            success = True
                            break
                    
                    if not success and verbose:
                        print(f"⚠️  无法放置 {class_name} 对象")
                
                # 保存合成图像
                crop_stem = Path(crop_path).stem
                synth_name = f"{crop_stem}_synth.jpg"
                synth_path = synth_images_dir / synth_name
                cv2.imwrite(str(synth_path), composite, [cv2.IMWRITE_JPEG_QUALITY, SAVE_JPG_QUALITY])
                
                # 保存可视化
                if DRAW_VIZ and placements:
                    viz_img = composite.copy()
                    for placement in placements:
                        draw_viz(viz_img, placement["bbox"], placement["class_name"])
                    viz_path = synth_viz_dir / f"{crop_stem}_viz.jpg"
                    cv2.imwrite(str(viz_path), viz_img, [cv2.IMWRITE_JPEG_QUALITY, SAVE_JPG_QUALITY])
                
                # 记录所有placements
                for placement in placements:
                    placement.update({
                        "synth_image": str(synth_path),
                        "src_bg": crop_record["src_bg"],
                        "crop_region": crop_record["crop_region"]
                    })
                    all_placements.append(placement)
                
                synth_count += 1
    
    stats["total_synthesized"] = synth_count
    
    # 4) 保存placement记录
    placements_file = Path(synth_root) / "placements_cropped1088_margins.ndjson"
    with open(placements_file, 'w', encoding='utf-8') as f:
        for placement in all_placements:
            f.write(json.dumps(placement, ensure_ascii=False) + '\n')
    
    # 5) 保存摘要
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_backgrounds": stats["total_backgrounds"],
        "total_cropped": stats["total_cropped"],
        "total_synthesized": stats["total_synthesized"],
        "total_placements": len(all_placements),
        "mixed_mode": mixed_classes,
        "bg_repeat": bg_repeat,
        "classes": stats["classes_processed"],
        "config": {
            "left_trim": left_trim,
            "right_trim": right_trim,
            "top_trim": top_trim,
            "bottom_trim": bottom_trim,
            "target_size": [TARGET_W, TARGET_H]
        }
    }
    
    summary_file = Path(synth_root) / "synth_summary_cropped1088_margins.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    if verbose:
        print(f"\n📊 合成完成统计:")
        print(f"   📝 背景图片: {stats['total_backgrounds']}")
        print(f"   ✂️  裁剪图片: {stats['total_cropped']}")
        print(f"   🎨 合成图片: {stats['total_synthesized']}")
        print(f"   📍 总放置数: {len(all_placements)}")
        print(f"   📁 输出目录: {synth_root}")
        print(f"   📄 placement记录: {placements_file}")
        print(f"   📋 摘要文件: {summary_file}")
    
    return stats

def compute_iou_xywh(box1, box2):
    """计算两个边界框的IoU"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # 计算交集
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # 计算并集
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def blend_cutout_onto_bg(bg_img, cutout_img, x, y):
    """将抠图合成到背景上"""
    if cutout_img.shape[2] == 4:  # 有alpha通道
        alpha = cutout_img[:, :, 3] / 255.0
        for c in range(3):
            bg_img[y:y+cutout_img.shape[0], x:x+cutout_img.shape[1], c] = \
                bg_img[y:y+cutout_img.shape[0], x:x+cutout_img.shape[1], c] * (1 - alpha) + \
                cutout_img[:, :, c] * alpha
    else:  # 没有alpha通道，直接覆盖
        bg_img[y:y+cutout_img.shape[0], x:x+cutout_img.shape[1]] = cutout_img
    
    return bg_img

def main():
    try:
        args = build_argparser().parse_args()
        set_seed(args.seed)
        cutout_root = args.cutout_root
        background_root = args.background_root
        synth_root = args.synth_root
        left_trim = args.left_trim
        right_trim = args.right_trim
        top_trim = args.top_trim
        bottom_trim = args.bottom_trim
        mixed_classes = args.mixed_classes
        bg_repeat = max(1, args.bg_repeat)  # 确保至少为1

        ensure_dir(synth_root)
        crops_dir = Path(synth_root) / "crops_1088_margin"
        ensure_dir(crops_dir)

        # 1) 列出背景，做裁剪
        bg_files = list_all_images(background_root)
        if len(bg_files) == 0:
            raise RuntimeError("未找到背景图片，请检查 --background-root")
        print(f"Found {len(bg_files)} background images, repeat {bg_repeat} times = {len(bg_files) * bg_repeat} total")

        crop_records = []  # list of dicts {crop_path, src_bg}
        
        # 根据重复次数扩展背景文件列表
        expanded_bg_files = bg_files * bg_repeat
        
        for idx, bg_path in enumerate(tqdm(expanded_bg_files, desc="Crop backgrounds")):
            img = cv2.imread(bg_path)
            if img is None:
                continue
            H, W = img.shape[:2]
            # preferred crop coords from user margins
            # attempt to crop at (left_trim, top_trim) -> (left_trim+TARGET_W, top_trim+TARGET_H)
            crop_img, (x1, y1, x2, y2) = crop_with_margins(img, left_trim, right_trim, top_trim, bottom_trim, TARGET_W, TARGET_H)
            # save with mirrored dir structure
            rel = os.path.relpath(bg_path, background_root)
            rel_dir = os.path.dirname(rel)
            out_dir = Path(crops_dir) / rel_dir
            ensure_dir(out_dir)
            
            # 为重复的背景添加序号
            repeat_idx = idx // len(bg_files) + 1
            if bg_repeat > 1:
                out_name = f"{Path(bg_path).stem}_crop1088_margin_r{repeat_idx:02d}{Path(bg_path).suffix}"
            else:
                out_name = f"{Path(bg_path).stem}_crop1088_margin{Path(bg_path).suffix}"
                
            out_path = str(out_dir / out_name)
            cv2.imwrite(out_path, crop_img, [int(cv2.IMWRITE_JPEG_QUALITY), SAVE_JPG_QUALITY])
            crop_records.append({"crop_path": out_path, "src_bg": bg_path, "crop_box": [int(x1), int(y1), int(x2), int(y2)], "repeat_idx": repeat_idx})

        print(f"Saved {len(crop_records)} crops to {crops_dir}")

        if args.only_crop:
            print("只做裁剪，已退出 (only-crop=True).")
            return

        # 2) 合成阶段
        placement_ndjson = Path(synth_root) / "placements_cropped1088_margins.ndjson"
        nd_f = open(placement_ndjson, "w", encoding="utf-8")
        summary = {"time": datetime.now().isoformat(), "classes": {}, "num_crops": len(crop_records), "mixed_classes": mixed_classes, "bg_repeat": bg_repeat}

        if mixed_classes:
            # 混合模式：在一张图上放置多种类别
            print("\n== Mixed Classes Synthesis Mode ==")
            
            # 预加载所有类别的抠图
            all_cutouts = {}
            for cls_name in CLASS_NAME_TO_ID.keys():
                cutouts = list_cutouts_by_class(cutout_root, cls_name)
                if cutouts:
                    all_cutouts[cls_name] = cutouts
                    print(f"Loaded {len(cutouts)} cutouts for {cls_name}")
                else:
                    print(f"[WARN] class {cls_name} 没有抠图，跳过")
            
            if not all_cutouts:
                print("[ERROR] 没有找到任何抠图，退出")
                return
            
            out_img_dir = Path(synth_root) / "mixed" / "images"
            out_viz_dir = Path(synth_root) / "mixed" / "viz"
            ensure_dir(out_img_dir); ensure_dir(out_viz_dir)
            
            images_created = 0
            total_objects_all = 0
            class_counts = {cls_name: 0 for cls_name in CLASS_NAME_TO_ID.keys()}
            
            for crop_idx, crop_rec in enumerate(tqdm(crop_records, desc="Mixed synthesis")):
                crop_path = crop_rec["crop_path"]
                bg = cv2.imread(crop_path)
                if bg is None:
                    continue
                H, W = bg.shape[:2]
                if (W, H) != (TARGET_W, TARGET_H):
                    # safety fallback: ensure correct size by center crop/pad
                    bg, _ = crop_with_margins(bg, max(0,(W-TARGET_W)//2), max(0,(W-TARGET_W)//2), max(0,(H-TARGET_H)//2), max(0,(H-TARGET_H)//2), TARGET_W, TARGET_H)

                vis = bg.copy() if DRAW_VIZ else None
                placed_bboxes = []
                objects = []
                
                # 随机确定这张图的总对象数量
                total_objects = random.randint(*MIXED_TOTAL_OBJECTS_RANGE)
                
                # 生成各类别的对象分布
                object_distribution = generate_mixed_object_distribution(total_objects)
                
                # 按分布放置各类别对象
                for cls_name, target_count in object_distribution.items():
                    if target_count <= 0 or cls_name not in all_cutouts:
                        continue
                    
                    cls_id = CLASS_NAME_TO_ID[cls_name]
                    cutouts = all_cutouts[cls_name]
                    placed_count = 0
                    attempts = 0
                    
                    while placed_count < target_count and attempts < target_count * MAX_TRY_PER_OBJECT:
                        attempts += 1
                        cut_path = random.choice(cutouts)
                        fg = cv2.imread(cut_path, cv2.IMREAD_UNCHANGED)
                        if fg is None:
                            continue
                        fh, fw = fg.shape[:2]
                        # too large skip
                        if fw > W - 2 * BORDER_MARGIN or fh > H - 2 * BORDER_MARGIN:
                            continue
                        tight = compute_bbox_from_alpha(fg)
                        if tight is None:
                            continue
                        x0, y0, tight_bbox_bg = try_place_object_in_card_area(bg, fg, placed_bboxes, MAX_TRY_PER_OBJECT)
                        if x0 is None:
                            continue
                        alpha_paste(bg, fg, x0, y0)
                        placed_bboxes.append(tight_bbox_bg)
                        objects.append({
                            "bbox": [round(coord, 4) for coord in tight_bbox_bg],  # 确保四位小数精度
                            "category_id": int(cls_id),
                            "category_name": cls_name,
                            "cutout_path": cut_path
                        })
                        if DRAW_VIZ:
                            draw_viz(vis, tight_bbox_bg, cls_name)
                        
                        placed_count += 1
                        class_counts[cls_name] += 1
                
                if len(objects) == 0:
                    continue
                
                images_created += 1
                total_objects_all += len(objects)
                out_name = f"mixed_{crop_idx+1:06d}.jpg"
                out_img_path = str(out_img_dir / out_name)
                cv2.imwrite(out_img_path, bg, [int(cv2.IMWRITE_JPEG_QUALITY), SAVE_JPG_QUALITY])
                if DRAW_VIZ and vis is not None:
                    out_viz_path = str(out_viz_dir / out_name.replace(".jpg", "_viz.jpg"))
                    cv2.imwrite(out_viz_path, vis, [int(cv2.IMWRITE_JPEG_QUALITY), SAVE_JPG_QUALITY])
                
                rec = {
                    "image_path": out_img_path,
                    "crop_path": crop_path,
                    "src_background": crop_rec.get("src_bg"),
                    "width": TARGET_W,
                    "height": TARGET_H,
                    "category_mixed": True,
                    "objects": objects
                }
                nd_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
            summary["classes"]["mixed"] = {"images": images_created, "objects": total_objects_all, "class_distribution": class_counts}
            
        else:
            # 原有的单类别模式
            print("\n== Single Class Synthesis Mode ==")
            for cls_name, cls_id in CLASS_NAME_TO_ID.items():
                density = DENSITY_PER_IMAGE.get(cls_name, 5)
                cutouts = list_cutouts_by_class(cutout_root, cls_name)
                if not cutouts:
                    print(f"[WARN] class {cls_name} 没有抠图，跳过")
                    continue

                out_img_dir = Path(synth_root) / cls_name / "images"
                out_viz_dir = Path(synth_root) / cls_name / "viz"
                ensure_dir(out_img_dir); ensure_dir(out_viz_dir)

                images_created = 0
                objects_total = 0
                img_idx = 0

                print(f"\n== Synthesizing class {cls_name} (density={density}) ==")
                for crop_rec in tqdm(crop_records, desc=f"Synth {cls_name}"):
                    crop_path = crop_rec["crop_path"]
                    bg = cv2.imread(crop_path)
                    if bg is None:
                        continue
                    H, W = bg.shape[:2]
                    if (W, H) != (TARGET_W, TARGET_H):
                        # safety fallback: ensure correct size by center crop/pad
                        bg, _ = crop_with_margins(bg, max(0,(W-TARGET_W)//2), max(0,(W-TARGET_W)//2), max(0,(H-TARGET_H)//2), max(0,(H-TARGET_H)//2), TARGET_W, TARGET_H)

                    vis = bg.copy() if DRAW_VIZ else None
                    placed_bboxes = []
                    objects = []

                    attempts = 0
                    while len(objects) < density and attempts < density * MAX_TRY_PER_OBJECT:
                        attempts += 1
                        cut_path = random.choice(cutouts)
                        fg = cv2.imread(cut_path, cv2.IMREAD_UNCHANGED)
                        if fg is None:
                            continue
                        fh, fw = fg.shape[:2]
                        # too large skip
                        if fw > W - 2 * BORDER_MARGIN or fh > H - 2 * BORDER_MARGIN:
                            continue
                        tight = compute_bbox_from_alpha(fg)
                        if tight is None:
                            continue
                        x0, y0, tight_bbox_bg = try_place_object_in_card_area(bg, fg, placed_bboxes, MAX_TRY_PER_OBJECT)
                        if x0 is None:
                            continue
                        alpha_paste(bg, fg, int(x0), int(y0))
                        placed_bboxes.append(tight_bbox_bg)
                        objects.append({
                            "bbox": [round(coord, 4) for coord in tight_bbox_bg],  # 确保四位小数精度
                            "category_id": int(cls_id),
                            "category_name": cls_name,
                            "cutout_path": cut_path
                        })
                        if DRAW_VIZ:
                            draw_viz(vis, tight_bbox_bg, cls_name)

                    if len(objects) == 0:
                        continue

                    img_idx += 1
                    images_created += 1
                    objects_total += len(objects)
                    out_name = f"{cls_name}_{img_idx:06d}.jpg"
                    out_img_path = str(out_img_dir / out_name)
                    cv2.imwrite(out_img_path, bg, [int(cv2.IMWRITE_JPEG_QUALITY), SAVE_JPG_QUALITY])
                    if DRAW_VIZ and vis is not None:
                        out_viz_path = str(out_viz_dir / out_name.replace(".jpg", "_viz.jpg"))
                        cv2.imwrite(out_viz_path, vis, [int(cv2.IMWRITE_JPEG_QUALITY), SAVE_JPG_QUALITY])

                    rec = {
                        "image_path": out_img_path,
                        "crop_path": crop_path,
                        "src_background": crop_rec.get("src_bg"),
                        "width": TARGET_W,
                        "height": TARGET_H,
                        "category_single": cls_name,
                        "category_id": int(cls_id),
                        "objects": objects
                    }
                    nd_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                summary["classes"][cls_name] = {"images": images_created, "objects": objects_total}

        nd_f.close()
        summary_path = Path(synth_root) / "synth_summary_cropped1088_margins.json"
        with open(summary_path, "w", encoding="utf-8") as sf:
            json.dump(summary, sf, indent=2, ensure_ascii=False)

        print("\n✅ Done.")
        print("placements ndjson:", str(placement_ndjson))
        print("summary:", str(summary_path))
        print("crops dir:", str(crops_dir))
        print("synth root:", synth_root)
        print(json.dumps(summary, indent=2, ensure_ascii=False))

    except FileNotFoundError as e:
        print(f"[ERROR] 文件未找到: {e}")
        return 1
    except PermissionError as e:
        print(f"[ERROR] 权限不足: {e}")
        return 1
    except Exception as e:
        print(f"[ERROR] 未预期的错误: {e}")
        return 1
    return 0

def validate_output(synth_root, expected_classes):
    """验证输出结果的完整性"""
    issues = []
    
    # 检查NDJSON文件
    ndjson_path = Path(synth_root) / "placements_cropped1088_margins.ndjson"
    if not ndjson_path.exists():
        issues.append("NDJSON文件缺失")
    
    # 检查各类别输出
    for cls_name in expected_classes:
        img_dir = Path(synth_root) / cls_name / "images"
        viz_dir = Path(synth_root) / cls_name / "viz"
        
        if not img_dir.exists():
            issues.append(f"{cls_name}: 图像目录缺失")
        elif len(list(img_dir.glob("*.jpg"))) == 0:
            issues.append(f"{cls_name}: 无输出图像")
            
        if DRAW_VIZ and not viz_dir.exists():
            issues.append(f"{cls_name}: 可视化目录缺失")
    
    return issues

if __name__ == "__main__":
    main()


