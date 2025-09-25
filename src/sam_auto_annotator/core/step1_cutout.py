"""
  不使用yaml配置的话
  conda activate sam39
  python /mnt/afs/xiemingjin/sam_project/tool/sam_step1_cutout.py \
    --data-root /mnt/afs/share_data/Pokemon/v0.2/data/raw_data/det_markers/20250902_pokemon_det_maskers_DIY02_cam1_greenbg_traindata/img_h \
    --sam-checkpoint /mnt/afs/xiemingjin/sam_project/checkpoints/sam_vit_h_4b8939.pth \
    --out-root /mnt/afs/xiemingjin/sam_project/saved/20250905_sam_step1_out \
    --device cuda \
    --save-masks-npy
脚本会为每种新形状生成： 
1.精确分割轮廓:SAM自动识别物体边缘  
2.透明背景抠图:保持原始形状的RGBA PNG  
3.边界框标注：自动计算最小外接矩形
4.可视化效果：在原图上显示分割结果和边界框
"""
"""
 - 在保持原流程与过滤/拆分/去重逻辑的前提下，新增：
   * 每个最终保留目标的透明背景抠图（RGBA PNG）
   * 尝试为 contour_recover 恢复得到的 bbox 生成掩码（白纸场景下效果好）
   * 输出 metadata_cutouts.ndjson（包含 cutout_path、mask_npy 等）
 - 仍然输出 COCO bbox-only JSON（用于训练检测器）
 - 保存 overlay_init（初次 merged mask）和 overlay_kept（最终保留）并在 overlay 标注 cutout 文件名

注意：
 - 本脚本在原逻辑基础上尽量少改参数；如果你想更强的抠图（更少碎片/漏检），调大 PAD_PX、或调整 profile 中 points_per_side、pred_iou_thresh 等。
"""
import os
import cv2
import json
import math
import shutil
import argparse
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict,Counter

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def detect_background_color(image, sample_size=50):
    """
    检测图像背景颜色（白色或绿色）
    """
    h, w = image.shape[:2]
    
    # 采样四个边缘的像素
    edges_pixels = []
    edges_pixels.extend(image[:sample_size, :].reshape(-1, 3))
    edges_pixels.extend(image[h-sample_size:, :].reshape(-1, 3))
    edges_pixels.extend(image[:, :sample_size].reshape(-1, 3))
    edges_pixels.extend(image[:, w-sample_size:].reshape(-1, 3))
    
    edges_pixels = np.array(edges_pixels)
    edges_pixels_hsv = cv2.cvtColor(edges_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    
    white_count = 0
    green_count = 0
    
    for pixel_hsv in edges_pixels_hsv:
        h_val, s, v = pixel_hsv
        if s < 30 and v > 200:  # 白色
            white_count += 1
        elif 40 <= h_val <= 80 and s > 50:  # 绿色
            green_count += 1
    
    total = len(edges_pixels_hsv)
    if white_count / total > 0.6:
        return 'white'
    elif green_count / total > 0.4:
        return 'green'
    else:
        return 'white'  # 默认白色背景

# -------------------------（用户配置区）-------------------------
DATASET_PATH = "/mnt/afs/xiemingjin/pkm_dataset/v0.2/20250805_pokemon_det_maskers_DIY02_cam1_occ_traindata/img_h"
CROPPED_IMAGE_ROOT = "/mnt/afs/xiemingjin/sam_project/20250905_v12_cropped1088"
OUTPUT_JSON_PATH = "/mnt/afs/xiemingjin/sam_project/coco/20250905_pokemon_annotations_v12_crop1088.json"

ENABLE_VISUALIZATION = True
VIS_OUTPUT_DIR = "/mnt/afs/xiemingjin/sam_project/scripts/20250905_visualization_output_v12_crop1088"
MAX_VIZ_IMAGES = -1

SAM_CHECKPOINT = "/mnt/afs/xiemingjin/sam_project/checkpoints/sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CUTOUTS_DIRNAME = "cutouts"   # 保存透明抠图的子目录
METADATA_NDJSON = "metadata_cutouts.ndjson"

CATEGORY_MAPPING = {
    "Coin_H": 1, "Coin_T": 2,
    "Markers_Damage010": 3, "Markers_Damage050": 4, "Markers_Damage100": 5,
    "Markers_Poisoned": 6, "Markers_Burned": 7,
    "Markers_GX_Avaiable": 8, "Markers_GX_Used": 9
}

# SAM/阈值参数（保持原逻辑，只略作注释）
ENABLE_PREPROCESSING = True
ENABLE_CONTENT_FILTER = True
CONTENT_FILTER_STD_DEV_THRESHOLD = 13.0

NMS_IOU_THRESHOLD = 0.30
NESTED_CONTAINMENT_THRESHOLD = 0.80

ENABLE_PADDED_SAM = True
PAD_PX = 52
               
CATEGORY_PROFILES = {
    'large_objects_profile': {
        'sam_params': {"points_per_side": 20, "pred_iou_thresh": 0.5, "stability_score_thresh": 0.5},
        'filter_params': {"min_area": 2800, "max_area": 13500, "max_aspect_ratio": 1.1}
    },
    'small_objects_profile': {
        'sam_params': {"points_per_side": 60, "pred_iou_thresh": 0.6, "stability_score_thresh": 0.6},
        'filter_params': {"min_area": 400, "max_area": 800, "max_aspect_ratio": 1.15}
    },
    'large_mid_objects_profile': {
        'sam_params': {"points_per_side": 8, "pred_iou_thresh": 0.6, "stability_score_thresh": 0.6},
        'filter_params': {"min_area": 5000, "max_area": 18000, "max_aspect_ratio": 2}
    },
    'middle_objects_profile': {
        'sam_params': {"points_per_side": 30, "pred_iou_thresh": 0.6, "stability_score_thresh": 0.5},
        'filter_params': {"min_area": 1900, "max_area": 2800, "max_aspect_ratio": 1.1}
    },
    'small_large_objects_profile': {
        'sam_params': {"points_per_side": 50, "pred_iou_thresh": 0.6, "stability_score_thresh": 0.6},
        'filter_params': {"min_area": 1300, "max_area": 1850, "max_aspect_ratio": 1.1}
    },
    'small_mid_objects_profile': {
        'sam_params': {"points_per_side": 60, "pred_iou_thresh": 0.6, "stability_score_thresh": 0.5},
        'filter_params': {"min_area": 700, "max_area": 1100, "max_aspect_ratio": 1.1}
    },

}

CATEGORY_TO_PROFILE_MAPPING = {
    "Coin_H": 'large_objects_profile',
    "Coin_T": 'large_objects_profile',
    "Markers_Damage010": 'small_objects_profile',
    "Markers_Damage050": 'small_mid_objects_profile',
    "Markers_Damage100": 'small_large_objects_profile',
    "Markers_Poisoned": 'middle_objects_profile',
    "Markers_Burned": 'middle_objects_profile',
    "Markers_GX_Avaiable": 'large_mid_objects_profile',
    "Markers_GX_Used": 'large_mid_objects_profile'
}

# ------------------------- Monkey patch -------------------------
from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator as _SAMGen
_old_process_crop = _SAMGen._process_crop
def _patched_process_crop(self, image, crop_box, crop_layer_idx, orig_size):
    masks = _old_process_crop(self, image, crop_box, crop_layer_idx, orig_size)
    device = (next(self.predictor.model.parameters()).device
              if hasattr(self, "predictor") and hasattr(self, "model")
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if hasattr(masks, "_stats"):
        s = masks._stats
        for k in ("boxes", "masks", "iou_preds"):
            if k in s and isinstance(s[k], torch.Tensor):
                s[k] = s[k].to(device)
    return masks
_SAMGen._process_crop = _patched_process_crop
print("✅ Monkey Patch：_process_crop 设备一致性修复")

# ------------------------- 保留原来的工具函数（未改动） -------------------------
TARGET_W, TARGET_H = 1088, 1088
H_ALIGN = 'center'

def preprocess_image(image_bgr):
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def create_coco_structure():
    categories_list = [{"supercategory": k, "id": v, "name": str(v)}
                       for k, v in CATEGORY_MAPPING.items()]
    return {"info": {"description": "Pokemon Markers Dataset v12 crop1088", "version": "12.0"},
            "licenses": [], "images": [], "annotations": [], "categories": categories_list}

def box_area_xywh(b): return float(b[2] * b[3])

def iou_xywh(b1, b2):
    x1, y1, w1, h1 = b1; x2, y2, w2, h2 = b2
    xa1, ya1, xa2, ya2 = x1, y1, x1 + w1, y1 + h1
    xb1, yb1, xb2, yb2 = x2, y2, x2 + w2, y2 + h2
    inter = max(0, min(xa2, xb2) - max(xa1, xb1)) * max(0, min(ya2, yb2) - max(ya1, yb1))
    union = w1*h1 + w2*h2 - inter + 1e-6
    return inter / union

def containment_ratio(inner, outer):
    x, y, w, h = inner
    xi1, yi1 = max(x, outer[0]), max(y, outer[1])
    xi2, yi2 = min(x+w, outer[0]+outer[2]), min(y+h, outer[1]+outer[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    return inter / (w*h + 1e-6)

def calculate_std_dev(image_gray, bbox):
    x, y, w, h = [int(c) for c in bbox]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(image_gray.shape[1], x + w), min(image_gray.shape[0], y + h)
    if (y2 - y1) <= 0 or (x2 - x1) <= 0: return 0.0
    roi = image_gray[y1:y2, x1:x2]
    return float(np.std(roi))

def crop_seg_to_bbox(seg):
    ys, xs = np.where(seg)
    if xs.size == 0 or ys.size == 0: return None
    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()), int(ys.max())
    return [float(x1), float(y1), float(x2-x1+1), float(y2-y1+1)]

def sam_generate_with_padding(rgb, mask_generator, pad_px):
    H, W = rgb.shape[:2]
    if not ENABLE_PADDED_SAM or pad_px <= 0:
        masks = mask_generator.generate(rgb)
        out = []
        for m in masks:
            seg = m['segmentation'].astype(np.uint8)
            bbox = crop_seg_to_bbox(seg)
            if bbox is None:
                continue
            m2 = dict(m)
            m2['segmentation'] = seg
            m2['bbox'] = bbox
            m2['area'] = float(seg.sum())
            out.append(m2)
        return out

    pad = int(pad_px)
    padded = cv2.copyMakeBorder(rgb, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    masks = mask_generator.generate(padded)

    out = []
    for m in masks:
        seg_pad = m['segmentation'].astype(np.uint8)
        seg = seg_pad[pad:pad+H, pad:pad+W]
        if seg.sum() == 0: continue
        bbox = crop_seg_to_bbox(seg)
        if bbox is None: continue
        m2 = dict(m)
        m2['segmentation'] = seg
        m2['bbox'] = bbox
        m2['area'] = float(seg.sum())
        out.append(m2)
    return out

def smart_box_selector(annotations, nms_iou_thresh, nested_containment_thresh):
    if not annotations: return []
    boxes = list(annotations)
    discard = set()
    for i in range(len(boxes)):
        if i in discard: continue
        for j in range(i+1, len(boxes)):
            if j in discard: continue
            bi, bj = boxes[i]['bbox'], boxes[j]['bbox']
            pi, pj = boxes[i].get('predicted_iou', 0.5), boxes[j].get('predicted_iou', 0.5)
            iou = iou_xywh(bi, bj)
            c_j_in_i = containment_ratio(bj, bi)
            c_i_in_j = containment_ratio(bi, bj)
            drop = -1
            if iou > nms_iou_thresh or c_j_in_i > nested_containment_thresh or c_i_in_j > nested_containment_thresh:
                drop = j if pi > pj else i
            if drop != -1:
                discard.add(drop)
                if drop == i: break
    return [b for idx, b in enumerate(boxes) if idx not in discard]

def anti_merge_suppression(boxes,
                           contain_thr_each=0.62,
                           area_sum_ratio=0.70,
                           min_children=2,
                           iou_pair_thr=0.10):
    if len(boxes) < 2: return boxes
    keep = [True] * len(boxes)
    areas = [box_area_xywh(b['bbox']) for b in boxes]

    for i, big in enumerate(boxes):
        if not keep[i]: continue
        big_box = big['bbox']; big_area = areas[i]
        children = []
        for j, small in enumerate(boxes):
            if i == j or not keep[j]: continue
            small_box = small['bbox']
            if containment_ratio(small_box, big_box) >= contain_thr_each and \
               iou_xywh(small_box, big_box) >= iou_pair_thr and \
               areas[j] < big_area:
                children.append(j)
        if len(children) >= min_children:
            sum_small = sum(areas[c] for c in children)
            if sum_small / (big_area + 1e-6) >= area_sum_ratio:
                keep[i] = False
    return [b for k, b in zip(keep, boxes) if k]

def split_disconnected_by_cc(candidates, fp, min_comp_area_factor=0.50,
                             sum_area_keep_ratio=0.80):
    out = []
    min_comp_area = fp['min_area'] * float(min_comp_area_factor)
    max_ar = fp['max_aspect_ratio']
    kernel = np.ones((3,3), np.uint8)

    for m in candidates:
        seg = m.get('segmentation', None)
        if seg is None:
            out.append(m); continue

        x, y, w, h = [int(c) for c in m['bbox']]
        sub = seg[y:y+h, x:x+w].astype(np.uint8)
        if sub.sum() == 0:
            out.append(m); continue

        sub_proc = cv2.morphologyEx(sub, cv2.MORPH_OPEN, kernel, iterations=1)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(sub_proc, connectivity=8)
        if num < 3:
            out.append(m); continue

        comps = []
        comp_area_sum = 0.0
        for k in range(1, num):
            xk, yk, wk, hk, ak = stats[k]
            if ak < min_comp_area:
                continue
            ar = max(wk, hk) / (min(wk, hk) + 1e-6)
            if ar > max_ar:
                continue
            comps.append((xk, yk, wk, hk, ak))
            comp_area_sum += float(ak)

        if len(comps) >= 2 and comp_area_sum >= sum_area_keep_ratio * float(sub.sum()):
            for (xk, yk, wk, hk, ak) in comps:
                full_mask = np.zeros_like(seg, dtype=np.uint8)
                # labels' origin is inside sub_proc; map to full mask positions
                comp_label = labels[yk, xk]
                full_mask[y:y+h, x:x+w][labels == comp_label] = 1
                out.append({
                    **m,
                    "bbox": [float(x + xk), float(y + yk), float(wk), float(hk)],
                    "area": float(ak),
                    "segmentation": full_mask
                })
        else:
            out.append(m)
    return out

def recover_missing_by_contours(image_bgr, existing_boxes, fp):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 31, 5)
    thr = cv2.medianBlur(thr, 3)

    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_a = fp['min_area'] * 0.85
    max_a = fp['max_area'] * 1.25
    new_boxes = []

    def overlaps_any(b, others, thr=0.25):
        return any(iou_xywh(b, o['bbox']) >= thr for o in others)

    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_a or area > max_a:
            continue
        peri = cv2.arcLength(c, True) + 1e-6
        circularity = 4.0 * math.pi * area / (peri * peri)
        if circularity < 0.60:
            continue
        x, y, w, h = cv2.boundingRect(c)
        candidate = [float(x), float(y), float(w), float(h)]
        if not overlaps_any(candidate, existing_boxes, thr=0.20):
            new_boxes.append({
                "bbox": candidate,
                "area": float(area),
                "predicted_iou": 0.50,
                "source": "contour_recover"
            })
    return new_boxes

# ------------------------- --- ADDED/CHANGED: 抠图和掩码恢复函数 --- -------------------------
def mask_to_cutout_rgba(img_rgb, seg_bool, bbox):
    """
    将指定全图大小的 boolean seg 转为 bbox 裁剪后的 BGRA (uint8) 图像（alpha=mask）
    img_rgb: RGB uint8
    seg_bool: boolean ndarray same HxW
    bbox: [x,y,w,h] (可以用来裁出更紧的 ROI)
    """
    x, y, w, h = [int(v) for v in bbox]
    H, W = img_rgb.shape[:2]
    x1 = max(0, x); y1 = max(0, y)
    x2 = min(W, x + w); y2 = min(H, y + h)
    if x2 <= x1 or y2 <= y1:
        return None
    crop_rgb = img_rgb[y1:y2, x1:x2]
    crop_mask = (seg_bool[y1:y2, x1:x2].astype(np.uint8) * 255)
    # ensure 3 channels BGR then convert to BGRA
    crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
    crop_bgra = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2BGRA)
    crop_bgra[:, :, 3] = crop_mask
    return crop_bgra

def recover_mask_from_bbox_on_white(img_bgr, bbox, background_color='white'):
    """
    给定整张裁剪图 (BGR) 和 bbox（xywh float），自适应背景色提取目标掩码。
    返回 boolean mask (HxW) 或 None。
    """
    H, W = img_bgr.shape[:2]
    x, y, w, h = [int(round(v)) for v in bbox]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)
    if x2 <= x1 or y2 <= y1:
        return None

    roi = img_bgr[y1:y2, x1:x2]
    
    if background_color == 'green':
        # 绿色背景处理
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        green_mask = cv2.inRange(roi_hsv, lower_green, upper_green)
        th1 = cv2.bitwise_not(green_mask)  # 反转：非绿色为目标
        
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        th1 = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kern, iterations=2)
    else:
        # 白色背景处理（原逻辑）
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        th1 = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kern, iterations=1)

    # 如果结果太小，尝试自适应阈值
    if th1.sum() < 50:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        if background_color == 'green':
            th2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 15, 5)
        else:
            th2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        th2 = cv2.morphologyEx(th2, cv2.MORPH_OPEN, kern, iterations=1)
        mask_local = th2
    else:
        mask_local = th1

    # 选择最大轮廓作为目标
    cnts, _ = cv2.findContours(mask_local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    
    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < 20:
        return None

    mask_roi = np.zeros_like(mask_local, dtype=np.uint8)
    cv2.drawContours(mask_roi, [cnt], -1, 255, thickness=-1)
    
    # 映射到全尺寸掩码
    full_mask = np.zeros((H, W), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = (mask_roi // 255).astype(np.uint8)
    
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_CLOSE, kern, iterations=1)
    return full_mask.astype(bool)

# ------------------------- 剩余主流程保持原来代码的逻辑，只在写入 COCO 与可视化地方新增 cutout 保存 -------------------------
def bottom_aligned_center_lr_crop(img, tw=1088, th=1088, h_align='center'):
    H, W = img.shape[:2]
    pad_top = pad_bottom = pad_left = pad_right = 0
    if H < th:
        pad_top = th - H
    if W < tw:
        need = tw - W
        pad_left = need // 2
        pad_right = need - pad_left
    if pad_top or pad_bottom or pad_left or pad_right:
        img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT)
        H, W = img.shape[:2]
    y2 = H
    y1 = max(0, H - th)
    if W == tw:
        x1, x2 = 0, W
    else:
        if h_align == 'left':
            x1 = 0
            x2 = x1 + tw
        elif h_align == 'right':
            x2 = W
            x1 = x2 - tw
        else:
            x1 = max(0, (W - tw) // 2)
            x2 = x1 + tw
        if x2 > W:
            x2 = W; x1 = W - tw
        if x1 < 0:
            x1 = 0; x2 = tw
    return img[y1:y2, x1:x2].copy()

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def run_step1_cutout(
    input_images: str,
    output_dir: str, 
    sam_checkpoint: str,
    config,
    device: str = "cuda",
    save_masks_npy: bool = False,
    pad_px: int = None,
    **kwargs
) -> dict:
    """
    Step1: SAM分割抠图主函数 - 供pipeline调用
    
    Args:
        input_images: 输入图像目录路径
        output_dir: 输出目录路径
        sam_checkpoint: SAM模型检查点路径
        config: 配置对象
        device: 设备类型
        save_masks_npy: 是否保存mask的npy文件
        pad_px: 填充像素数，如果为None则使用配置中的值
        **kwargs: 其他参数
        
    Returns:
        dict: 处理结果统计
    """
    from pathlib import Path
    
    data_root = Path(input_images)
    out_root = Path(output_dir)
    ensure_dir(out_root)
    
    # 从配置获取参数
    step1_config = config.step1
    category_mapping = config.config_manager.get_category_mapping()
    category_profiles = config.config_manager.get_category_profiles()
    category_to_profile_mapping = config.config_manager.get_category_to_profile_mapping()
    
    # 使用配置中的参数
    if pad_px is None:
        pad_px = step1_config.pad_px
    
    # 创建输出目录
    cropped_image_root = out_root / "cropped_images"
    os.makedirs(cropped_image_root, exist_ok=True)
    
    if config.global_config.verbose:
        vis_output_dir = out_root / "visualization"
        os.makedirs(vis_output_dir, exist_ok=True)

    # cutouts root
    cutouts_root = out_root / step1_config.cutout_dir_name
    ensure_dir(cutouts_root)

    metadata_lines = []

    print(f"设备={device}, PAD={pad_px}, save_masks_npy={save_masks_npy}")
    
    # SAM init
    sam = sam_model_registry[step1_config.model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    coco = create_coco_structure()
    image_id, ann_id, viz_cnt = 0, 0, 0
    
    stats = {
        'processed_categories': 0,
        'processed_images': 0,
        'generated_cutouts': 0,
        'generated_annotations': 0
    }

    for cat_name, cat_id in category_mapping.items():
        src_dir = data_root / cat_name
        if not src_dir.is_dir():
            print(f"⚠️ 跳过：{src_dir} 不存在")
            continue

        profile_name = category_to_profile_mapping.get(cat_name)
        if not profile_name:
            print(f"⚠️ 跳过：未定义类别 '{cat_name}' 的profile")
            continue
            
        profile = category_profiles[profile_name]
        fp = profile['filter_params']
        mask_generator = SamAutomaticMaskGenerator(model=sam, **profile['sam_params'])

        dst_dir = cropped_image_root / cat_name
        dst_dir.mkdir(exist_ok=True)

        cutout_class_dir = cutouts_root / cat_name
        ensure_dir(cutout_class_dir)

        files = [f for f in src_dir.iterdir() if f.suffix.lower() in ['.png','.jpg','.jpeg']]
        print(f"\n处理 '{cat_name}'（{profile_name}），共 {len(files)} 张")
        
        stats['processed_categories'] += 1

        for img_path in tqdm(files, desc=f"裁剪+标注 {cat_name}"):
            img = cv2.imread(str(img_path))
            background_color = detect_background_color(img)
            if img is None:
                continue

            stats['processed_images'] += 1
            
            # 1 crop
            cropped = bottom_aligned_center_lr_crop(img, TARGET_W, TARGET_H, H_ALIGN)
            proc = preprocess_image(cropped) if ENABLE_PREPROCESSING else cropped
            rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)

            # 2 SAM with padding
            masks = sam_generate_with_padding(rgb, mask_generator, pad_px)

            # 3 initial filter
            candidates = []
            for m in masks:
                seg = m['segmentation']
                bbox = crop_seg_to_bbox(seg)
                if bbox is None:
                    continue
                x, y, w, h = bbox
                area = w * h
                aspect_ratio = max(w, h) / min(w, h)
                
                if (fp['min_area'] <= area <= fp['max_area'] and 
                    aspect_ratio <= fp['max_aspect_ratio']):
                    candidates.append({
                        'bbox': bbox,
                        'segmentation': seg,
                        'area': area,
                        'predicted_iou': m.get('predicted_iou', 0),
                        'stability_score': m.get('stability_score', 0)
                    })

            if not candidates:
                continue

            # 4 connected components split
            candidates = split_disconnected_by_cc(candidates, fp)

            # 5 smart box selector (NMS + nested containment)
            kept_boxes = smart_box_selector(
                candidates, 
                NMS_IOU_THRESHOLD, 
                NESTED_CONTAINMENT_THRESHOLD
            )

            # 6 anti-merge suppression
            kept_boxes = anti_merge_suppression(kept_boxes)

            # 7 content filter
            if ENABLE_CONTENT_FILTER:
                gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
                filtered_boxes = []
                for box_info in kept_boxes:
                    bbox = box_info['bbox']
                    std_dev = calculate_std_dev(gray, bbox)
                    if std_dev >= CONTENT_FILTER_STD_DEV_THRESHOLD:
                        filtered_boxes.append(box_info)
                kept_boxes = filtered_boxes

            # 8 contour recovery
            kept_boxes.extend(recover_missing_by_contours(proc, kept_boxes, fp))

            # 9 save cutouts and annotations
            fname_base = img_path.stem
            for i, box_info in enumerate(kept_boxes):
                bbox = box_info['bbox']
                seg = box_info.get('segmentation')
                
                # Generate cutout
                if seg is not None:
                    cutout_rgba = mask_to_cutout_rgba(rgb, seg, bbox)
                else:
                    # Fallback: try to recover mask from bbox
                    recovered_mask = recover_mask_from_bbox_on_white(proc, bbox, background_color)
                    if recovered_mask is not None:
                        cutout_rgba = mask_to_cutout_rgba(rgb, recovered_mask, bbox)
                    else:
                        continue

                if cutout_rgba is None:
                    continue

                # Save cutout
                cutout_filename = f"{fname_base}_{i:03d}.png"
                cutout_path = cutout_class_dir / cutout_filename
                cv2.imwrite(str(cutout_path), cutout_rgba)
                stats['generated_cutouts'] += 1

                # Save metadata
                metadata_lines.append({
                    'cutout_path': str(cutout_path),
                    'original_image': str(img_path),
                    'category': cat_name,
                    'bbox': bbox,
                    'area': box_info['area']
                })

                # Add COCO annotation
                x, y, w, h = bbox
                coco['annotations'].append({
                    'id': ann_id,
                    'image_id': image_id,
                    'category_id': cat_id,
                    'bbox': [x, y, w, h],
                    'area': w * h,
                    'iscrowd': 0
                })
                ann_id += 1
                stats['generated_annotations'] += 1

            # Add COCO image info
            if kept_boxes:
                coco['images'].append({
                    'id': image_id,
                    'file_name': f"{cat_name}/{fname_base}.jpg",
                    'width': TARGET_W,
                    'height': TARGET_H
                })
                
                # Save cropped image
                dst_path = dst_dir / f"{fname_base}.jpg"
                cv2.imwrite(str(dst_path), cropped)
                
                image_id += 1

    # Save results
    output_json_path = out_root / "annotations.json"
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)
    
    metadata_ndjson_path = out_root / step1_config.metadata_ndjson_name
    with open(metadata_ndjson_path, 'w', encoding='utf-8') as f:
        for line in metadata_lines:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
    
    print(f"✅ Step1完成，处理了 {stats['processed_categories']} 个类别，{stats['processed_images']} 张图像")
    print(f"   生成了 {stats['generated_cutouts']} 个抠图，{stats['generated_annotations']} 个标注")
    
    return stats

def build_argparser():
    p = argparse.ArgumentParser(description="SAM cutout + bbox pipeline (v2) producing transparent cutouts")
    p.add_argument("--data-root", required=True)
    p.add_argument("--sam-checkpoint", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--device", default=DEVICE)
    p.add_argument("--pad-px", type=int, default=PAD_PX)
    p.add_argument("--save-masks-npy", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p

def main():
    args = build_argparser().parse_args()
    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    ensure_dir(out_root)
    os.makedirs(CROPPED_IMAGE_ROOT, exist_ok=True)
    if ENABLE_VISUALIZATION:
        os.makedirs(VIS_OUTPUT_DIR, exist_ok=True)

    # cutouts root
    cutouts_root = out_root / CUTOUTS_DIRNAME
    ensure_dir(cutouts_root)

    metadata_lines = []

    print(f"设备={args.device}, PAD={args.pad_px}, save_masks_npy={args.save_masks_npy}")
    # SAM init
    sam = sam_model_registry[MODEL_TYPE](checkpoint=args.sam_checkpoint)
    sam.to(device=args.device)

    coco = create_coco_structure()
    image_id, ann_id, viz_cnt = 0, 0, 0

    for cat_name, cat_id in CATEGORY_MAPPING.items():
        src_dir = os.path.join(args.data_root, cat_name)
        if not os.path.isdir(src_dir):
            print(f"⚠️ 跳过：{src_dir} 不存在")
            continue

        profile_name = CATEGORY_TO_PROFILE_MAPPING.get(cat_name)
        if not profile_name:
            print(f"⚠️ 跳过：未定义类别 '{cat_name}' 的profile")
            continue
        profile = CATEGORY_PROFILES[profile_name]
        fp = profile['filter_params']
        mask_generator = SamAutomaticMaskGenerator(model=sam, **profile['sam_params'])

        dst_dir = os.path.join(CROPPED_IMAGE_ROOT, cat_name)
        os.makedirs(dst_dir, exist_ok=True)

        cutout_class_dir = cutouts_root / cat_name
        ensure_dir(cutout_class_dir)

        files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))]
        print(f"\n处理 '{cat_name}'（{profile_name}），共 {len(files)} 张")

        for fname in tqdm(files, desc=f"裁剪+标注 {cat_name}"):
            src_path = os.path.join(src_dir, fname)
            img = cv2.imread(src_path)
            background_color=detect_background_color(img)
            if img is None:
                continue

            # 1 crop
            cropped = bottom_aligned_center_lr_crop(img, TARGET_W, TARGET_H, H_ALIGN)
            proc = preprocess_image(cropped) if ENABLE_PREPROCESSING else cropped
            rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)

            # 2 SAM with padding
            masks = sam_generate_with_padding(rgb, mask_generator, args.pad_px)

            # 3 initial filter
            candidates = []
            for m in masks:
                x, y, w, h = m['bbox']; area = m['area']
                ar = max(w, h) / (min(w, h) + 1e-6)
                if fp['min_area'] <= area <= fp['max_area'] and ar <= fp['max_aspect_ratio']:
                    candidates.append(m)

            # 4 split connected components, anti-merge, nms-like selection, content filter (same as original)
            candidates = split_disconnected_by_cc(candidates, fp,
                                                  min_comp_area_factor=0.50,
                                                  sum_area_keep_ratio=0.80)
            candidates = anti_merge_suppression(candidates,
                                                contain_thr_each=0.62,
                                                area_sum_ratio=0.70,
                                                min_children=2,
                                                iou_pair_thr=0.10)
            selected = smart_box_selector(candidates, NMS_IOU_THRESHOLD, NESTED_CONTAINMENT_THRESHOLD)

            final_boxes = []
            if ENABLE_CONTENT_FILTER:
                gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                for a in selected:
                    s = calculate_std_dev(gray, a['bbox'])
                    if s >= CONTENT_FILTER_STD_DEV_THRESHOLD:
                        a['std_dev'] = s
                        final_boxes.append(a)
            else:
                final_boxes = selected

            # 5 recover missing via contour, then re-run anti_merge/smart selection
            recovered = recover_missing_by_contours(cropped, final_boxes, fp)
            if recovered:
                mix = final_boxes + recovered
                mix = anti_merge_suppression(mix,
                                             contain_thr_each=0.62,
                                             area_sum_ratio=0.70,
                                             min_children=2,
                                             iou_pair_thr=0.10)
                final_boxes = smart_box_selector(mix, NMS_IOU_THRESHOLD, NESTED_CONTAINMENT_THRESHOLD)

            # 6 save cropped image
            base, ext = os.path.splitext(fname)
            out_name = f"{base}_crop1088{ext.lower()}"
            dst_path = os.path.join(dst_dir, out_name)
            cv2.imwrite(dst_path, cropped)

            # prepare metadata entry (for ndjson)
            meta_entry = {
                "image_path": os.path.abspath(dst_path),
                "class_name": cat_name,
                "class_id": int(cat_id),
                "width": TARGET_W,
                "height": TARGET_H,
                "objects": []
            }

            # overlay init and kept visuals
            vis_init = cropped.copy()
            # draw all candidate contours/bboxes (initial merged)
            for c in candidates:
                seg = c.get('segmentation', None)
                if seg is not None:
                    contours, _ = cv2.findContours((seg.astype(np.uint8)*255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(vis_init, contours, -1, (120,120,255), 1)
                x,y,w,h = [int(round(_) ) for _ in c['bbox']]
                cv2.rectangle(vis_init, (x,y), (x+w, y+h), (200,200,0), 1)

            # 7 对每个 final_box 生成 cutout（优先用 SAM mask，否则用 recover_mask_from_bbox_on_white）
            for idx, b in enumerate(final_boxes):
                seg = b.get('segmentation', None)
                bbox = [float(v) for v in b['bbox']]
                # if segmentation missing (likely from contour_recover), try to recover mask in bbox
                seg_bool = None
                if seg is None or seg.sum() == 0:
                    seg_try = recover_mask_from_bbox_on_white(cropped, bbox, background_color)
                    if seg_try is not None and seg_try.sum() > 0:
                        seg_bool = seg_try
                    else:
                        # fallback: create mask by simple non-white threshold across bbox
                        seg_try2 = recover_mask_from_bbox_on_white(cropped, bbox, background_color)
                        if seg_try2 is not None and seg_try2.sum() > 0:
                            seg_bool = seg_try2
                        else:
                            seg_bool = None
                else:
                    seg_bool = seg.astype(bool)

                cutout_fname = f"{base}_m{idx:03d}.png"
                cutout_path = str(cutout_class_dir / cutout_fname)

                if seg_bool is not None:
                    cut_bgra = mask_to_cutout_rgba(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB), seg_bool, bbox)
                    if cut_bgra is not None:
                        # write BGRA PNG
                        cv2.imwrite(cutout_path, cut_bgra)
                        mask_npy = None
                        if args.save_masks_npy:
                            mask_npy = str(cutout_class_dir / f"{base}_m{idx:03d}.npy")
                            np.save(mask_npy, seg_bool.astype(np.uint8))
                    else:
                        # fallback: save tight rect crop with opaque alpha
                        x, y, w, h = [int(round(v)) for v in bbox]
                        x1, y1 = max(0, x), max(0, y)
                        x2, y2 = min(TARGET_W, x + w), min(TARGET_H, y + h)
                        crop_rgb = cv2.cvtColor(cropped[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
                        crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
                        crop_bgra = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2BGRA)
                        crop_bgra[:, :, 3] = 255
                        cv2.imwrite(cutout_path, crop_bgra)
                        mask_npy = None
                else:
                    # no segmentation at all -> save rect crop fully opaque
                    x, y, w, h = [int(round(v)) for v in bbox]
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(TARGET_W, x + w), min(TARGET_H, y + h)
                    crop_rgb = cv2.cvtColor(cropped[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
                    crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
                    crop_bgra = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2BGRA)
                    crop_bgra[:, :, 3] = 255
                    cv2.imwrite(cutout_path, crop_bgra)
                    mask_npy = None

                # record meta, and also COCO annotation
                image_id += 1
                coco["images"].append({
                    "height": int(TARGET_H), "width": int(TARGET_W),
                    "id": image_id,
                    "file_name": os.path.abspath(dst_path)
                })
                ann_id += 1
                bbox_f = [float(v) for v in b['bbox']]
                coco["annotations"].append({
                    "bbox": bbox_f,
                    "area": float(b.get('area', box_area_xywh(bbox_f))),
                    "image_id": image_id,
                    "category_id": int(cat_id),
                    "id": ann_id,
                    "segmentation": [],      # detection training only
                    "iscrowd": 0
                })

                # annotate overlay_kept
                x, y, w, h = [int(round(v)) for v in b['bbox']]
                if b.get('segmentation', None) is not None:
                    # draw contour
                    seg_vis = (b['segmentation'].astype(np.uint8) * 255)
                    contours, _ = cv2.findContours(seg_vis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(vis_init, contours, -1, (0, 255, 0), 2)
                cv2.rectangle(vis_init, (x, y), (x+w, y+h), (255, 0, 0), 2)
                label_txt = f"{cat_name}_{idx}"
                cv2.putText(vis_init, label_txt, (x, max(12, y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

                obj_meta = {
                    "cutout_path": os.path.abspath(cutout_path),
                    "mask_npy": mask_npy,
                    "bbox_xywh": [float(v) for v in b['bbox']],
                    "area": float(b.get('area', box_area_xywh(b['bbox']))),
                    "predicted_iou": float(b.get('predicted_iou', -1.0)),
                    "stability_score": float(b.get('stability_score', -1.0)),
                    "source": b.get('source', 'sam')
                }
                meta_entry['objects'].append(obj_meta)

            # save overlay image for this crop (kept view)
            if ENABLE_VISUALIZATION:
                out_vis_dir = Path(VIS_OUTPUT_DIR)
                out_vis_dir.mkdir(parents=True, exist_ok=True)
                viz_name = f"{cat_name}_{out_name}"
                cv2.imwrite(str(out_vis_dir / viz_name), vis_init)

            # append metadata line
            metadata_lines.append(meta_entry)

    # 写入 metadata ndjson
    meta_out_path = out_root / METADATA_NDJSON
    with open(meta_out_path, "w", encoding="utf-8") as mf:
        for line in metadata_lines:
            mf.write(json.dumps(line, ensure_ascii=False) + "\n")

    # 写 COCO json
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(coco, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 完成。COCO: {OUTPUT_JSON_PATH}")
    print(f"抠图 metadata: {meta_out_path}")
    print(f"抠图区根: {cutouts_root}")
    if ENABLE_VISUALIZATION:
        print(f"可视化: {VIS_OUTPUT_DIR}")

if __name__ == "__main__":
    main()


