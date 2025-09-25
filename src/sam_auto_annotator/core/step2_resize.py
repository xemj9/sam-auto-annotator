"""
step2_resize.py
# 启用类别特定放大
python /mnt/afs/xiemingjin/sam_project/tool/step2_resize.py \
  --cutout-root /mnt/afs/xiemingjin/sam_project/saved/20250905_sam_step1_out/cutouts \
  --output-root /mnt/afs/xiemingjin/sam_project/saved/resized/20250919_cutouts_0.1_v1 \
  --pixel-density 1.554 \
  --enable-boost \

# 不启用放大（保持原有行为）
python step2_resize.py \
  --cutout-root /mnt/afs/xiemingjin/sam_project/saved/_sam_step1_out/cutouts \
  --output-root /mnt/afs/xiemingjin/sam_project/saved/resized/20250912_cutouts_normal \
  --pixel-density 1.554

# 详细模式（推荐调试时使用）
python step2_resize.py \
  --cutout-root /path/to/step1/cutouts \
  --output-root /path/to/resized/cutouts \
  --pixel-density 1.554 \
  --verbose

# 使用配置文件
python step2_resize.py \
  --cutout-root /path/to/step1/cutouts \
  --output-root /path/to/resized/cutouts \
  --config resize_config.yaml \
  --verbose

# 预览模式（不实际处理）
python step2_resize.py \
  --cutout-root /path/to/step1/cutouts \
  --output-root /path/to/resized/cutouts \
  --dry-run --verbose
"""
"""
根据物理尺寸将step1抠出的图案缩放到适配背景图的正确尺寸
特别优化处理不规则裁剪的情况（如圆形可能是53x56像素）

核心公式：1mm / 1.554px = physical_size_mm / target_pixel_size
即：target_pixel_size = physical_size_mm * 1.554

特性：
 - 支持圆形、长方形物体的智能缩放
 - 处理硬币的两种尺寸（大62mm，小30mm）
 - 基于物理尺寸和像素密度的精确计算
 - 特别处理不规则裁剪情况，保持宽高比
 - 支持透明背景的正确处理
"""

import os
import cv2
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Tuple, List, Optional
import yaml
import math

# 物理尺寸配置（单位：mm）
PHYSICAL_SIZES = {
    # 硬币类 - 两种尺寸
    "Coin_H": {"type": "circle", "sizes": [32, 64], "auto_detect": True},
    "Coin_T": {"type": "circle", "sizes": [32, 64], "auto_detect": True},
    
    # 标记类 - 圆形（label 3-7）
    "Markers_Damage010": {"type": "circle", "diameter": 15.8},
    "Markers_Damage050": {"type": "circle", "diameter": 18}, 
    "Markers_Damage100": {"type": "circle", "diameter": 22.6},
    "Markers_Poisoned": {"type": "circle", "diameter": 27.2},
    "Markers_Burned": {"type": "circle", "diameter": 27.2},
    
    # 标记类 - 长方形（label 8-9）
    "Markers_GX_Avaiable": {"type": "rectangle", "width": 41, "height": 64.5},
    "Markers_GX_Used": {"type": "rectangle", "width": 41, "height": 64.5},
}

# 硬币尺寸自动检测阈值（像素）
COIN_SIZE_THRESHOLDS = {
    "small_max": 75,   # 小硬币最大像素尺寸
    "large_min": 110,  # 大硬币最小像素尺寸
}

# 添加类别特定的放大倍数配置
BOOST_MULTIPLIERS = {
    'Markers_Damage010': 1.18,  # damage010 放大1.05倍
    'Markers_Damage050': 1.12,  # damage050 放大1.05倍
    'Markers_Damage100': 1.1,  # damage100 放大1.05倍
    'Coin_H': 1.05,            # 硬币正面适当放大
    'Coin_T': 1.05,            # 硬币反面适当放大
    'Markers_Poisoned': 1.06,   # 放大
    'Markers_Burned': 1.06,     # 放大
    'Markers_GX_Avaiable': 1.05, # 放大
    'Markers_GX_Used': 1.05,    # 放大
}

def parse_args():
    parser = argparse.ArgumentParser(description='Resize cutouts based on physical dimensions')
    parser.add_argument('--cutout-root', required=True, help='Step1 cutouts directory')
    parser.add_argument('--output-root', required=True, help='Output directory for resized cutouts')
    parser.add_argument('--bg-size', nargs=2, type=int, default=[1088, 1088], help='Background size (width height)')
    parser.add_argument('--pixel-density', type=float, default=1.554, help='Pixels per mm (default: 1.554)')
    parser.add_argument('--config', help='Optional YAML config file to override physical sizes')
    parser.add_argument('--dry-run', action='store_true', help='Only show what would be done')
    parser.add_argument('--verbose', action='store_true', help='Show detailed processing info')
    # 添加放大功能参数
    parser.add_argument('--enable-boost', action='store_true', help='Enable category-specific size boosting')
    return parser.parse_args()

def load_config(config_path: str) -> Dict:
    """加载配置文件覆盖默认物理尺寸"""
    if not config_path or not os.path.exists(config_path):
        return PHYSICAL_SIZES
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 合并配置
    merged = PHYSICAL_SIZES.copy()
    if 'physical_sizes' in config:
        merged.update(config['physical_sizes'])
    
    if 'coin_thresholds' in config:
        COIN_SIZE_THRESHOLDS.update(config['coin_thresholds'])
    
    return merged

def get_object_dimensions(img: np.ndarray) -> Tuple[int, int, Tuple[int, int, int, int]]:
    """获取物体的实际像素尺寸和边界框"""
    if img.shape[2] == 4:  # RGBA
        alpha = img[:, :, 3]
        non_transparent = np.where(alpha > 0)
        if len(non_transparent[0]) == 0:
            return 0, 0, (0, 0, 0, 0)
        
        min_y, max_y = non_transparent[0].min(), non_transparent[0].max()
        min_x, max_x = non_transparent[1].min(), non_transparent[1].max()
        
        width = max_x - min_x + 1
        height = max_y - min_y + 1
        bbox = (min_x, min_y, max_x, max_y)
        
    else:
        # RGB图像，使用轮廓检测
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0, 0, (0, 0, 0, 0)
        
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        width, height = w, h
        bbox = (x, y, x + w - 1, y + h - 1)
    
    return width, height, bbox

def detect_coin_size(cutout_path: str, verbose: bool = False) -> float:
    """自动检测硬币尺寸（大62mm或小30mm）"""
    img = cv2.imread(cutout_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return 30  # 默认小硬币
    
    width, height, bbox = get_object_dimensions(img)
    max_dimension = max(width, height)
    
    if verbose:
        print(f"  Coin detection - dimensions: {width}x{height}, max: {max_dimension}")
    
    # 根据阈值判断大小硬币
    if max_dimension <= COIN_SIZE_THRESHOLDS["small_max"]:
        detected_size = 30
    elif max_dimension >= COIN_SIZE_THRESHOLDS["large_min"]:
        detected_size = 62
    else:
        # 中间区域，根据比例判断
        detected_size = 62 if max_dimension > 95 else 30
    
    if verbose:
        print(f"  Detected coin size: {detected_size}mm")
    
    return detected_size

def calculate_target_size(cutout_path: str, class_name: str, physical_sizes: Dict, 
                         pixel_density: float, verbose: bool = False) -> Tuple[int, int, Dict]:
    """
    计算目标尺寸，返回(width, height, info)
    
    注意：由于我们使用统一的缩放比例，这里只是返回类型信息用于日志记录。
    实际的尺寸计算在 resize_cutout_smart 中完成。
    """
    if class_name not in physical_sizes:
        return 0, 0, {"type": "unknown", "class_name": class_name}
    
    config = physical_sizes[class_name]
    info = {"type": config["type"], "class_name": class_name}
    
    if config["type"] == "circle":
        if "sizes" in config and config.get("auto_detect", False):
            # 硬币自动检测
            diameter_mm = detect_coin_size(cutout_path, verbose)
            info["detected_diameter_mm"] = diameter_mm
        else:
            diameter_mm = config["diameter"]
            info["diameter_mm"] = diameter_mm
        
        # 返回简单的占位值，实际尺寸由统一缩放比例决定
        return 100, 100, info
    
    elif config["type"] == "rectangle":
        width_mm = config["width"]
        height_mm = config["height"]
        
        info.update({
            "width_mm": width_mm,
            "height_mm": height_mm
        })
        
        # 返回简单的占位值，实际尺寸由统一缩放比例决定
        return 100, 100, info
    
    else:
        raise ValueError(f"Unknown object type: {config['type']}")

def resize_cutout_smart(cutout_path: str, target_size: Tuple[int, int], output_path: str, 
                       resize_info: Dict, verbose: bool = False, 
                       enable_boost: bool = False, boost_multipliers: Dict = None) -> bool:
    """
    使用统一的缩放比例智能缩放抠图，保持物理尺寸一致性。
    
    核心思路：
    - 原始图像：1200x1400，像素密度 = 2 px/mm
    - 目标图像：1088x1088，像素密度由pixel_density参数决定
    - 统一缩放比例：pixel_density/2
    - 可选的类别特定放大：scale * boost_multiplier
    
    这样既保持了形状比例不变，也完成了物理尺寸的正确映射。
    无论是圆形还是矩形，都使用相同的缩放逻辑。
    """
    img = cv2.imread(cutout_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Failed to load: {cutout_path}")
        return False

    current_h, current_w = img.shape[:2]
    
    # 计算统一的缩放比例：目标像素密度 / 原始像素密度
    original_pixel_density = 2.0  # 1200x1400 原图中 1mm = 2px
    target_pixel_density = resize_info.get('pixel_density', 1.554)  # 使用传入的像素密度
    scale = target_pixel_density / original_pixel_density
    
    # 应用类别特定的放大倍数
    class_name = resize_info.get('class_name')
    boost_factor = 1.0
    if enable_boost and boost_multipliers and class_name and class_name in boost_multipliers:
        boost_factor = boost_multipliers[class_name]
        scale *= boost_factor
        if verbose:
            print(f"  应用 {class_name} 类别放大倍数: {boost_factor}x")
    
    if verbose:
        print(f"  Original size: {current_w}x{current_h}")
        print(f"  Base scale factor: {target_pixel_density/original_pixel_density:.4f} ({target_pixel_density}/{original_pixel_density})")
        if boost_factor != 1.0:
            print(f"  Boost factor: {boost_factor}x")
        print(f"  Final scale factor: {scale:.4f}")
        print(f"  Target size: {int(current_w * scale)}x{int(current_h * scale)}")
    
    # 应用统一缩放
    new_w = int(round(current_w * scale))
    new_h = int(round(current_h * scale))
    
    resized = cv2.resize(img, (new_w, new_h), 
                        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    
    # 保存结果（直接使用缩放后的图像）
    success = cv2.imwrite(output_path, resized)
    if verbose and success:
        print(f"  Saved to: {output_path}")
    
    return success

def process_class_directory(class_dir: Path, class_name: str, output_root: Path, 
                          physical_sizes: Dict, pixel_density: float, 
                          dry_run: bool = False, verbose: bool = False,
                          enable_boost: bool = False, boost_multipliers: Dict = None) -> Dict:
    """处理单个类别目录"""
    cutout_files = list(class_dir.glob("*.png"))
    if not cutout_files:
        return {"processed": 0, "failed": 0, "details": []}
    
    output_class_dir = output_root / class_name
    if not dry_run:
        output_class_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {"processed": 0, "failed": 0, "details": []}
    
    for cutout_file in tqdm(cutout_files, desc=f"Processing {class_name}"):
        try:
            # 计算目标尺寸
            target_w, target_h, resize_info = calculate_target_size(
                str(cutout_file), class_name, physical_sizes, pixel_density, verbose
            )
            
            output_file = output_class_dir / cutout_file.name
            
            detail = {
                "file": cutout_file.name,
                "target_size": [target_w, target_h],
                "resize_info": resize_info
            }
            
            if dry_run:
                if verbose:
                    print(f"Would resize {cutout_file.name}: {target_w}x{target_h}")
                    print(f"  Info: {resize_info}")
                stats["processed"] += 1
            else:
                # 将pixel_density和class_name添加到resize_info中
                resize_info['pixel_density'] = pixel_density
                resize_info['class_name'] = class_name  # 添加这行
                success = resize_cutout_smart(
                    str(cutout_file), (target_w, target_h), str(output_file), 
                    resize_info, verbose, enable_boost, boost_multipliers  # 添加新参数
                )
                if success:
                    stats["processed"] += 1
                    detail["status"] = "success"
                else:
                    stats["failed"] += 1
                    detail["status"] = "failed"
            
            stats["details"].append(detail)
            
        except Exception as e:
            print(f"Error processing {cutout_file}: {e}")
            stats["failed"] += 1
            stats["details"].append({
                "file": cutout_file.name,
                "status": "error",
                "error": str(e)
            })
    
    return stats

def run_step2_resize(
    cutout_root: str,
    output_root: str,
    config,
    dry_run: bool = False,
    **kwargs
) -> dict:
    """
    Step2: 图像缩放主函数 - 供pipeline调用
    
    Args:
        cutout_root: step1输出的cutouts目录路径
        output_root: 输出目录路径
        config: 配置对象
        dry_run: 是否为预览模式
        **kwargs: 其他参数
        
    Returns:
        dict: 处理结果统计
    """
    from pathlib import Path
    
    cutout_root = Path(cutout_root)
    output_root = Path(output_root)
    
    # 从配置获取参数
    step2_config = config.step2
    physical_sizes = config.config_manager.get_physical_sizes()
    boost_multipliers = config.config_manager.get_boost_multipliers()
    
    pixel_density = step2_config.pixel_density
    enable_boost = step2_config.enable_boost
    verbose = config.global_config.verbose
    
    # 创建输出目录
    if not dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
    
    # 统计信息
    total_stats = {
        "total_processed": 0,
        "total_success": 0,
        "total_failed": 0,
        "classes": {}
    }
    
    if verbose:
        print(f"🔄 开始处理缩放任务")
        print(f"📁 输入目录: {cutout_root}")
        print(f"📁 输出目录: {output_root}")
        print(f"📏 像素密度: {pixel_density} px/mm")
        print(f"🚀 启用放大: {enable_boost}")
        if dry_run:
            print("👀 预览模式 - 不会实际处理文件")
    
    # 遍历所有类别目录
    for class_dir in cutout_root.iterdir():
        if not class_dir.is_dir():
            continue
            
        class_name = class_dir.name
        if verbose:
            print(f"\n🏷️  处理类别: {class_name}")
        
        # 处理该类别
        class_stats = process_class_directory(
            class_dir=class_dir,
            class_name=class_name,
            output_root=output_root,
            physical_sizes=physical_sizes,
            pixel_density=pixel_density,
            dry_run=dry_run,
            verbose=verbose,
            enable_boost=enable_boost,
            boost_multipliers=boost_multipliers
        )
        
        # 更新总统计
        total_stats["total_processed"] += class_stats["processed"]
        total_stats["total_success"] += class_stats["success"]
        total_stats["total_failed"] += class_stats["failed"]
        total_stats["classes"][class_name] = class_stats
        
        if verbose:
            print(f"   ✅ 成功: {class_stats['success']}")
            print(f"   ❌ 失败: {class_stats['failed']}")
    
    if verbose:
        print(f"\n📊 总体统计:")
        print(f"   📝 总处理: {total_stats['total_processed']}")
        print(f"   ✅ 总成功: {total_stats['total_success']}")
        print(f"   ❌ 总失败: {total_stats['total_failed']}")
        if not dry_run:
            print(f"   📁 输出目录: {output_root}")
    
    return total_stats
    
def main():
    args = parse_args()
    
    cutout_root = Path(args.cutout_root)
    output_root = Path(args.output_root)
    bg_width, bg_height = args.bg_size
    pixel_density = args.pixel_density
    
    # 加载配置
    config_data = load_config(args.config)
    physical_sizes = config_data.get('physical_sizes', PHYSICAL_SIZES) if isinstance(config_data, dict) else config_data
    
    # 获取放大倍数配置
    boost_multipliers = BOOST_MULTIPLIERS.copy()
    
    print(f"=== Step2 Resize Configuration (简化版) ===")
    print(f"Background size: {bg_width}x{bg_height}")
    print(f"原始像素密度: 2.0 px/mm (1200x1400)")
    print(f"目标像素密度: {pixel_density} px/mm (1088x1088)")
    print(f"统一缩放比例: {pixel_density/2.0:.4f}")
    print(f"启用类别特定放大: {args.enable_boost}")
    print(f"Dry run: {args.dry_run}")
    print(f"Verbose: {args.verbose}")
    print(f"Physical sizes config: {len(physical_sizes)} classes")
    
    if args.enable_boost:
        print(f"\n放大倍数配置:")
        for class_name, multiplier in boost_multipliers.items():
            print(f"  {class_name}: {multiplier}x")
    
    if args.verbose:
        print(f"\nPhysical sizes:")
        for class_name, config in physical_sizes.items():
            print(f"  {class_name}: {config}")
    
    if not cutout_root.exists():
        print(f"Error: Cutout root directory not found: {cutout_root}")
        return
    
    # 处理每个类别目录
    total_stats = {"processed": 0, "failed": 0, "classes": {}}
    
    for class_dir in cutout_root.iterdir():
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        print(f"\n=== Processing class: {class_name} ===")
        
        class_stats = process_class_directory(
            class_dir, class_name, output_root, 
            physical_sizes, pixel_density, args.dry_run, args.verbose,
            args.enable_boost, boost_multipliers  # 添加新参数
        )
        
        total_stats["processed"] += class_stats["processed"]
        total_stats["failed"] += class_stats["failed"]
        total_stats["classes"][class_name] = class_stats
        
        print(f"  Processed: {class_stats['processed']}, Failed: {class_stats['failed']}")
    
    # 保存处理报告
    if not args.dry_run:
        report_path = output_root / "resize_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump({
                "args": vars(args),
                "physical_sizes": physical_sizes,
                "coin_thresholds": COIN_SIZE_THRESHOLDS,
                "stats": total_stats
            }, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed report saved to: {report_path}")
    
    print(f"\n=== Final Summary ===")
    print(f"Total processed: {total_stats['processed']}")
    print(f"Total failed: {total_stats['failed']}")
    print(f"Classes processed: {len(total_stats['classes'])}")
    
    # 显示每个类别的统计
    for class_name, class_stats in total_stats["classes"].items():
        success_rate = (class_stats["processed"] / (class_stats["processed"] + class_stats["failed"]) * 100) if (class_stats["processed"] + class_stats["failed"]) > 0 else 0
        print(f"  {class_name}: {class_stats['processed']}/{class_stats['processed'] + class_stats['failed']} ({success_rate:.1f}%)")

if __name__ == "__main__":
    main()