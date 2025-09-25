#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step3_ndjson_to_coco_strict.py

将 Step2 placements NDJSON -> COCO bbox-only JSON（严格对齐 NDJSON）
特性：
 - 优先使用 NDJSON 中的 category_id 和 category_name（不随意重映射）
 - 可选输出军哥风格：categories 的 supercategory/name = id （--junge-style）
 - 可选扁平化图片文件名为连续编号（--flatten-images）
 - bbox clip & min-area filter

 python /mnt/afs/xiemingjin/sam_project/tool/step4_ndjson_to_coco.py \
  --ndjson /mnt/afs/xiemingjin/sam_project/saved/training_data/20250919_sam_step2_synth_crop1088_margins_0.1_v2/placements_cropped1088_margins.ndjson \
  --out /mnt/afs/xiemingjin/sam_project/saved/training_data/20250919_sam_step2_synth_crop1088_margins_0.1_v2/annotations_coco_junge_style.json \
  --make-filepaths-absolute \
  --junge-style
"""
import argparse
import json
import os
import sys
from pathlib import Path
import cv2

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ndjson", required=True, help="Path to placements NDJSON (one JSON per line)")
    p.add_argument("--out", required=True, help="Path to output COCO JSON")
    p.add_argument("--make-filepaths-absolute", action="store_true",
                   help="Make images.file_name absolute (default: keep as-is from NDJSON)")
    p.add_argument("--image-root", default=None,
                   help="If provided and not using make-filepaths-absolute, make file_name relative to this root")
    p.add_argument("--min-area", type=float, default=1.0, help="Minimum bbox area to keep")
    p.add_argument("--start-image-id", type=int, default=1)
    p.add_argument("--start-anno-id", type=int, default=1)
    p.add_argument("--junge-style", action="store_true",
                   help="Output categories in Junge style: supercategory=id and name=id (numeric).")
    p.add_argument("--flatten-images", action="store_true",
                   help="Flatten image filenames to sequential names under --flatten-image-root (e.g. /.../images/0000001.jpg)")
    p.add_argument("--flatten-image-root", default=None,
                   help="Root dir to use when --flatten-images is set (required then).")
    p.add_argument("--digits", type=int, default=7, help="Zero-pad digits when flattening filenames (default 7 -> 0000001)")
    p.add_argument("--require-cat-id", action="store_true",
                   help="If set, skip objects that lack category_id (fail-fast style). Otherwise assign new ids if missing.")
    return p.parse_args()

def read_ndjson(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                records.append(json.loads(s))
            except Exception as e:
                print(f"[WARN] failed to parse line {i} in {path}: {e}", file=sys.stderr)
    return records

def clip_bbox(bbox, img_w, img_h):
    # bbox: [x,y,w,h]
    x, y, w, h = [float(v) for v in bbox]
    if w <= 0 or h <= 0:
        return None, 0.0
    x1 = max(0.0, x)
    y1 = max(0.0, y)
    x2 = min(float(img_w), x + w)
    y2 = min(float(img_h), y + h)
    nw = x2 - x1
    nh = y2 - y1
    if nw <= 0 or nh <= 0:
        return None, 0.0
    return [x1, y1, nw, nh], float(nw * nh)

def try_infer_size(img_path):
    try:
        im = cv2.imread(img_path)
        if im is not None:
            h, w = im.shape[:2]
            return int(w), int(h)
    except Exception:
        pass
    return None, None

def run_step4_coco_convert(
    ndjson_path: str,
    output_path: str,
    config,
    **kwargs
) -> dict:
    """
    Step4: NDJSON转COCO格式主函数 - 供pipeline调用
    
    Args:
        ndjson_path: step3输出的placements NDJSON文件路径
        output_path: 输出COCO JSON文件路径
        config: 配置对象
        **kwargs: 其他参数
        
    Returns:
        dict: 处理结果统计
    """
    from pathlib import Path
    
    ndjson_path = Path(ndjson_path)
    if not ndjson_path.exists():
        raise FileNotFoundError(f"NDJSON文件不存在: {ndjson_path}")
    
    # 从配置获取参数
    step4_config = config.step4
    verbose = config.global_config.verbose
    
    # 读取NDJSON记录
    records = read_ndjson(str(ndjson_path))
    if not records:
        raise ValueError("NDJSON文件中没有记录")
    
    if verbose:
        print(f"🔄 开始处理COCO格式转换")
        print(f"📁 输入NDJSON: {ndjson_path}")
        print(f"📁 输出COCO: {output_path}")
        print(f"📝 总记录数: {len(records)}")
    
    # 统计信息
    stats = {
        "total_records": len(records),
        "total_images": 0,
        "total_annotations": 0,
        "total_categories": 0,
        "skipped_records": 0,
        "filtered_annotations": 0
    }
    
    images = []
    annotations = []
    categories = {}      # category_id (int) -> category_name (str or None)
    image_path_to_id = {}
    ann_id = step4_config.start_anno_id
    img_id_counter = step4_config.start_image_id
    flatten_counter = 1
    
    for rec_idx, rec in enumerate(records, 1):
        raw_img_path = rec.get("synth_image") or rec.get("image_path") or rec.get("file_name")
        if raw_img_path is None:
            if verbose:
                print(f"⚠️  记录 #{rec_idx} 缺少图像路径，跳过")
            stats["skipped_records"] += 1
            continue
        raw_img_path = str(raw_img_path)
        
        # 计算最终的file_name
        if step4_config.make_filepaths_absolute:
            file_name = os.path.abspath(raw_img_path)
        elif step4_config.flatten_images:
            # 构建扁平化路径
            img_dir = os.path.join(step4_config.flatten_image_root, "images")
            os.makedirs(img_dir, exist_ok=True)
            padded = str(flatten_counter).zfill(step4_config.digits)
            file_name = os.path.join(img_dir, padded + ".jpg")
            flatten_counter += 1
        elif step4_config.image_root:
            try:
                file_name = os.path.relpath(raw_img_path, step4_config.image_root)
            except ValueError:
                file_name = raw_img_path
        else:
            file_name = raw_img_path
        
        # 处理图像记录
        if file_name not in image_path_to_id:
            # 尝试获取图像尺寸
            img_w, img_h = try_infer_size(raw_img_path)
            if img_w is None or img_h is None:
                if verbose:
                    print(f"⚠️  无法获取图像尺寸: {raw_img_path}，跳过")
                stats["skipped_records"] += 1
                continue
            
            # 如果启用扁平化，复制图像文件
            if step4_config.flatten_images:
                import shutil
                try:
                    shutil.copy2(raw_img_path, file_name)
                except Exception as e:
                    if verbose:
                        print(f"⚠️  复制图像失败: {e}")
                    stats["skipped_records"] += 1
                    continue
            
            # 添加图像记录
            image_path_to_id[file_name] = img_id_counter
            images.append({
                "id": img_id_counter,
                "file_name": file_name,
                "width": img_w,
                "height": img_h
            })
            img_id_counter += 1
        
        # 处理标注
        bbox = rec.get("bbox")
        if bbox is None:
            if verbose:
                print(f"⚠️  记录 #{rec_idx} 缺少bbox，跳过")
            stats["skipped_records"] += 1
            continue
        
        # 获取类别信息
        category_id = rec.get("class_id") or rec.get("category_id")
        category_name = rec.get("class_name") or rec.get("category_name")
        
        if category_id is None:
            if step4_config.require_cat_id:
                if verbose:
                    print(f"⚠️  记录 #{rec_idx} 缺少category_id，跳过")
                stats["skipped_records"] += 1
                continue
            else:
                # 分配新的category_id
                category_id = len(categories) + 1
        
        category_id = int(category_id)
        
        # 记录类别
        if category_id not in categories:
            categories[category_id] = category_name
        
        # 处理bbox
        if len(bbox) == 4:
            x, y, w_or_x2, h_or_y2 = bbox
            # 判断是xywh还是xyxy格式
            if w_or_x2 > x and h_or_y2 > y and w_or_x2 < x + 2000 and h_or_y2 < y + 2000:
                # 可能是xyxy格式
                if w_or_x2 > 100 and h_or_y2 > 100:  # 简单启发式判断
                    w, h = w_or_x2 - x, h_or_y2 - y
                else:
                    w, h = w_or_x2, h_or_y2
            else:
                w, h = w_or_x2, h_or_y2
        else:
            if verbose:
                print(f"⚠️  记录 #{rec_idx} bbox格式错误，跳过")
            stats["skipped_records"] += 1
            continue
        
        # 裁剪bbox到图像边界
        img_w, img_h = try_infer_size(raw_img_path)
        if img_w and img_h:
            x, y, w, h = clip_bbox([x, y, w, h], img_w, img_h)
        
        # 过滤小面积bbox
        area = w * h
        if area < step4_config.min_area:
            stats["filtered_annotations"] += 1
            continue
        
        # 添加标注
        annotations.append({
            "id": ann_id,
            "image_id": image_path_to_id[file_name],
            "category_id": category_id,
            "bbox": [x, y, w, h],
            "area": area,
            "iscrowd": 0
        })
        ann_id += 1
    
    # 构建categories列表
    categories_list = []
    for cat_id in sorted(categories.keys()):
        cat_name = categories[cat_id]
        if step4_config.junge_style:
            # 军哥风格：supercategory和name都设为id
            categories_list.append({
                "id": cat_id,
                "name": str(cat_id),
                "supercategory": str(cat_id)
            })
        else:
            # 标准风格
            categories_list.append({
                "id": cat_id,
                "name": cat_name or f"class_{cat_id}",
                "supercategory": "object"
            })
    
    # 构建COCO格式
    coco_data = {
        "info": {
            "description": "Generated by SAM Auto Annotator",
            "version": "1.0",
            "year": 2024,
            "contributor": "SAM Auto Annotator",
            "date_created": "2024-01-01"
        },
        "licenses": [
            {
                "id": 1,
                "name": "Unknown",
                "url": ""
            }
        ],
        "images": images,
        "annotations": annotations,
        "categories": categories_list
    }
    
    # 保存COCO JSON
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=2, ensure_ascii=False)
    
    # 更新统计信息
    stats["total_images"] = len(images)
    stats["total_annotations"] = len(annotations)
    stats["total_categories"] = len(categories_list)
    
    if verbose:
        print(f"\n📊 COCO转换完成统计:")
        print(f"   📝 总记录数: {stats['total_records']}")
        print(f"   🖼️  图像数量: {stats['total_images']}")
        print(f"   📍 标注数量: {stats['total_annotations']}")
        print(f"   🏷️  类别数量: {stats['total_categories']}")
        print(f"   ⚠️  跳过记录: {stats['skipped_records']}")
        print(f"   🔍 过滤标注: {stats['filtered_annotations']}")
        print(f"   📁 输出文件: {output_path}")
    
    return stats

def main():
    args = parse_args()

    ndjson_path = Path(args.ndjson)
    if not ndjson_path.exists():
        print(f"[ERROR] NDJSON not found: {ndjson_path}", file=sys.stderr)
        return

    records = read_ndjson(str(ndjson_path))
    if not records:
        print(f"[ERROR] no records read from NDJSON", file=sys.stderr)
        return

    if args.flatten_images and not args.flatten_image_root:
        print("[ERROR] --flatten-images requires --flatten-image-root to be set", file=sys.stderr)
        return

    images = []
    annotations = []
    categories = {}      # category_id (int) -> category_name (str or None)
    image_path_to_id = {}
    ann_id = int(args.start_anno_id)
    img_id_counter = int(args.start_image_id)
    flatten_counter = 1

    for rec_idx, rec in enumerate(records, 1):
        raw_img_path = rec.get("image_path") or rec.get("file_name")
        if raw_img_path is None:
            print(f"[WARN] record #{rec_idx} missing image_path/file_name, skipping", file=sys.stderr)
            continue
        raw_img_path = str(raw_img_path)

        # compute final file_name to record in COCO
        if args.make_filepaths_absolute:
            file_name = os.path.abspath(raw_img_path)
        elif args.flatten_images:
            # construct flattened path: <flatten_image_root>/images/0000001.jpg
            img_dir = os.path.join(args.flatten_image_root, "images")
            os.makedirs(img_dir, exist_ok=True)
            padded = str(flatten_counter).zfill(args.digits)
            file_name = os.path.join(img_dir, padded + ".jpg")
            flatten_counter += 1
        elif args.image_root:
            try:
                file_name = os.path.relpath(raw_img_path, args.image_root)
            except Exception:
                file_name = raw_img_path
        else:
            file_name = raw_img_path

        # assign image id (preserve insertion order)
        if file_name in image_path_to_id:
            img_id = image_path_to_id[file_name]
        else:
            # determine width/height - prefer rec values else try infer
            width = rec.get("width")
            height = rec.get("height")
            if width is None or height is None:
                w, h = try_infer_size(raw_img_path)
                if w and h:
                    width, height = int(w), int(h)
                else:
                    width = int(width or 0)
                    height = int(height or 0)
            try:
                width = int(width)
                height = int(height)
            except Exception:
                width = int(width or 0)
                height = int(height or 0)
            if width <= 0 or height <= 0:
                print(f"[WARN] cannot determine image size for {raw_img_path}, skip record", file=sys.stderr)
                continue

            img_id = img_id_counter
            img_id_counter += 1
            image_path_to_id[file_name] = img_id
            images.append({
                "height": int(height),
                "width": int(width),
                "id": int(img_id),
                "file_name": file_name
            })

        # objects -> annotations
        objs = rec.get("objects", [])
        for obj_idx, obj in enumerate(objs, 1):
            bbox = obj.get("bbox") or obj.get("bbox_xywh")
            if not bbox:
                print(f"[WARN] no bbox for object #{obj_idx} in image {file_name}, skipping object", file=sys.stderr)
                continue

            # Prefer object's category_id, fallback to record-level category_id
            raw_cat_id = obj.get("category_id", rec.get("category_id"))
            raw_cat_name = obj.get("category_name") or rec.get("category_single")
            if raw_cat_id is None:
                if raw_cat_name is None:
                    if args.require_cat_id:
                        print(f"[WARN] object missing category info in {file_name}, skipping (require-cat-id=True)", file=sys.stderr)
                        continue
                    # fallback: assign new id (warn)
                    # choose next available integer id
                    if categories:
                        next_id = max(categories.keys()) + 1
                    else:
                        next_id = 1
                    print(f"[WARN] object in {file_name} missing category_id but has name '{raw_cat_name}'. Assigning id {next_id}", file=sys.stderr)
                    raw_cat_id = next_id
                else:
                    # if name exists but no id: either map name->id if present, else assign new
                    # try find existing id for this name
                    found = None
                    for k, v in categories.items():
                        if v == raw_cat_name:
                            found = k; break
                    if found:
                        raw_cat_id = found
                    else:
                        if categories:
                            raw_cat_id = max(categories.keys()) + 1
                        else:
                            raw_cat_id = 1
                        print(f"[WARN] assigning new id {raw_cat_id} for unseen category name '{raw_cat_name}'", file=sys.stderr)

            # convert cat id to int if possible
            try:
                cat_id = int(raw_cat_id)
            except Exception:
                # if cannot convert, assign new int
                if categories:
                    cat_id = max(categories.keys()) + 1
                else:
                    cat_id = 1
                print(f"[WARN] category_id '{raw_cat_id}' is not int; assigned {cat_id}", file=sys.stderr)

            # record category name if provided (prefer provided name)
            if raw_cat_name:
                categories[cat_id] = raw_cat_name
            else:
                # keep existing name if any, else set to string of id
                categories.setdefault(cat_id, str(cat_id))

            # clip bbox
            img_info = next((im for im in images if im["id"] == img_id), None)
            if img_info is None:
                print(f"[WARN] cannot find image info for id {img_id}", file=sys.stderr)
                continue
            img_w = img_info["width"]; img_h = img_info["height"]
            fixed_bbox, area = clip_bbox(bbox, img_w, img_h)
            if fixed_bbox is None:
                continue
            if area < float(args.min_area):
                continue

            annotations.append({
                "id": int(ann_id),
                "image_id": int(img_id),
                "category_id": int(cat_id),
                "bbox": [float(v) for v in fixed_bbox],
                "area": float(area),
                "iscrowd": 0,
                "segmentation": []
            })
            ann_id += 1

    # Ensure categories cover all category_ids appearing in annotations
    ann_cat_ids = set([int(a["category_id"]) for a in annotations])
    for cid in ann_cat_ids:
        if cid not in categories:
            # fill missing name by numeric placeholder
            categories[cid] = str(cid)
            print(f"[WARN] annotation uses category_id {cid} but no name found in NDJSON; filled name='{categories[cid]}'", file=sys.stderr)

    # Build categories list
    cats_list = []
    for cid in sorted(categories.keys()):
        if args.junge_style:
            # Junge style: supercategory = id (numeric), name = id (numeric)
            cats_list.append({"supercategory": int(cid), "id": int(cid), "name": int(cid)})
        else:
            cats_list.append({"supercategory": "none", "id": int(cid), "name": categories[cid]})

    coco = {
        "info": {"description": "SYNTH Dataset converted from placements NDJSON"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": cats_list
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2, ensure_ascii=False)

    print(f"[OK] wrote COCO JSON to {out_path}")
    print(f"Images: {len(images)}, Annotations: {len(annotations)}, Categories: {len(cats_list)}")

if __name__ == "__main__":
    main()


