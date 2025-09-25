"""
SAM Auto Annotator 命令行接口
提供便捷的命令行工具来执行各个步骤或完整流程
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .pipeline import SAMPipeline
from .config import ConfigManager


def create_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="SAM Auto Annotator - 基于SAM的自动标注工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 执行完整流程
  sam-annotator full --input-images /path/to/images --background-root /path/to/backgrounds \\
                     --output-root /path/to/output --sam-checkpoint /path/to/sam_model.pth

  # 只执行Step1
  sam-annotator step1 --input-images /path/to/images --output-dir /path/to/output \\
                      --sam-checkpoint /path/to/sam_model.pth

  # 执行Step1-3
  sam-annotator full --input-images /path/to/images --background-root /path/to/backgrounds \\
                     --output-root /path/to/output --sam-checkpoint /path/to/sam_model.pth \\
                     --steps step1,step2,step3
        """
    )
    
    # 全局参数
    parser.add_argument('--config-dir', type=str, help='配置文件目录路径')
    parser.add_argument('--config-file', type=str, default='default.yaml', help='配置文件名')
    parser.add_argument('--verbose', action='store_true', help='详细输出模式')
    
    # 子命令
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 完整流程命令
    full_parser = subparsers.add_parser('full', help='执行完整的处理流程')
    full_parser.add_argument('--input-images', required=True, help='输入图像目录')
    full_parser.add_argument('--background-root', required=True, help='背景图像目录')
    full_parser.add_argument('--output-root', required=True, help='输出根目录')
    full_parser.add_argument('--sam-checkpoint', required=True, help='SAM模型检查点路径')
    full_parser.add_argument('--steps', type=str, help='要执行的步骤，用逗号分隔 (如: step1,step2,step3)')
    
    # Step1命令
    step1_parser = subparsers.add_parser('step1', help='执行Step1: SAM分割抠图')
    step1_parser.add_argument('--input-images', required=True, help='输入图像目录')
    step1_parser.add_argument('--output-dir', required=True, help='输出目录')
    step1_parser.add_argument('--sam-checkpoint', required=True, help='SAM模型检查点路径')
    step1_parser.add_argument('--device', default='cuda', help='设备类型 (cuda/cpu)')
    step1_parser.add_argument('--save-masks-npy', action='store_true', help='保存mask的npy文件')
    
    # Step2命令
    step2_parser = subparsers.add_parser('step2', help='执行Step2: 物理尺寸调整')
    step2_parser.add_argument('--cutout-root', required=True, help='Step1输出的cutouts目录')
    step2_parser.add_argument('--output-root', required=True, help='输出根目录')
    step2_parser.add_argument('--pixel-density', type=float, help='像素密度 (px/mm)')
    step2_parser.add_argument('--dry-run', action='store_true', help='预览模式，不实际处理')
    
    # Step3命令
    step3_parser = subparsers.add_parser('step3', help='执行Step3: 背景合成')
    step3_parser.add_argument('--cutout-root', required=True, help='Step2输出的cutouts目录')
    step3_parser.add_argument('--background-root', required=True, help='背景图像目录')
    step3_parser.add_argument('--synth-root', required=True, help='合成输出目录')
    
    # Step4命令
    step4_parser = subparsers.add_parser('step4', help='执行Step4: COCO格式转换')
    step4_parser.add_argument('--ndjson-path', required=True, help='输入NDJSON文件路径')
    step4_parser.add_argument('--output-path', required=True, help='输出COCO JSON文件路径')
    
    return parser


def main():
    """CLI主入口函数"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        # 初始化pipeline
        config_manager = ConfigManager(args.config_dir)
        config = config_manager.load_config(args.config_file)
        
        # 设置verbose模式
        if args.verbose:
            config.global_config.verbose = True
        
        pipeline = SAMPipeline(config, args.config_dir)
        
        # 执行对应命令
        if args.command == 'full':
            steps = None
            if args.steps:
                steps = [s.strip() for s in args.steps.split(',')]
            
            result = pipeline.run_full_pipeline(
                input_images=args.input_images,
                background_root=args.background_root,
                output_root=args.output_root,
                sam_checkpoint=args.sam_checkpoint,
                steps=steps
            )
            
        elif args.command == 'step1':
            result = pipeline.run_step1(
                input_images=args.input_images,
                output_dir=args.output_dir,
                sam_checkpoint=args.sam_checkpoint,
                device=args.device,
                save_masks_npy=args.save_masks_npy
            )
            
        elif args.command == 'step2':
            result = pipeline.run_step2(
                cutout_root=args.cutout_root,
                output_root=args.output_root,
                pixel_density=args.pixel_density,
                dry_run=args.dry_run
            )
            
        elif args.command == 'step3':
            result = pipeline.run_step3(
                cutout_root=args.cutout_root,
                background_root=args.background_root,
                synth_root=args.synth_root
            )
            
        elif args.command == 'step4':
            result = pipeline.run_step4(
                ndjson_path=args.ndjson_path,
                output_path=args.output_path
            )
        
        # 输出结果
        if result.get('status') == 'success':
            print("✅ 执行成功!")
            if args.verbose and 'result' in result:
                print(f"📊 处理结果: {result['result']}")
            return 0
        else:
            print(f"❌ 执行失败: {result.get('error', '未知错误')}")
            return 1
            
    except Exception as e:
        print(f"❌ 程序异常: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())