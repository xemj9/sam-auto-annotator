"""
SAM Auto Annotator - 基于SAM的自动标注工具
提供完整的图像分割、抠图、缩放、合成和COCO格式转换流程
"""

from .pipeline import SAMPipeline
from .config import ConfigManager, SAMAutoAnnotatorConfig, load_config
from .core import step1_cutout, step2_resize, step3_synthesize, step4_coco_convert

__version__ = "1.0.0"
__author__ = "SAM Auto Annotator Team"

# 导出主要组件
__all__ = [
    # 主要类
    'SAMPipeline',
    'ConfigManager', 
    'SAMAutoAnnotatorConfig',
    
    # 便捷函数
    'load_config',
    
    # 核心模块
    'step1_cutout',
    'step2_resize', 
    'step3_synthesize',
    'step4_coco_convert',
    
    # 版本信息
    '__version__',
    '__author__'
]