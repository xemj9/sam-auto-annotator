"""
SAM Auto Annotator 核心处理模块
包含Step1-4的核心处理逻辑
"""

# 导入各个步骤的处理函数
from . import step1_cutout
from . import step2_resize  
from . import step3_synthesize
from . import step4_coco_convert

# 导出主要函数
__all__ = [
    'step1_cutout',
    'step2_resize', 
    'step3_synthesize',
    'step4_coco_convert'
]