"""
SAM Auto Annotator 主流程控制器
协调Step1-4的执行流程
"""
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

from .config import ConfigManager, SAMAutoAnnotatorConfig
from .core import step1_cutout, step2_resize, step3_synthesize, step4_coco_convert


class SAMPipeline:
    """SAM Auto Annotator 主流程控制器"""
    
    def __init__(self, config: Optional[SAMAutoAnnotatorConfig] = None, config_dir: Optional[str] = None):
        """
        初始化流程控制器
        
        Args:
            config: 配置对象，如果为None则加载默认配置
            config_dir: 配置文件目录
        """
        if config is None:
            config_manager = ConfigManager(config_dir)
            config = config_manager.load_config()
        
        self.config = config
        self.config_manager = ConfigManager(config_dir) if config_dir else ConfigManager()
        
        # 设置日志
        self._setup_logging()
        
        self.logger = logging.getLogger(__name__)
    
    def _setup_logging(self):
        """设置日志"""
        level = logging.DEBUG if self.config.global_config.verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def run_step1(self, 
                  input_images: str,
                  output_dir: str,
                  sam_checkpoint: Optional[str] = None,
                  **kwargs) -> Dict[str, Any]:
        """
        执行Step1: SAM分割抠图
        
        Args:
            input_images: 输入图像目录或文件路径
            output_dir: 输出目录
            sam_checkpoint: SAM模型检查点路径
            **kwargs: 其他参数
            
        Returns:
            执行结果信息
        """
        self.logger.info("开始执行Step1: SAM分割抠图")
        
        # 更新配置
        if sam_checkpoint:
            self.config.step1.sam_checkpoint = sam_checkpoint
        
        try:
            # 调用step1_cutout模块的run函数
            result = step1_cutout.run_step1_cutout(
                input_images=input_images,
                output_dir=output_dir,
                sam_checkpoint=sam_checkpoint or self.config.step1.sam_checkpoint,
                config=self.config,
                **kwargs
            )
            
            self.logger.info("Step1执行完成")
            return {"status": "success", "result": result}
            
        except Exception as e:
            self.logger.error(f"Step1执行失败: {e}")
            return {"status": "error", "error": str(e)}

    def run_step2(self,
                  cutout_root: str,
                  output_root: str,
                  pixel_density: Optional[float] = None,
                  **kwargs) -> Dict[str, Any]:
        """
        执行Step2: 物理尺寸调整
        
        Args:
            cutout_root: 抠图根目录
            output_root: 输出根目录
            pixel_density: 像素密度
            **kwargs: 其他参数
            
        Returns:
            执行结果信息
        """
        self.logger.info("开始执行Step2: 物理尺寸调整")
        
        # 更新配置
        if pixel_density:
            self.config.step2.pixel_density = pixel_density
        
        try:
            result = step2_resize.run_step2_resize(
                cutout_root=cutout_root,
                output_root=output_root,
                config=self.config,
                **kwargs
            )
            
            self.logger.info("Step2执行完成")
            return {"status": "success", "result": result}
            
        except Exception as e:
            self.logger.error(f"Step2执行失败: {e}")
            return {"status": "error", "error": str(e)}

    def run_step3(self,
                  cutout_root: str,
                  background_root: str,
                  synth_root: str,
                  **kwargs) -> Dict[str, Any]:
        """
        执行Step3: 背景合成
        
        Args:
            cutout_root: 抠图根目录
            background_root: 背景图根目录
            synth_root: 合成输出根目录
            **kwargs: 其他参数
            
        Returns:
            执行结果信息
        """
        self.logger.info("开始执行Step3: 背景合成")
        
        try:
            result = step3_synthesize.run_step3_synthesize(
                cutout_root=cutout_root,
                background_root=background_root,
                output_root=synth_root,
                config=self.config,
                **kwargs
            )
            
            self.logger.info("Step3执行完成")
            return {"status": "success", "result": result}
            
        except Exception as e:
            self.logger.error(f"Step3执行失败: {e}")
            return {"status": "error", "error": str(e)}

    def run_step4(self,
                  ndjson_path: str,
                  output_path: str,
                  **kwargs) -> Dict[str, Any]:
        """
        执行Step4: COCO格式转换
        
        Args:
            ndjson_path: 输入NDJSON文件路径
            output_path: 输出COCO JSON文件路径
            **kwargs: 其他参数
            
        Returns:
            执行结果信息
        """
        self.logger.info("开始执行Step4: COCO格式转换")
        
        try:
            result = step4_coco_convert.run_step4_coco_convert(
                ndjson_path=ndjson_path,
                output_path=output_path,
                config=self.config,
                **kwargs
            )
            
            self.logger.info("Step4执行完成")
            return {"status": "success", "result": result}
            
        except Exception as e:
            self.logger.error(f"Step4执行失败: {e}")
            return {"status": "error", "error": str(e)}
    
    def run_full_pipeline(self,
                         input_images: str,
                         background_root: str,
                         output_root: str,
                         sam_checkpoint: str,
                         steps: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        执行完整的流程
        
        Args:
            input_images: 输入图像目录
            background_root: 背景图根目录
            output_root: 输出根目录
            sam_checkpoint: SAM模型检查点路径
            steps: 要执行的步骤列表，如['step1', 'step2']，None表示执行所有步骤
            
        Returns:
            执行结果信息
        """
        if steps is None:
            steps = ['step1', 'step2', 'step3', 'step4']
        
        results = {}
        
        # Step1: SAM分割抠图
        if 'step1' in steps:
            step1_output = Path(output_root) / "step1_cutouts"
            result = self.run_step1(
                input_images=input_images,
                output_dir=str(step1_output),
                sam_checkpoint=sam_checkpoint
            )
            results['step1'] = result
            if result['status'] != 'success':
                return results
        
        # Step2: 物理尺寸调整
        if 'step2' in steps:
            step1_output = Path(output_root) / "step1_cutouts" / "cutouts"
            step2_output = Path(output_root) / "step2_resized"
            result = self.run_step2(
                cutout_root=str(step1_output),
                output_root=str(step2_output)
            )
            results['step2'] = result
            if result['status'] != 'success':
                return results
        
        # Step3: 背景合成
        if 'step3' in steps:
            step2_output = Path(output_root) / "step2_resized"
            step3_output = Path(output_root) / "step3_synthesized"
            result = self.run_step3(
                cutout_root=str(step2_output),
                background_root=background_root,
                synth_root=str(step3_output)
            )
            results['step3'] = result
            if result['status'] != 'success':
                return results
        
        # Step4: COCO格式转换
        if 'step4' in steps:
            step3_output = Path(output_root) / "step3_synthesized"
            ndjson_file = step3_output / "placements_cropped1088_margins.ndjson"
            coco_output = step3_output / "annotations_coco.json"
            result = self.run_step4(
                ndjson_path=str(ndjson_file),
                output_path=str(coco_output)
            )
            results['step4'] = result
        
        return results
