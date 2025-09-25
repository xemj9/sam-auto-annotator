"""
配置管理模块
统一管理SAM Auto Annotator的所有配置
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class Step1Config:
    """Step1 SAM分割抠图配置"""
    sam_checkpoint: str = ""
    model_type: str = "vit_h"
    device: str = "cuda"
    category_profiles_file: str = "category_profiles.yaml"
    cutout_dir_name: str = "cutouts"
    metadata_ndjson_name: str = "cutouts_metadata.ndjson"
    enable_content_filter: bool = True
    content_filter_std_dev_threshold: float = 13.0
    nms_iou_threshold: float = 0.30
    nested_containment_threshold: float = 0.80
    enable_padded_sam: bool = True
    pad_px: int = 52


@dataclass
class Step2Config:
    """Step2 物理尺寸调整配置"""
    pixel_density: float = 1.554
    enable_boost: bool = False
    boost_multipliers_file: str = "boost_multipliers.yaml"


@dataclass
class Step3Config:
    """Step3 背景合成配置"""
    target_width: int = 1088
    target_height: int = 1088
    left_trim: int = 296
    right_trim: int = 296
    top_trim: int = 50
    bottom_trim: int = 50
    mixed_classes: bool = False
    bg_repeat: int = 1
    density_config_file: str = "density_config.yaml"


@dataclass
class Step4Config:
    """Step4 COCO转换配置"""
    make_filepaths_absolute: bool = True
    junge_style: bool = True
    min_area: float = 1.0
    start_image_id: int = 1
    start_anno_id: int = 1


@dataclass
class GlobalConfig:
    """全局配置"""
    seed: int = 42
    verbose: bool = False


@dataclass
class SAMAutoAnnotatorConfig:
    """SAM Auto Annotator 主配置类"""
    step1: Step1Config = field(default_factory=Step1Config)
    step2: Step2Config = field(default_factory=Step2Config)
    step3: Step3Config = field(default_factory=Step3Config)
    step4: Step4Config = field(default_factory=Step4Config)
    global_config: GlobalConfig = field(default_factory=GlobalConfig)
    
    # 配置文件路径
    config_dir: Optional[Path] = None
    
    # 缓存的配置数据
    _category_profiles: Optional[Dict] = None
    _physical_sizes: Optional[Dict] = None


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_dir: 配置文件目录路径，如果为None则使用默认路径
        """
        if config_dir is None:
            # 默认配置目录：项目根目录下的configs
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent.parent
            config_dir = project_root / "configs"
        
        self.config_dir = Path(config_dir)
        self.config = SAMAutoAnnotatorConfig()
        self.config.config_dir = self.config_dir
    
    def load_config(self, config_file: str = "default.yaml") -> SAMAutoAnnotatorConfig:
        """
        加载配置文件
        
        Args:
            config_file: 配置文件名
            
        Returns:
            配置对象
        """
        config_path = self.config_dir / config_file
        
        if not config_path.exists():
            print(f"警告: 配置文件不存在 {config_path}，使用默认配置")
            return self.config
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # 更新配置
            if 'step1' in config_data:
                self._update_dataclass(self.config.step1, config_data['step1'])
            if 'step2' in config_data:
                self._update_dataclass(self.config.step2, config_data['step2'])
            if 'step3' in config_data:
                self._update_dataclass(self.config.step3, config_data['step3'])
            if 'step4' in config_data:
                self._update_dataclass(self.config.step4, config_data['step4'])
            if 'global' in config_data:
                self._update_dataclass(self.config.global_config, config_data['global'])
                
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            print("使用默认配置")
        
        return self.config
    
    def _update_dataclass(self, obj, data: Dict[str, Any]):
        """更新dataclass对象的属性"""
        for key, value in data.items():
            if hasattr(obj, key):
                setattr(obj, key, value)
    
    def load_category_profiles(self) -> Dict[str, Any]:
        """加载类别配置文件"""
        if self.config._category_profiles is not None:
            return self.config._category_profiles
        
        profiles_file = self.config_dir / self.config.step1.category_profiles_file
        if not profiles_file.exists():
            profiles_file = self.config_dir / "category_profiles.yaml"
        
        try:
            with open(profiles_file, 'r', encoding='utf-8') as f:
                self.config._category_profiles = yaml.safe_load(f)
        except Exception as e:
            print(f"加载类别配置文件失败: {e}")
            self.config._category_profiles = {}
        
        return self.config._category_profiles
    
    def load_physical_sizes(self) -> Dict[str, Any]:
        """加载物理尺寸配置文件"""
        if self.config._physical_sizes is not None:
            return self.config._physical_sizes
        
        sizes_file = self.config_dir / "physical_sizes.yaml"
        
        try:
            with open(sizes_file, 'r', encoding='utf-8') as f:
                self.config._physical_sizes = yaml.safe_load(f)
        except Exception as e:
            print(f"加载物理尺寸配置文件失败: {e}")
            self.config._physical_sizes = {}
        
        return self.config._physical_sizes
    
    def get_category_mapping(self) -> Dict[str, int]:
        """获取类别ID映射"""
        profiles = self.load_category_profiles()
        return profiles.get('category_mapping', {})
    
    def get_category_profiles(self) -> Dict[str, Any]:
        """获取类别检测配置"""
        profiles = self.load_category_profiles()
        return profiles.get('category_profiles', {})
    
    def get_category_to_profile_mapping(self) -> Dict[str, str]:
        """获取类别到配置文件的映射"""
        profiles = self.load_category_profiles()
        return profiles.get('category_to_profile_mapping', {})
    
    def get_physical_sizes(self) -> Dict[str, Any]:
        """获取物理尺寸配置"""
        sizes = self.load_physical_sizes()
        return sizes.get('physical_sizes', {})
    
    def get_boost_multipliers(self) -> Dict[str, float]:
        """获取类别特定放大倍数"""
        sizes = self.load_physical_sizes()
        return sizes.get('boost_multipliers', {})


# 全局配置管理器实例
_config_manager = None

def get_config_manager(config_dir: Optional[str] = None) -> ConfigManager:
    """获取全局配置管理器实例"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_dir)
    return _config_manager

def load_config(config_file: str = "default.yaml", config_dir: Optional[str] = None) -> SAMAutoAnnotatorConfig:
    """便捷函数：加载配置"""
    manager = get_config_manager(config_dir)
    return manager.load_config(config_file)
