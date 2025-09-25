"""
配置管理测试
"""
import pytest
import tempfile
import yaml
from pathlib import Path

from sam_auto_annotator.config import ConfigManager, SAMAutoAnnotatorConfig

def test_config_manager_default():
    """测试默认配置加载"""
    config_manager = ConfigManager()
    config = config_manager.load_config("default.yaml")
    
    assert isinstance(config, SAMAutoAnnotatorConfig)
    assert config.step1.model_type == "vit_h"
    assert config.step3.target_width == 1088

def test_config_manager_custom_dir():
    """测试自定义配置目录"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # 创建测试配置文件
        config_data = {
            'step1': {'model_type': 'vit_b', 'device': 'cpu'},
            'step2': {'pixel_density': 2.0},
            'step3': {'target_width': 512, 'target_height': 512},
            'step4': {'min_area': 5.0},
            'global': {'seed': 123, 'verbose': True}
        }
        
        config_file = Path(temp_dir) / "test.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # 测试加载
        config_manager = ConfigManager(temp_dir)
        config = config_manager.load_config("test.yaml")
        
        assert config.step1.model_type == "vit_b"
        assert config.step2.pixel_density == 2.0
        assert config.step3.target_width == 512