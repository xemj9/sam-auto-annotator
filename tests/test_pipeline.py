"""
Pipeline测试
"""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from sam_auto_annotator import SAMPipeline, ConfigManager

def test_pipeline_initialization():
    """测试Pipeline初始化"""
    config_manager = ConfigManager()
    config = config_manager.load_config("default.yaml")
    pipeline = SAMPipeline(config)
    
    assert pipeline.config is not None
    assert pipeline.config_manager is not None

@patch('sam_auto_annotator.core.step1_cutout.run_step1_cutout')
def test_run_step1(mock_step1):
    """测试Step1执行"""
    mock_step1.return_value = {"status": "success"}
    
    config_manager = ConfigManager()
    config = config_manager.load_config("default.yaml")
    pipeline = SAMPipeline(config)
    
    result = pipeline.run_step1(
        input_images="/fake/path",
        output_dir="/fake/output",
        sam_checkpoint="/fake/model.pth"
    )
    
    assert result["status"] == "success"
    mock_step1.assert_called_once()