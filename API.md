# API 文档

## 核心类

### SAMPipeline

主要的流程控制类，负责协调各个步骤的执行。

#### 初始化

```python
from sam_auto_annotator import SAMPipeline, ConfigManager

config_manager = ConfigManager()
config = config_manager.load_config("default.yaml")
pipeline = SAMPipeline(config)
```

#### 方法

- `run_full_pipeline()`: 执行完整的四步流程
- `run_step1()`: 执行Step1 SAM分割抠图
- `run_step2()`: 执行Step2 物理尺寸调整
- `run_step3()`: 执行Step3 背景合成
- `run_step4()`: 执行Step4 COCO格式转换

### ConfigManager

配置管理类，负责加载和管理配置文件。

```python
config_manager = ConfigManager("/path/to/configs")
config = config_manager.load_config("default.yaml")
```