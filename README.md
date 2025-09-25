# 🎯 SAM Auto Annotator

[![CI](https://github.com/xiemj/sam-auto-annotator/workflows/CI/badge.svg)](https://github.com/xiemj/sam-auto-annotator/actions)
[![GitHub release](https://img.shields.io/github/release/xiemj/sam-auto-annotator.svg)](https://github.com/xiemj/sam-auto-annotator/releases)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![GitHub stars](https://img.shields.io/github/stars/xiemj/sam-auto-annotator.svg)](https://github.com/xiemj/sam-auto-annotator/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/xiemj/sam-auto-annotator.svg)](https://github.com/xiemj/sam-auto-annotator/network)
[![GitHub issues](https://img.shields.io/github/issues/xiemj/sam-auto-annotator.svg)](https://github.com/xiemj/sam-auto-annotator/issues)

> **🚀 流水线计算机视觉数据集生成工具**  
> 基于Meta SAM模型的全自动标注流水线，让数据集制作从几周缩短到几小时！

## 💡 为什么选择 SAM Auto Annotator？

### 🎯 解决核心痛点
- **告别手工标注**：传统数据集标注需要数周甚至数月，我们将其缩短到几小时
- **零基础上手**：无需深度学习背景，一键运行完整流水线
- **工业级质量**：基于Meta最新SAM模型，分割精度媲美人工标注
- **成本大幅降低**：减少90%以上的人工标注成本

### 🌟 技术优势
- **🔥 最新SAM技术**：集成Meta Segment Anything Model，零样本分割能力
- **⚡ 全流程自动化**：从原始图片到COCO数据集，一站式解决方案
- **🎨 智能数据增强**：自动背景合成，大幅扩充数据集规模
- **🔧 高度可配置**：灵活的YAML配置，适应各种业务场景

## 🎯 应用场景

### 🏭 工业检测
- **质量控制**：产品缺陷检测数据集生成
- **安全监控**：工业设备异常检测
- **自动化生产**：零件识别与分类

### 🚗 自动驾驶
- **目标检测**：车辆、行人、交通标志识别
- **语义分割**：道路、车道线分割
- **场景理解**：复杂交通场景解析

### 🏥 医疗影像
- **病灶检测**：医学图像中的异常区域标注
- **器官分割**：CT、MRI图像的器官轮廓提取
- **辅助诊断**：影像学特征自动标注

### 🛒 电商零售
- **商品识别**：商品图像自动分割与标注
- **库存管理**：商品计数与分类
- **推荐系统**：基于视觉的商品推荐

### 🎮 内容创作
- **游戏开发**：游戏素材自动分割
- **影视制作**：视频内容自动标注
- **AR/VR应用**：虚拟场景物体识别

## 🚀 核心优势对比

| 特性 | 传统手工标注 | 其他自动化工具 | SAM Auto Annotator |
|------|-------------|---------------|-------------------|
| **标注速度** | 慢（天/周） | 中等（小时） | **极快（分钟）** |
| **标注精度** | 高 | 中等 | **高（SAM加持）** |
| **学习成本** | 高 | 中等 | **低（一键运行）** |
| **扩展性** | 差 | 中等 | **强（模块化设计）** |
| **成本** | 高 | 中等 | **低（开源免费）** |

## 🌟 功能特性

- **Step1**: SAM自动分割和抠图
- **Step2**: 基于物理尺寸的图像缩放调整  
- **Step3**: 背景合成和数据增强
- **Step4**: COCO格式数据集转换

## 📦 安装

### 环境要求

- Python 3.8+
- CUDA 11.0+ (可选，用于GPU加速)

### 🎯 快速体验（推荐）

**不想下载整个项目？** 只需要核心功能？你可以直接使用我们的核心代码！

```bash
# 只需要复制这4个核心文件到你的项目中
# /mnt/afs/xiemingjin/sam_auto_annotator/src/sam_auto_annotator/core/
# ├── step1_cutout.py      # SAM自动分割和抠图
# ├── step2_resize.py      # 基于物理尺寸的图像缩放
# ├── step3_synthesize.py  # 背景合成和数据增强  
# └── step4_coco_convert.py # COCO格式数据集转换

# 然后安装必要依赖
pip install torch torchvision segment-anything opencv-python pillow pyyaml
```

**极简使用示例：**
```python
# 直接导入核心函数使用
from step1_cutout import run_step1_cutout
from step2_resize import run_step2_resize  
from step3_synthesize import run_step3_synthesize
from step4_coco_convert import run_step4_coco_convert

# 执行完整流水线
result1 = run_step1_cutout(input_images="./images", output_dir="./step1_output")
result2 = run_step2_resize(cutout_root="./step1_output", output_root="./step2_output")
result3 = run_step3_synthesize(cutout_root="./step2_output", background_root="./backgrounds", synth_root="./step3_output")
result4 = run_step4_coco_convert(ndjson_path="./step3_output/annotations.ndjson", output_path="./final_dataset.json")
```

> 💡 **提示**：这种方式适合想要快速集成到现有项目中的开发者，只需要4个核心文件就能完成整个标注流水线！

### 完整安装（推荐用于生产环境）

```bash
git clone https://github.com/xemj9/sam-auto-annotator.git
cd sam-auto-annotator
pip install -e .
```
## 📦 安装

### 环境要求

- Python 3.8+
- CUDA 11.0+ (可选，用于GPU加速)

### 从源码安装

```bash
git clone https://github.com/xemj9/sam-auto-annotator.git
cd sam-auto-annotator
pip install -e .
```

### 安装依赖

```bash
# 基础依赖
pip install -r requirements.txt

# 开发依赖（可选）
pip install -r requirements-dev.txt
```

### SAM模型下载

```bash
# 下载SAM模型文件（约2.4GB）
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
# 或者使用curl
curl -O https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## 🚀 快速开始

### 1. 准备数据

```bash
# 创建工作目录
mkdir -p demo/{input,backgrounds,output}

# 准备输入图片和背景图片
cp your_images/* demo/input/
cp background_images/* demo/backgrounds/
```

### 2. 命令行使用

```bash
# 执行完整流程
sam-annotator full \
    --input-images demo/input \
    --background-root demo/backgrounds \
    --output-root demo/output \
    --sam-checkpoint sam_vit_h_4b8939.pth \
    --verbose

# 只执行特定步骤
sam-annotator step1 \
    --input-images demo/input \
    --output-dir demo/output/step1 \
    --sam-checkpoint sam_vit_h_4b8939.pth \
    --device cuda

# 执行指定的步骤组合
sam-annotator full \
    --input-images demo/input \
    --background-root demo/backgrounds \
    --output-root demo/output \
    --sam-checkpoint sam_vit_h_4b8939.pth \
    --steps step1,step2,step3
```

### 3. Python模块使用

```bash
# 使用python -m方式运行
python -m sam_auto_annotator full \
    --input-images demo/input \
    --background-root demo/backgrounds \
    --output-root demo/output \
    --sam-checkpoint sam_vit_h_4b8939.pth
```

### 4. Python API使用

```python
from sam_auto_annotator import SAMPipeline, ConfigManager, load_config

# 方法1：使用默认配置
config_manager = ConfigManager()
config = config_manager.load_config("default.yaml")

# 方法2：指定配置目录
config_manager = ConfigManager("/path/to/configs")
config = config_manager.load_config("default.yaml")

# 方法3：使用便捷函数
config = load_config("default.yaml", config_dir="/path/to/configs")

# 创建pipeline并执行
pipeline = SAMPipeline(config)
result = pipeline.run_full_pipeline(
    input_images="demo/input",
    background_root="demo/backgrounds", 
    output_root="demo/output",
    sam_checkpoint="sam_vit_h_4b8939.pth"
)

# 检查结果
if result.get('status') == 'success':
    print("✅ 处理完成!")
else:
    print(f"❌ 处理失败: {result.get('error')}")
```

## ⚙️ 配置文件

项目使用YAML配置文件管理参数，默认配置文件位于 `configs/default.yaml`。

### 配置结构

```yaml
# Step1: SAM分割抠图配置
step1:
  sam_checkpoint: "/path/to/sam_checkpoint.pth"
  model_type: "vit_h"
  device: "cuda"
  category_profiles_file: "category_profiles.yaml"
  cutout_dir_name: "cutouts"
  metadata_ndjson_name: "cutouts_metadata.ndjson"
  enable_content_filter: true
  content_filter_std_dev_threshold: 13.0
  nms_iou_threshold: 0.30
  nested_containment_threshold: 0.80
  enable_padded_sam: true
  pad_px: 52

# Step2: 物理尺寸调整配置  
step2:
  pixel_density: 1.554
  enable_boost: false
  boost_multipliers_file: "boost_multipliers.yaml"
  
# Step3: 背景合成配置
step3:
  target_width: 1088
  target_height: 1088
  left_trim: 296
  right_trim: 296
  top_trim: 50
  bottom_trim: 50
  mixed_classes: false
  bg_repeat: 1
  density_config_file: "density_config.yaml"

# Step4: COCO转换配置
step4:
  make_filepaths_absolute: true
  junge_style: true
  min_area: 1.0
  start_image_id: 1
  start_anno_id: 1

# 全局配置
global:
  seed: 42
  verbose: false
```

### 配置文件说明

- **category_profiles.yaml**: 定义不同类别的SAM检测参数
- **physical_sizes.yaml**: 定义各类别的物理尺寸信息
- **default.yaml**: 主配置文件

## 📁 项目结构

```
sam_auto_annotator/
├── LICENSE                     # 开源许可证
├── MANIFEST.in                 # 包分发清单
├── README.md                   # 项目说明文档
├── pyproject.toml             # 项目配置和依赖管理
├── requirements.txt           # 运行时依赖
├── requirements-dev.txt       # 开发依赖
├── setup.py                   # 包安装脚本
├── src/                       # 源代码目录
│   └── sam_auto_annotator/    # 主包目录
│       ├── __init__.py        # 包初始化文件
│       ├── __main__.py        # python -m 支持
│       ├── cli.py             # 命令行接口
│       ├── config.py          # 配置管理
│       ├── pipeline.py        # 主流程控制
│       ├── py.typed           # 类型提示标记
│       ├── core/              # 核心处理模块
│       │   ├── __init__.py
│       │   ├── step1_cutout.py      # 步骤1：图像裁剪
│       │   ├── step2_resize.py      # 步骤2：图像缩放
│       │   ├── step3_synthesize.py  # 步骤3：图像合成
│       │   └── step4_coco_convert.py # 步骤4：COCO格式转换
│       ├── data/              # 数据处理工具
│       │   └── __init__.py
│       └── models/            # 模型相关
│           └── __init__.py
├── configs/                   # 配置文件目录
│   ├── category_profiles.yaml # 类别配置
│   ├── default.yaml          # 默认配置
│   └── physical_sizes.yaml   # 物理尺寸配置
├── examples/                  # 使用示例
│   └── basic_usage.py        # 基础使用示例
├── tests/                     # 测试文件目录
├── docs/                      # 文档目录
└── data/                      # 示例数据
    ├── sample_configs/        # 示例配置
    └── sample_images/         # 示例图片
```

### 目录说明

- **src/sam_auto_annotator/**: 主要的Python包代码
- **core/**: 包含四个核心处理步骤的实现
- **configs/**: 存放各种配置文件
- **examples/**: 提供使用示例和教程
- **tests/**: 单元测试和集成测试
- **docs/**: 项目文档和API文档
- **data/**: 示例数据和配置模板


## 详细使用说明

## 💡 使用技巧

### 1. 处理大量图片

```bash
# 使用GPU加速（推荐）
sam-annotator full --device cuda \
    --input-images large_dataset/ \
    --background-root backgrounds/ \
    --output-root output/ \
    --sam-checkpoint sam_vit_h_4b8939.pth

# 如果GPU内存不足，使用CPU
sam-annotator full --device cpu \
    --input-images large_dataset/ \
    --background-root backgrounds/ \
    --output-root output/ \
    --sam-checkpoint sam_vit_h_4b8939.pth
```

### 2. 只生成抠图（不做背景合成）

```bash
# 只执行Step1和Step2
sam-annotator step1 --input-images input/ --output-dir cutouts/ --sam-checkpoint sam_vit_h_4b8939.pth
sam-annotator step2 --cutout-root cutouts/ --output-root resized/
```

### 3. 批量处理多个类别

```bash
# 为不同类别的图片分别处理
for category in cat dog bird; do
    sam-annotator full \
        --input-images data/${category}/ \
        --background-root backgrounds/ \
        --output-root output/${category}/ \
        --sam-checkpoint sam_vit_h_4b8939.pth
done
```

## 🔧 故障排除

### 常见问题

**1. CUDA内存不足**
```bash
# 解决方案：使用CPU模式
sam-annotator full --device cpu ...
```

**2. SAM模型下载失败**
```bash
# 手动下载模型文件
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
# 或使用其他下载工具
curl -O https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

**3. 没有检测到目标对象**
- 检查输入图片质量
- 尝试调整SAM模型参数
- 确保图片中有清晰的目标对象

**4. 生成的数据集为空**
- 检查背景图片目录是否有图片
- 确保Step1成功生成了抠图
- 查看详细日志：`sam-annotator full --verbose ...`

### 性能优化

- **使用GPU**: 设置 `--device cuda` 可大幅提升处理速度
- **调整图片尺寸**: 较小的图片处理更快
- **并行处理**: 可以同时运行多个实例处理不同的数据

## 📖 完整示例

这里是一个完整的端到端示例：

```bash
# 1. 创建工作目录
mkdir -p demo/{input,backgrounds,output}

# 2. 准备测试图片（假设你有一些猫的图片）
cp /path/to/cat_photos/* demo/input/
cp /path/to/background_images/* demo/backgrounds/

# 3. 下载SAM模型
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# 4. 运行完整流程
sam-annotator full \
    --input-images demo/input \
    --background-root demo/backgrounds \
    --output-root demo/output \
    --sam-checkpoint sam_vit_h_4b8939.pth \
    --verbose

# 5. 查看结果
ls demo/output/
# 你会看到：
# - cutouts/          # 抠出的目标对象
# - resized/          # 调整尺寸后的对象
# - synthesized/      # 合成的训练图片
# - annotations.json  # COCO格式标注文件
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！
1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 📞 联系方式

- 项目主页: https://github.com/xemj9/sam-auto-annotator
- 问题反馈: https://github.com/xemj9/sam-auto-annotator/issues
- 邮箱: 785631669@qq.com
