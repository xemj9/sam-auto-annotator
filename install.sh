#!/bin/bash
# SAM Auto Annotator 安装脚本

echo "🚀 开始安装 SAM Auto Annotator..."

# 检查Python版本
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "检测到Python版本: $python_version"

if [[ $(echo "$python_version < 3.8" | bc -l) -eq 1 ]]; then
    echo "❌ 需要Python 3.8或更高版本"
    exit 1
fi

# 安装依赖
echo "📦 安装依赖包..."
pip install -r requirements.txt

# 安装项目
echo "🔧 安装项目..."
pip install -e .

# 下载SAM模型（可选）
read -p "是否下载SAM模型文件？(y/n): " download_model
if [[ $download_model == "y" || $download_model == "Y" ]]; then
    echo "⬇️ 下载SAM模型..."
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
fi

echo "✅ 安装完成！"
echo "运行 'sam-annotator --help' 查看使用说明"