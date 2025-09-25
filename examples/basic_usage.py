"""
SAM Auto Annotator 基本使用示例
"""

from pathlib import Path
from sam_auto_annotator import SAMPipeline, ConfigManager

def main():
    """基本使用示例"""
    
    # 1. 初始化配置管理器
    config_dir = Path(__file__).parent.parent / "configs"
    config_manager = ConfigManager(str(config_dir))
    config = config_manager.load_config("default.yaml")
    
    # 2. 创建pipeline
    pipeline = SAMPipeline(config, str(config_dir))
    
    # 3. 设置路径（请根据实际情况修改）
    input_images = "/path/to/your/input/images"
    background_root = "/path/to/your/background/images"  
    output_root = "/path/to/your/output"
    sam_checkpoint = "/path/to/your/sam_model.pth"
    
    # 4. 执行完整流程
    print("🚀 开始执行SAM Auto Annotator完整流程...")
    
    result = pipeline.run_full_pipeline(
        input_images=input_images,
        background_root=background_root,
        output_root=output_root,
        sam_checkpoint=sam_checkpoint
    )
    
    # 5. 检查结果
    if result.get('status') == 'success':
        print("✅ 流程执行成功!")
        print("📊 各步骤结果:")
        for step, step_result in result.items():
            if step != 'status':
                print(f"  {step}: {step_result.get('status', 'unknown')}")
    else:
        print(f"❌ 流程执行失败: {result.get('error', '未知错误')}")

def step_by_step_example():
    """分步执行示例"""
    
    # 初始化
    config_manager = ConfigManager()
    config = config_manager.load_config()
    pipeline = SAMPipeline(config)
    
    # 路径设置
    input_images = "/path/to/your/input/images"
    background_root = "/path/to/your/background/images"
    output_root = "/path/to/your/output"
    sam_checkpoint = "/path/to/your/sam_model.pth"
    
    # Step1: SAM分割抠图
    print("🔄 执行Step1: SAM分割抠图")
    step1_output = Path(output_root) / "step1_cutouts"
    result1 = pipeline.run_step1(
        input_images=input_images,
        output_dir=str(step1_output),
        sam_checkpoint=sam_checkpoint
    )
    
    if result1['status'] != 'success':
        print(f"❌ Step1失败: {result1['error']}")
        return
    
    # Step2: 物理尺寸调整
    print("🔄 执行Step2: 物理尺寸调整")
    step1_cutouts = step1_output / "cutouts"
    step2_output = Path(output_root) / "step2_resized"
    result2 = pipeline.run_step2(
        cutout_root=str(step1_cutouts),
        output_root=str(step2_output)
    )
    
    if result2['status'] != 'success':
        print(f"❌ Step2失败: {result2['error']}")
        return
    
    # Step3: 背景合成
    print("🔄 执行Step3: 背景合成")
    step3_output = Path(output_root) / "step3_synthesized"
    result3 = pipeline.run_step3(
        cutout_root=str(step2_output),
        background_root=background_root,
        synth_root=str(step3_output)
    )
    
    if result3['status'] != 'success':
        print(f"❌ Step3失败: {result3['error']}")
        return
    
    # Step4: COCO格式转换
    print("🔄 执行Step4: COCO格式转换")
    ndjson_file = step3_output / "placements_cropped1088_margins.ndjson"
    coco_output = step3_output / "annotations_coco.json"
    result4 = pipeline.run_step4(
        ndjson_path=str(ndjson_file),
        output_path=str(coco_output)
    )
    
    if result4['status'] == 'success':
        print("✅ 所有步骤执行成功!")
    else:
        print(f"❌ Step4失败: {result4['error']}")

if __name__ == '__main__':
    # 运行基本示例
    main()
    
    # 或者运行分步示例
    # step_by_step_example()