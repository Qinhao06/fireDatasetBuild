#!/usr/bin/env python3
"""
火焰数据集生成器运行脚本
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """安装必要的依赖包"""
    requirements = [
        'opencv-python',
        'numpy',
        'Pillow'
    ]
    
    print("正在安装依赖包...")
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✓ {package} 安装成功")
        except subprocess.CalledProcessError:
            print(f"✗ {package} 安装失败")
            return False
    
    return True

def check_directories():
    """检查输入目录是否存在"""
    fire_dir = Path("fire-photo")
    bg_dir = Path("middle_photo/fire-scene-photo")
    
    if not fire_dir.exists():
        print(f"错误：火焰图像目录不存在: {fire_dir}")
        return False
    
    if not bg_dir.exists():
        print(f"错误：背景图像目录不存在: {bg_dir}")
        return False
    
    # 检查是否有图像文件
    fire_images = list(fire_dir.glob("*.png")) + list(fire_dir.glob("*.jpg"))
    bg_images = list(bg_dir.glob("*.jpg")) + list(bg_dir.glob("*.png"))
    
    if len(fire_images) == 0:
        print(f"错误：火焰图像目录中没有找到图像文件: {fire_dir}")
        return False
    
    if len(bg_images) == 0:
        print(f"错误：背景图像目录中没有找到图像文件: {bg_dir}")
        return False
    
    print(f"✓ 找到 {len(fire_images)} 张火焰图像")
    print(f"✓ 找到 {len(bg_images)} 张背景图像")
    
    return True

def main():
    print("=== 火焰数据集生成器 ===\n")
    
    # 检查目录
    if not check_directories():
        return
    
    # 询问是否安装依赖
    install_deps = input("是否需要安装依赖包？(y/n): ").lower().strip()
    if install_deps in ['y', 'yes', '是']:
        if not install_requirements():
            print("依赖安装失败，尝试运行简化版本...")
    
    # 询问生成参数
    try:
        num_images = int(input("请输入要生成的图像数量 (默认300): ") or "300")
    except ValueError:
        num_images = 300
    
    print(f"\n开始生成 {num_images} 张训练图像...")
    
    # 尝试运行完整版本
    try:
        from fire_dataset_generator import FireDatasetGenerator
        
        generator = FireDatasetGenerator(
            fire_images_dir="fire-photo",
            background_dir="middle_photo/fire-scene-photo",
            output_dir="fire_yolo_dataset"
        )
        
        generator.generate_dataset(
            num_images=num_images,
            fires_per_image_range=(1, 2)
        )
        
        print("\n✓ 数据集生成完成！")
        
    except ImportError as e:
        print(f"完整版本运行失败: {e}")
        print("尝试运行简化版本...")
        
        try:
            from simple_fire_dataset_generator import SimpleFireDatasetGenerator
            
            generator = SimpleFireDatasetGenerator(
                fire_images_dir="fire-photo",
                background_dir="middle_photo/fire-scene-photo",
                output_dir="fire_yolo_dataset"
            )
            
            generator.generate_dataset(num_images=num_images)
            
            print("\n✓ 数据集生成完成！")
            
        except Exception as e:
            print(f"简化版本也运行失败: {e}")
            print("请检查Python环境和依赖包安装情况")
    
    except Exception as e:
        print(f"生成过程中出现错误: {e}")

if __name__ == "__main__":
    main()