#!/usr/bin/env python3
"""
运行光源负样本数据集生成器
用于生成YOLO训练的负样本数据集
"""

from light_negative_dataset_generator import LightNegativeDatasetGenerator
import argparse

def main():
    parser = argparse.ArgumentParser(description='生成光源负样本数据集')
    parser.add_argument('--light_dir', default='light', help='光源图像文件夹路径')
    parser.add_argument('--background_dir', default='middle_photo', help='背景图像文件夹路径')
    parser.add_argument('--output_dir', default='light_negative_dataset', help='输出文件夹路径')
    parser.add_argument('--num_samples', type=int, default=1000, help='生成的样本数量')
    parser.add_argument('--min_lights', type=int, default=1, help='每张图片最少光源数量')
    parser.add_argument('--max_lights', type=int, default=2, help='每张图片最多光源数量')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔥 光源负样本数据集生成器")
    print("=" * 60)
    print(f"光源图像文件夹: {args.light_dir}")
    print(f"背景图像文件夹: {args.background_dir}")
    print(f"输出文件夹: {args.output_dir}")
    print(f"生成样本数量: {args.num_samples}")
    print(f"每张图片光源数量: {args.min_lights}-{args.max_lights}")
    print("=" * 60)
    
    # 创建生成器
    generator = LightNegativeDatasetGenerator(
        light_images_dir=args.light_dir,
        background_dir=args.background_dir,
        output_dir=args.output_dir
    )
    
    # 生成负样本
    generator.generate_negative_samples(
        num_samples=args.num_samples,
        lights_per_image_range=(args.min_lights, args.max_lights)
    )
    
    print("\n✅ 负样本数据集生成完成！")
    print(f"📁 图像文件: {args.output_dir}/images/")
    print(f"📁 标签文件: {args.output_dir}/labels/")
    print("\n💡 使用提示:")
    print("- 生成的标签文件为空文件，表示图像中没有目标对象（负样本）")
    print("- 可以将此数据集与正样本数据集合并用于YOLO训练")

if __name__ == "__main__":
    main()