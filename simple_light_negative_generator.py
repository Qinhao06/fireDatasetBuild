#!/usr/bin/env python3
"""
简化版光源负样本数据集生成器
快速生成用于YOLO训练的负样本数据集
"""

import os
import random
import cv2
import numpy as np
from pathlib import Path

def load_light_with_transparency(light_path):
    """加载光源图像并处理透明背景"""
    try:
        # 读取PNG图像（包含透明通道）
        light_img = cv2.imread(str(light_path), cv2.IMREAD_UNCHANGED)
        if light_img is None:
            return None
        
        # 如果是PNG且有alpha通道
        if len(light_img.shape) == 3 and light_img.shape[2] == 4:
            return light_img
        elif len(light_img.shape) == 3:
            # 没有alpha通道，添加alpha通道
            light_img = cv2.cvtColor(light_img, cv2.COLOR_BGR2BGRA)
            # 创建简单mask（去除黑色背景）
            gray = cv2.cvtColor(light_img[:,:,:3], cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
            light_img[:,:,3] = mask
            return light_img
        
        return None
    except Exception as e:
        print(f"加载光源图像失败 {light_path}: {e}")
        return None

def random_transform_light(light_img, bg_size):
    """随机变换光源图像"""
    if light_img is None:
        return None
    
    h, w = light_img.shape[:2]
    bg_h, bg_w = bg_size
    
    # 随机缩放（光源大小范围更大，波动更明显）
    scale = random.uniform(0.4, 2.8)  # 从3%到60%，范围更大
    new_w, new_h = int(w * scale), int(h * scale)
    light_img = cv2.resize(light_img, (new_w, new_h))
    
    # 随机旋转
    angle = random.uniform(-180, 180)
    center = (new_w // 2, new_h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 计算旋转后尺寸
    cos_val = abs(rotation_matrix[0, 0])
    sin_val = abs(rotation_matrix[0, 1])
    new_w_rot = int((new_h * sin_val) + (new_w * cos_val))
    new_h_rot = int((new_h * cos_val) + (new_w * sin_val))
    
    # 调整旋转中心
    rotation_matrix[0, 2] += (new_w_rot / 2) - center[0]
    rotation_matrix[1, 2] += (new_h_rot / 2) - center[1]
    
    light_img = cv2.warpAffine(light_img, rotation_matrix, (new_w_rot, new_h_rot))
    
    # 随机调整亮度
    brightness_factor = random.uniform(0.8, 1.3)
    light_rgb = light_img[:,:,:3].astype(np.float32) * brightness_factor
    light_img[:,:,:3] = np.clip(light_rgb, 0, 255).astype(np.uint8)
    
    return light_img

def blend_light_with_background(background, light_img, position):
    """将光源混合到背景上"""
    if light_img is None:
        return background
    
    x, y = position
    light_h, light_w = light_img.shape[:2]
    bg_h, bg_w = background.shape[:2]
    
    # 确保光源在背景范围内
    if x < 0 or y < 0 or x + light_w > bg_w or y + light_h > bg_h:
        return background
    
    # 提取背景区域
    bg_region = background[y:y+light_h, x:x+light_w]
    
    # Alpha混合
    alpha = light_img[:,:,3] / 255.0
    alpha = np.expand_dims(alpha, axis=2)
    
    light_rgb = light_img[:,:,:3]
    blended = bg_region * (1 - alpha) + light_rgb * alpha
    background[y:y+light_h, x:x+light_w] = blended.astype(np.uint8)
    
    return background

def generate_light_negative_dataset(light_dir="light", 
                                   background_dir="middle_photo", 
                                   output_dir="light_negative_dataset",
                                   num_samples=500,
                                   lights_per_image=(1, 2)):
    """生成光源负样本数据集"""
    
    # 创建路径对象
    light_path = Path(light_dir)
    bg_path = Path(background_dir)
    output_path = Path(output_dir)
    
    # 创建输出目录
    images_dir = output_path / "images"
    labels_dir = output_path / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有图像文件
    light_files = list(light_path.glob("*.png")) + list(light_path.glob("*.jpg"))
    bg_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        bg_files.extend(list(bg_path.rglob(ext)))
    
    print(f"找到 {len(light_files)} 张光源图像")
    print(f"找到 {len(bg_files)} 张背景图像")
    
    if len(light_files) == 0 or len(bg_files) == 0:
        print("错误: 没有找到足够的图像文件")
        return
    
    print(f"开始生成 {num_samples} 个负样本...")
    
    success_count = 0
    for i in range(num_samples):
        try:
            # 随机选择背景
            bg_file = random.choice(bg_files)
            background = cv2.imread(str(bg_file))
            if background is None:
                continue
            
            bg_h, bg_w = background.shape[:2]
            
            # 随机确定光源数量
            num_lights = random.randint(*lights_per_image)
            
            # 放置光源
            for j in range(num_lights):
                # 随机选择光源
                light_file = random.choice(light_files)
                light_img = load_light_with_transparency(light_file)
                
                if light_img is None:
                    continue
                
                # 变换光源
                transformed_light = random_transform_light(light_img, (bg_h, bg_w))
                if transformed_light is None:
                    continue
                
                light_h, light_w = transformed_light.shape[:2]
                
                # 随机位置
                if bg_w > light_w and bg_h > light_h:
                    x = random.randint(0, bg_w - light_w)
                    y = random.randint(0, bg_h - light_h)
                    
                    # 混合到背景
                    background = blend_light_with_background(
                        background, transformed_light, (x, y)
                    )
            
            # 保存图像
            img_filename = f"negative_{i:06d}.jpg"
            img_path = images_dir / img_filename
            cv2.imwrite(str(img_path), background)
            
            # 创建空标签文件
            label_filename = f"negative_{i:06d}.txt"
            label_path = labels_dir / label_filename
            label_path.write_text("")
            
            success_count += 1
            
            if (i + 1) % 100 == 0:
                print(f"已生成 {i + 1}/{num_samples} 个样本")
                
        except Exception as e:
            print(f"生成第 {i} 个样本时出错: {e}")
            continue
    
    print(f"\n✅ 负样本生成完成！")
    print(f"成功生成: {success_count}/{num_samples} 个样本")
    print(f"图像保存在: {images_dir}")
    print(f"标签保存在: {labels_dir}")

if __name__ == "__main__":
    # 直接运行生成
    generate_light_negative_dataset(
        light_dir="light",
        background_dir="middle_photo", 
        output_dir="light_negative_dataset",
        num_samples=2000,
        lights_per_image=(1, 2)
    )