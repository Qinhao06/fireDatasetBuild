#!/usr/bin/env python3
from ultralytics import YOLO
import cv2
import os

def test_yolo_model():
    # 模型路径
    model_path = "runs/detect/fire_detection/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("请先训练模型或指定正确的模型路径")
        return
    
    # 加载训练好的模型
    model = YOLO(model_path)
    
    # 测试图像目录
    test_images_dir = "middle_photo/fire-scene-photo"
    
    if not os.path.exists(test_images_dir):
        print(f"❌ 测试图像目录不存在: {test_images_dir}")
        return
    
    print("🔍 开始测试模型...")
    
    # 获取测试图像
    test_images = [f for f in os.listdir(test_images_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not test_images:
        print("❌ 测试目录中没有找到图像文件")
        return
    
    # 创建结果目录
    results_dir = "test_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 测试前几张图像
    for i, img_name in enumerate(test_images[:5]):
        img_path = os.path.join(test_images_dir, img_name)
        
        # 进行预测
        results = model(img_path)
        
        # 保存结果
        for j, result in enumerate(results):
            result.save(filename=f"{results_dir}/result_{i}_{img_name}")
        
        print(f"✅ 处理完成: {img_name}")
    
    print(f"🎉 测试完成！结果保存在: {results_dir}/")

if __name__ == "__main__":
    test_yolo_model()
