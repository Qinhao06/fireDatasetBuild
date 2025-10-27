import os
import random
import cv2
import numpy as np
from pathlib import Path

class SimpleFireDatasetGenerator:
    def __init__(self, fire_images_dir, background_dir, output_dir):
        self.fire_images_dir = Path(fire_images_dir)
        self.background_dir = Path(background_dir)
        self.output_dir = Path(output_dir)
        
        # 创建输出目录
        self.images_output_dir = self.output_dir / "images"
        self.labels_output_dir = self.output_dir / "labels"
        self.images_output_dir.mkdir(parents=True, exist_ok=True)
        self.labels_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取文件列表
        self.fire_images = list(self.fire_images_dir.glob("*.png")) + list(self.fire_images_dir.glob("*.jpg"))
        self.background_images = list(self.background_dir.glob("*.jpg")) + list(self.background_dir.glob("*.png"))
        
        print(f"找到 {len(self.fire_images)} 张火焰图像")
        print(f"找到 {len(self.background_images)} 张背景图像")
    
    def load_fire_image(self, fire_path):
        """加载火焰图像"""
        fire_img = cv2.imread(str(fire_path), cv2.IMREAD_UNCHANGED)
        
        if fire_img is None:
            return None
            
        # 如果是PNG且有4个通道（RGBA）
        if len(fire_img.shape) == 3 and fire_img.shape[2] == 4:
            return fire_img
        
        # 如果是3通道图像，添加alpha通道
        if len(fire_img.shape) == 3 and fire_img.shape[2] == 3:
            # 创建alpha通道（去除黑色背景）
            gray = cv2.cvtColor(fire_img, cv2.COLOR_BGR2GRAY)
            _, alpha = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
            
            # 转换为BGRA
            fire_img = cv2.cvtColor(fire_img, cv2.COLOR_BGR2BGRA)
            fire_img[:,:,3] = alpha
            
        return fire_img
    
    def resize_fire_image(self, fire_img, target_size_range=(50, 200)):
        """调整火焰图像大小"""
        h, w = fire_img.shape[:2]
        
        # 随机选择目标大小
        target_size = random.randint(*target_size_range)
        
        # 保持宽高比
        if w > h:
            new_w = target_size
            new_h = int(h * target_size / w)
        else:
            new_h = target_size
            new_w = int(w * target_size / h)
        
        return cv2.resize(fire_img, (new_w, new_h))
    
    def paste_fire_on_background(self, background_img, fire_img):
        """将火焰图像贴到背景图像上"""
        bg_h, bg_w = background_img.shape[:2]
        fire_h, fire_w = fire_img.shape[:2]
        
        # 确保火焰不会超出背景边界
        if fire_w >= bg_w or fire_h >= bg_h:
            scale = min(bg_w * 0.5 / fire_w, bg_h * 0.5 / fire_h)
            new_w, new_h = int(fire_w * scale), int(fire_h * scale)
            fire_img = cv2.resize(fire_img, (new_w, new_h))
            fire_h, fire_w = new_h, new_w
        
        # 随机选择粘贴位置
        paste_x = random.randint(0, bg_w - fire_w)
        paste_y = random.randint(0, bg_h - fire_h)
        
        # 创建结果图像
        result_img = background_img.copy()
        
        # 如果火焰图像有alpha通道，使用alpha混合
        if fire_img.shape[2] == 4:
            fire_rgb = fire_img[:,:,:3]
            alpha = fire_img[:,:,3] / 255.0
            
            # 获取背景区域
            bg_region = result_img[paste_y:paste_y+fire_h, paste_x:paste_x+fire_w]
            
            # Alpha混合
            for c in range(3):
                bg_region[:,:,c] = (alpha * fire_rgb[:,:,c] + (1 - alpha) * bg_region[:,:,c])
            
            result_img[paste_y:paste_y+fire_h, paste_x:paste_x+fire_w] = bg_region
        else:
            # 直接覆盖
            result_img[paste_y:paste_y+fire_h, paste_x:paste_x+fire_w] = fire_img[:,:,:3]
        
        # 计算YOLO格式的边界框
        x_center = (paste_x + fire_w / 2) / bg_w
        y_center = (paste_y + fire_h / 2) / bg_h
        width = fire_w / bg_w
        height = fire_h / bg_h
        
        return result_img, (x_center, y_center, width, height)
    
    def generate_dataset(self, num_images=500):
        """生成数据集"""
        generated_count = 0
        
        for i in range(num_images):
            try:
                # 随机选择背景图像
                bg_path = random.choice(self.background_images)
                background = cv2.imread(str(bg_path))
                
                if background is None:
                    print(f"无法加载背景图像: {bg_path}")
                    continue
                
                # 随机选择火焰图像
                fire_path = random.choice(self.fire_images)
                fire_img = self.load_fire_image(fire_path)
                
                if fire_img is None:
                    print(f"无法加载火焰图像: {fire_path}")
                    continue
                
                # 调整火焰图像大小
                fire_img = self.resize_fire_image(fire_img)
                
                # 将火焰贴到背景上
                result_img, bbox = self.paste_fire_on_background(background, fire_img)
                
                # 保存图像
                output_img_name = f"fire_dataset_{i:06d}.jpg"
                output_img_path = self.images_output_dir / output_img_name
                cv2.imwrite(str(output_img_path), result_img)
                
                # 保存YOLO标注
                output_label_name = f"fire_dataset_{i:06d}.txt"
                output_label_path = self.labels_output_dir / output_label_name
                
                # YOLO格式：class_id x_center y_center width height
                yolo_annotation = f"0 {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}"
                
                with open(output_label_path, 'w') as f:
                    f.write(yolo_annotation)
                
                generated_count += 1
                
                if (i + 1) % 50 == 0:
                    print(f"已生成 {i + 1} 张图像...")
                    
            except Exception as e:
                print(f"生成第 {i} 张图像时出错: {e}")
                continue
        
        print(f"数据集生成完成！成功生成 {generated_count} 张图像")
        
        # 创建classes.txt文件
        classes_path = self.output_dir / "classes.txt"
        with open(classes_path, 'w') as f:
            f.write("fire\n")
        
        # 创建数据集配置文件
        config_content = f"""# Fire Detection Dataset Configuration
path: {self.output_dir.absolute()}
train: images
val: images

# Classes
nc: 1  # number of classes
names: ['fire']  # class names
"""
        
        config_path = self.output_dir / "dataset.yaml"
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        print(f"输出目录: {self.output_dir}")
        print(f"图像目录: {self.images_output_dir}")
        print(f"标注目录: {self.labels_output_dir}")
        print(f"数据集配置文件: {config_path}")

def main():
    # 配置路径
    fire_images_dir = "fire-photo"
    background_dir = "middle_photo/fire-scene-photo"
    output_dir = "fire_yolo_dataset"
    
    # 检查输入目录是否存在
    if not Path(fire_images_dir).exists():
        print(f"错误：火焰图像目录不存在: {fire_images_dir}")
        return
    
    if not Path(background_dir).exists():
        print(f"错误：背景图像目录不存在: {background_dir}")
        return
    
    # 创建数据集生成器
    generator = SimpleFireDatasetGenerator(fire_images_dir, background_dir, output_dir)
    
    if len(generator.fire_images) == 0:
        print("错误：未找到火焰图像文件")
        return
    
    if len(generator.background_images) == 0:
        print("错误：未找到背景图像文件")
        return
    
    # 生成数据集
    generator.generate_dataset(num_images=300)

if __name__ == "__main__":
    main()