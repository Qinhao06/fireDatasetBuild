import os
import random
import cv2
import numpy as np
from pathlib import Path

class FireDatasetGenerator:
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
    
    def load_fire_image_with_transparency(self, fire_path):
        """加载火焰图像，处理透明背景"""
        try:
            if fire_path.suffix.lower() == '.png':
                # PNG图像可能有透明通道
                fire_img = cv2.imread(str(fire_path), cv2.IMREAD_UNCHANGED)
                if fire_img is None:
                    return None
                if len(fire_img.shape) == 3 and fire_img.shape[2] == 4:  # 有alpha通道
                    return fire_img
                else:
                    # 没有alpha通道，转换为RGBA
                    fire_img = cv2.cvtColor(fire_img, cv2.COLOR_BGR2BGRA)
                    return fire_img
            else:
                # JPG图像，需要创建mask
                fire_img = cv2.imread(str(fire_path))
                if fire_img is None:
                    return None
                fire_img = cv2.cvtColor(fire_img, cv2.COLOR_BGR2BGRA)
                
                # 创建简单的mask（去除黑色背景）
                gray = cv2.cvtColor(fire_img[:,:,:3], cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
                fire_img[:,:,3] = mask
                
                return fire_img
        except Exception as e:
            print(f"加载火焰图像失败 {fire_path}: {e}")
            return None
    
    def random_transform_fire(self, fire_img, bg_size=None):
        """随机变换火焰图像：缩放、旋转、调整亮度等"""
        if fire_img is None:
            return None
        h, w = fire_img.shape[:2]
        
        # 根据背景大小调整缩放范围
        if bg_size is not None:
            bg_h, bg_w = bg_size
            # 计算相对于背景的最大允许尺寸
            max_fire_w = bg_w * 0.25  # 火焰最大宽度不超过背景的25%
            max_fire_h = bg_h * 0.35  # 火焰最大高度不超过背景的35%
            min_fire_w = bg_w * 0.05  # 火焰最小宽度不少于背景的5%
            min_fire_h = bg_h * 0.08  # 火焰最小高度不少于背景的8%
            
            # 计算缩放范围
            max_scale_w = max_fire_w / w
            max_scale_h = max_fire_h / h
            min_scale_w = min_fire_w / w
            min_scale_h = min_fire_h / h
            
            # 取更严格的限制
            max_scale = min(max_scale_w, max_scale_h, 0.9)  # 最大不超过1.2倍
            min_scale = max(min_scale_w, min_scale_h, 0.2)  # 最小不少于0.2倍
            
            # 确保min_scale不大于max_scale
            if min_scale > max_scale:
                min_scale = max_scale * 0.8
        else:
            # 默认缩放范围
            min_scale, max_scale = 0.3, 0.9
        
        # 随机缩放
        scale = random.uniform(min_scale, max_scale)
        new_w, new_h = int(w * scale), int(h * scale)
        fire_img = cv2.resize(fire_img, (new_w, new_h))
        
        # 随机旋转 (-30到30度)
        angle = random.uniform(-30, 30)
        center = (new_w // 2, new_h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 计算旋转后的边界框
        cos_val = abs(rotation_matrix[0, 0])
        sin_val = abs(rotation_matrix[0, 1])
        new_w_rot = int((new_h * sin_val) + (new_w * cos_val))
        new_h_rot = int((new_h * cos_val) + (new_w * sin_val))
        
        # 调整旋转中心
        rotation_matrix[0, 2] += (new_w_rot / 2) - center[0]
        rotation_matrix[1, 2] += (new_h_rot / 2) - center[1]
        
        fire_img = cv2.warpAffine(fire_img, rotation_matrix, (new_w_rot, new_h_rot))
        
        # 随机调整亮度和对比度
        brightness = random.uniform(0.7, 1.3)
        fire_img[:,:,:3] = np.clip(fire_img[:,:,:3] * brightness, 0, 255)
        
        return fire_img
    
    def get_adaptive_fire_size_range(self, bg_size, fire_count):
        """根据背景大小和火焰数量，自适应调整火焰尺寸范围"""
        bg_h, bg_w = bg_size
        
        # 根据火焰数量调整尺寸 - 火焰越多，单个火焰应该越小
        if fire_count == 1:
            # 单个火焰可以稍大一些
            max_w_ratio, max_h_ratio = 0.35, 0.45
            min_w_ratio, min_h_ratio = 0.08, 0.12
        elif fire_count == 2:
            # 两个火焰，中等大小
            max_w_ratio, max_h_ratio = 0.25, 0.35
            min_w_ratio, min_h_ratio = 0.06, 0.10
        else:
            # 多个火焰，每个都应该较小
            max_w_ratio, max_h_ratio = 0.20, 0.30
            min_w_ratio, min_h_ratio = 0.05, 0.08
        
        return {
            'max_width': int(bg_w * max_w_ratio),
            'max_height': int(bg_h * max_h_ratio),
            'min_width': int(bg_w * min_w_ratio),
            'min_height': int(bg_h * min_h_ratio)
        }
    
    def check_overlap(self, new_bbox, existing_bboxes, min_distance=0.1):
        """检查新的边界框是否与已存在的边界框重叠"""
        if not existing_bboxes:
            return False
            
        new_x, new_y, new_w, new_h = new_bbox
        new_left = new_x - new_w / 2
        new_right = new_x + new_w / 2
        new_top = new_y - new_h / 2
        new_bottom = new_y + new_h / 2
        
        for existing_bbox in existing_bboxes:
            ex_x, ex_y, ex_w, ex_h = existing_bbox
            ex_left = ex_x - ex_w / 2
            ex_right = ex_x + ex_w / 2
            ex_top = ex_y - ex_h / 2
            ex_bottom = ex_y + ex_h / 2
            
            # 检查是否重叠（包括最小距离）
            if not (new_right + min_distance < ex_left or 
                   new_left > ex_right + min_distance or
                   new_bottom + min_distance < ex_top or 
                   new_top > ex_bottom + min_distance):
                return True
        return False

    def paste_fire_on_background(self, background_img, fire_img, existing_bboxes=None, max_attempts=50):
        """将火焰图像贴到背景图像上，避免与已存在的火焰重叠"""
        bg_h, bg_w = background_img.shape[:2]
        fire_h, fire_w = fire_img.shape[:2]
        
        # 更严格的尺寸控制 - 确保火焰不会过大
        max_width_ratio = 0.3   # 火焰最大宽度不超过背景的30%
        max_height_ratio = 0.4  # 火焰最大高度不超过背景的40%
        min_width_ratio = 0.05  # 火焰最小宽度不少于背景的5%
        min_height_ratio = 0.08 # 火焰最小高度不少于背景的8%
        
        # 检查并调整火焰尺寸
        needs_resize = False
        scale_w = scale_h = 1.0
        
        if fire_w > bg_w * max_width_ratio:
            scale_w = (bg_w * max_width_ratio) / fire_w
            needs_resize = True
        elif fire_w < bg_w * min_width_ratio:
            scale_w = (bg_w * min_width_ratio) / fire_w
            needs_resize = True
            
        if fire_h > bg_h * max_height_ratio:
            scale_h = (bg_h * max_height_ratio) / fire_h
            needs_resize = True
        elif fire_h < bg_h * min_height_ratio:
            scale_h = (bg_h * min_height_ratio) / fire_h
            needs_resize = True
        
        if needs_resize:
            # 使用更严格的缩放比例（取较小值以确保两个维度都符合要求）
            final_scale = min(scale_w, scale_h)
            new_fire_w = max(int(fire_w * final_scale), 1)
            new_fire_h = max(int(fire_h * final_scale), 1)
            fire_img = cv2.resize(fire_img, (new_fire_w, new_fire_h))
            fire_h, fire_w = new_fire_h, new_fire_w
        
        if existing_bboxes is None:
            existing_bboxes = []
        
        # 尝试多次找到不重叠的位置
        for attempt in range(max_attempts):
            # 随机选择粘贴位置
            max_x = max(0, bg_w - fire_w)
            max_y = max(0, bg_h - fire_h)
            paste_x = random.randint(0, max_x) if max_x > 0 else 0
            paste_y = random.randint(0, max_y) if max_y > 0 else 0
            
            # 计算YOLO格式的边界框 (归一化坐标)
            x_center = (paste_x + fire_w / 2) / bg_w
            y_center = (paste_y + fire_h / 2) / bg_h
            width = fire_w / bg_w
            height = fire_h / bg_h
            
            new_bbox = (x_center, y_center, width, height)
            
            # 检查是否与已存在的边界框重叠
            if not self.check_overlap(new_bbox, existing_bboxes):
                # 没有重叠，可以放置火焰
                break
        else:
            # 如果尝试了最大次数仍然重叠，返回None表示放置失败
            return None, None
        
        # 创建结果图像
        result_img = background_img.copy()
        
        # 提取alpha通道作为mask
        if fire_img.shape[2] == 4:
            fire_rgb = fire_img[:,:,:3]
            alpha = fire_img[:,:,3] / 255.0
            
            # 应用alpha混合
            for c in range(3):
                result_img[paste_y:paste_y+fire_h, paste_x:paste_x+fire_w, c] = \
                    (alpha * fire_rgb[:,:,c] + 
                     (1 - alpha) * result_img[paste_y:paste_y+fire_h, paste_x:paste_x+fire_w, c])
        else:
            # 没有alpha通道，直接覆盖
            result_img[paste_y:paste_y+fire_h, paste_x:paste_x+fire_w] = fire_img[:,:,:3]
        
        return result_img, new_bbox
    
    def generate_dataset(self, num_images=1000, fires_per_image_range=(1, 4)):
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
                
                # 随机决定添加多少个火焰
                num_fires = random.randint(*fires_per_image_range)
                
                result_img = background.copy()
                yolo_annotations = []
                existing_bboxes = []  # 存储已放置的火焰边界框
                
                successful_fires = 0
                # 获取自适应的火焰尺寸范围
                bg_h, bg_w = background.shape[:2]
                size_limits = self.get_adaptive_fire_size_range((bg_h, bg_w), num_fires)
                
                for fire_attempt in range(num_fires * 3):  # 给更多尝试机会
                    if successful_fires >= num_fires:
                        break
                        
                    # 随机选择火焰图像
                    fire_path = random.choice(self.fire_images)
                    fire_img = self.load_fire_image_with_transparency(fire_path)
                    
                    if fire_img is None:
                        continue
                    
                    # 随机变换火焰图像，传入背景尺寸进行自适应缩放
                    transformed_fire = self.random_transform_fire(fire_img, bg_size=(bg_h, bg_w))
                    
                    if transformed_fire is None:
                        continue
                    
                    # 额外的尺寸检查和调整
                    t_h, t_w = transformed_fire.shape[:2]
                    if (t_w > size_limits['max_width'] or t_h > size_limits['max_height'] or
                        t_w < size_limits['min_width'] or t_h < size_limits['min_height']):
                        # 重新调整到合适的尺寸
                        target_w = random.randint(size_limits['min_width'], size_limits['max_width'])
                        target_h = random.randint(size_limits['min_height'], size_limits['max_height'])
                        # 保持宽高比
                        aspect_ratio = t_w / t_h
                        if target_w / target_h > aspect_ratio:
                            target_w = int(target_h * aspect_ratio)
                        else:
                            target_h = int(target_w / aspect_ratio)
                        transformed_fire = cv2.resize(transformed_fire, (target_w, target_h))
                    
                    # 将火焰贴到背景上，传入已存在的边界框避免重叠
                    result_img, bbox = self.paste_fire_on_background(result_img, transformed_fire, existing_bboxes)
                    
                    if bbox is not None:  # 成功放置火焰
                        # 添加到已存在的边界框列表
                        existing_bboxes.append(bbox)
                        
                        # 添加YOLO标注 (class_id=0 表示火焰)
                        yolo_annotations.append(f"0 {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}")
                        successful_fires += 1
                
                # 只有当至少成功放置一个火焰时才保存图像
                if successful_fires > 0:
                    # 保存图像
                    output_img_name = f"fire_dataset_{i:06d}.jpg"
                    output_img_path = self.images_output_dir / output_img_name
                    cv2.imwrite(str(output_img_path), result_img)
                    
                    # 保存YOLO标注
                    output_label_name = f"fire_dataset_{i:06d}.txt"
                    output_label_path = self.labels_output_dir / output_label_name
                    
                    with open(output_label_path, 'w') as f:
                        f.write('\n'.join(yolo_annotations))
                    
                    generated_count += 1
                else:
                    print(f"第 {i} 张图像无法放置任何火焰，跳过")
                
                if (i + 1) % 100 == 0:
                    print(f"已处理 {i + 1} 张图像，成功生成 {generated_count} 张...")
                    
            except Exception as e:
                print(f"生成第 {i} 张图像时出错: {e}")
                continue
        
        print(f"数据集生成完成！成功生成 {generated_count} 张图像")
        
        # 在根目录创建classes.txt文件
        classes_path = self.output_dir / "classes.txt"
        with open(classes_path, 'w') as f:
            f.write("fire\n")
        
        # 在labels目录也创建classes.txt文件
        labels_classes_path = self.labels_output_dir / "classes.txt"
        with open(labels_classes_path, 'w') as f:
            f.write("fire\n")
        
        print(f"输出目录: {self.output_dir}")
        print(f"图像目录: {self.images_output_dir}")    
        print(f"标注目录: {self.labels_output_dir}")
        print(f"类别文件: {classes_path} 和 {labels_classes_path}")

def main():
    # 配置路径
    fire_images_dir = "fire-photo"
    background_dir = "middle_photo/fire-scene-photo"
    output_dir = "fire_yolo_dataset"
    
    # 创建数据集生成器
    generator = FireDatasetGenerator(fire_images_dir, background_dir, output_dir)
    
    # 生成数据集
    generator.generate_dataset(
        num_images=4000,  # 生成500张图像
        fires_per_image_range=(1, 2)  # 每张图像1-2个火焰
    )

if __name__ == "__main__":
    main()