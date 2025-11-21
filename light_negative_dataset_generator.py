import os
import random
import cv2
import numpy as np
from pathlib import Path

class LightNegativeDatasetGenerator:
    def __init__(self, light_images_dir, background_dir, output_dir):
        """
        负样本数据集生成器 - 将light文件夹的图片随机贴到middle_photo的背景上
        
        Args:
            light_images_dir: light图片文件夹路径
            background_dir: middle_photo背景图片文件夹路径  
            output_dir: 输出文件夹路径
        """
        self.light_images_dir = Path(light_images_dir)
        self.background_dir = Path(background_dir)
        self.output_dir = Path(output_dir)
        
        # 创建输出目录
        self.images_output_dir = self.output_dir / "images"
        self.labels_output_dir = self.output_dir / "labels"
        self.images_output_dir.mkdir(parents=True, exist_ok=True)
        self.labels_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取文件列表 - 递归搜索所有子文件夹
        self.light_images = (list(self.light_images_dir.rglob("*.png")) + 
                           list(self.light_images_dir.rglob("*.jpg")) +
                           list(self.light_images_dir.rglob("*.jpeg")) +
                           list(self.light_images_dir.rglob("*.PNG")) +
                           list(self.light_images_dir.rglob("*.JPG")) +
                           list(self.light_images_dir.rglob("*.JPEG")))
        
        # 递归获取background_dir及其所有子文件夹中的图片
        self.background_images = (list(self.background_dir.rglob("*.jpg")) + 
                                 list(self.background_dir.rglob("*.png")) +
                                 list(self.background_dir.rglob("*.jpeg")) +
                                 list(self.background_dir.rglob("*.JPG")) +
                                 list(self.background_dir.rglob("*.PNG")) +
                                 list(self.background_dir.rglob("*.JPEG")))
        
        print(f"找到 {len(self.light_images)} 张光源图像")
        print(f"找到 {len(self.background_images)} 张背景图像")
    
    def load_light_image_with_transparency(self, light_path):
        """加载光源图像，处理透明背景"""
        try:
            if light_path.suffix.lower() == '.png':
                # PNG图像可能有透明通道
                light_img = cv2.imread(str(light_path), cv2.IMREAD_UNCHANGED)
                if light_img is None:
                    return None
                if len(light_img.shape) == 3 and light_img.shape[2] == 4:  # 有alpha通道
                    return light_img
                else:
                    # 没有alpha通道，转换为RGBA
                    light_img = cv2.cvtColor(light_img, cv2.COLOR_BGR2BGRA)
                    return light_img
            else:
                # JPG图像，需要创建mask
                light_img = cv2.imread(str(light_path))
                if light_img is None:
                    return None
                light_img = cv2.cvtColor(light_img, cv2.COLOR_BGR2BGRA)
                
                # 创建简单的mask（去除黑色背景）
                gray = cv2.cvtColor(light_img[:,:,:3], cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
                light_img[:,:,3] = mask
                
                return light_img
        except Exception as e:
            print(f"加载光源图像失败 {light_path}: {e}")
            return None

    def analyze_background_brightness(self, background_img, sample_regions=5):
        """分析背景图片的明暗程度"""
        bg_h, bg_w = background_img.shape[:2]
        
        # 转换为灰度图进行亮度分析
        gray_bg = cv2.cvtColor(background_img, cv2.COLOR_BGR2GRAY)
        
        # 计算整体平均亮度
        overall_brightness = np.mean(gray_bg)
        
        # 随机采样多个区域计算局部亮度
        local_brightness_values = []
        region_size = min(bg_h, bg_w) // 4  # 采样区域大小
        
        for _ in range(sample_regions):
            # 随机选择采样区域
            start_x = random.randint(0, max(0, bg_w - region_size))
            start_y = random.randint(0, max(0, bg_h - region_size))
            end_x = min(start_x + region_size, bg_w)
            end_y = min(start_y + region_size, bg_h)
            
            region = gray_bg[start_y:end_y, start_x:end_x]
            local_brightness_values.append(np.mean(region))
        
        # 计算局部亮度的平均值
        local_avg_brightness = np.mean(local_brightness_values)
        brightness_std = np.std(local_brightness_values)
        
        # 综合评估亮度（权重：整体70%，局部30%）
        final_brightness = 0.7 * overall_brightness + 0.3 * local_avg_brightness
        
        return {
            'brightness': final_brightness,  # 0-255
            'brightness_std': brightness_std,  # 亮度变化程度
            'brightness_ratio': final_brightness / 255.0  # 归一化亮度 0-1
        }

    def adaptive_light_processing(self, light_img, bg_brightness_info):
        """根据背景亮度自适应处理光源图像"""
        bg_brightness_ratio = bg_brightness_info['brightness_ratio']
        
        # 分离RGB通道和Alpha通道
        if light_img.shape[2] == 4:
            light_rgb = light_img[:,:,:3].astype(np.float32)
            alpha = light_img[:,:,3]
        else:
            light_rgb = light_img.astype(np.float32)
            alpha = None
        
        # 根据背景亮度调整光源亮度
        if bg_brightness_ratio < 0.3:  # 很暗的背景
            # 在暗环境中，光源应该更亮更突出
            brightness_factor = random.uniform(1.2, 1.5)
            contrast_factor = random.uniform(1.3, 1.6)
            
        elif bg_brightness_ratio < 0.6:  # 中等亮度背景
            # 保持适中的亮度
            brightness_factor = random.uniform(1.0, 1.3)
            contrast_factor = random.uniform(1.1, 1.4)
            
        else:  # 较亮的背景
            # 在明亮环境中，光源相对暗淡
            brightness_factor = random.uniform(0.8, 1.1)
            contrast_factor = random.uniform(0.9, 1.2)
        
        # 应用亮度调整
        light_rgb = light_rgb * brightness_factor
        
        # 应用对比度调整
        light_rgb = np.clip((light_rgb - 128) * contrast_factor + 128, 0, 255)
        
        # 重新组合图像
        if alpha is not None:
            result = np.zeros((light_rgb.shape[0], light_rgb.shape[1], 4), dtype=np.uint8)
            result[:,:,:3] = light_rgb.astype(np.uint8)
            result[:,:,3] = alpha
        else:
            result = light_rgb.astype(np.uint8)
        
        return result

    def random_transform_light(self, light_img, bg_size=None, background_img=None):
        """随机变换光源图像：缩放、旋转等"""
        if light_img is None:
            return None
        h, w = light_img.shape[:2]
        
        # 根据背景大小调整缩放范围
        if bg_size is not None:
            bg_h, bg_w = bg_size
            # 光源大小范围更大，波动更明显
            max_light_w = bg_w * 0.5  # 光源最大宽度可达背景的50%
            max_light_h = bg_h * 0.5  # 光源最大高度可达背景的50%
            min_light_w = bg_w * 0.01  # 光源最小宽度为背景的1%
            min_light_h = bg_h * 0.015  # 光源最小高度为背景的1.5%
            
            # 计算缩放范围
            max_scale_w = max_light_w / w
            max_scale_h = max_light_h / h
            min_scale_w = min_light_w / w
            min_scale_h = min_light_h / h
            
            # 取更严格的限制
            max_scale = min(max_scale_w, max_scale_h, 1.0)
            min_scale = max(min_scale_w, min_scale_h, 0.02)
            
            # 确保min_scale不大于max_scale
            if min_scale > max_scale:
                min_scale = max_scale * 0.3
        else:
            # 默认缩放范围更大
            min_scale, max_scale = 0.05, 0.8
        
        # 随机缩放
        scale = random.uniform(min_scale, max_scale)
        new_w, new_h = int(w * scale), int(h * scale)
        light_img = cv2.resize(light_img, (new_w, new_h))
        
        # 随机旋转 (-180到180度，光源可以任意角度)
        angle = random.uniform(-180, 180)
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
        
        light_img = cv2.warpAffine(light_img, rotation_matrix, (new_w_rot, new_h_rot))
        
        # 根据背景图像自适应调整光源外观
        if background_img is not None:
            bg_brightness_info = self.analyze_background_brightness(background_img)
            light_img = self.adaptive_light_processing(light_img, bg_brightness_info)
        
        return light_img

    def blend_light_with_background(self, background, light_img, position):
        """将光源图像混合到背景图像上，并添加边缘柔光效果"""
        if light_img is None:
            return background
        
        x, y = position
        light_h, light_w = light_img.shape[:2]
        bg_h, bg_w = background.shape[:2]
        
        # 确保光源完全在背景范围内
        if x < 0 or y < 0 or x + light_w > bg_w or y + light_h > bg_h:
            return background
        
        try:
            # 添加边缘柔光效果（光晕）
            # 创建一个扩展的光晕区域
            glow_size = random.randint(15, 40)  # 光晕扩展大小
            glow_intensity = random.uniform(0.4, 0.8)  # 光晕强度
            
            # 计算扩展后的区域
            glow_light_h = light_h + 2 * glow_size
            glow_light_w = light_w + 2 * glow_size
            
            # 创建扩展的光源图像（包含光晕）
            glow_light = np.zeros((glow_light_h, glow_light_w, 4), dtype=np.uint8)
            
            # 将原光源放在中心
            glow_light[glow_size:glow_size+light_h, glow_size:glow_size+light_w] = light_img
            
            # 对扩展图像进行多次模糊，创建柔光效果
            blur_kernel = glow_size * 2 + 1
            if blur_kernel > glow_light_h or blur_kernel > glow_light_w:
                blur_kernel = min(glow_light_h, glow_light_w) // 2 * 2 + 1
            if blur_kernel < 3:
                blur_kernel = 3
                
            glow_blur = cv2.GaussianBlur(glow_light, (blur_kernel, blur_kernel), glow_size//2)
            
            # 提取光晕的alpha通道并调整强度
            glow_alpha = glow_blur[:,:,3].astype(np.float32) * glow_intensity / 255.0
            glow_alpha = np.expand_dims(glow_alpha, axis=2)
            
            # 计算在背景上的实际位置（考虑光晕扩展）
            glow_x = max(0, x - glow_size)
            glow_y = max(0, y - glow_size)
            
            # 计算实际可以绘制的区域
            actual_glow_w = min(glow_light_w, bg_w - glow_x)
            actual_glow_h = min(glow_light_h, bg_h - glow_y)
            
            # 计算在光晕图像上的起始位置
            src_start_x = 0 if x >= glow_size else glow_size - x
            src_start_y = 0 if y >= glow_size else glow_size - y
            
            # 确保尺寸匹配
            if actual_glow_h <= 0 or actual_glow_w <= 0:
                return background
            
            # 提取背景区域
            bg_region = background[glow_y:glow_y+actual_glow_h, glow_x:glow_x+actual_glow_w].copy()
            
            # 提取对应的光晕区域
            glow_region = glow_blur[src_start_y:src_start_y+actual_glow_h, src_start_x:src_start_x+actual_glow_w]
            glow_alpha_region = glow_alpha[src_start_y:src_start_y+actual_glow_h, src_start_x:src_start_x+actual_glow_w]
            
            # 确保形状匹配
            if bg_region.shape[:2] != glow_region.shape[:2]:
                return background
            
            # 先混合柔光效果
            glow_rgb = glow_region[:,:,:3].astype(np.float32)
            bg_region_float = bg_region.astype(np.float32)
            blended = bg_region_float * (1 - glow_alpha_region) + glow_rgb * glow_alpha_region
            
            # 将混合后的区域写回背景
            background[glow_y:glow_y+actual_glow_h, glow_x:glow_x+actual_glow_w] = blended.astype(np.uint8)
            
            # 再次混合原始光源（使其中心更亮）
            if x >= 0 and y >= 0 and x + light_w <= bg_w and y + light_h <= bg_h:
                bg_region_center = background[y:y+light_h, x:x+light_w].copy()
                
                if light_img.shape[2] == 4:  # 有alpha通道
                    alpha_center = light_img[:,:,3].astype(np.float32) / 255.0
                    alpha_center = np.expand_dims(alpha_center, axis=2)
                    
                    light_rgb = light_img[:,:,:3].astype(np.float32)
                    bg_center_float = bg_region_center.astype(np.float32)
                    blended_center = bg_center_float * (1 - alpha_center) + light_rgb * alpha_center
                    background[y:y+light_h, x:x+light_w] = blended_center.astype(np.uint8)
                else:
                    # 直接覆盖（如果没有透明通道）
                    background[y:y+light_h, x:x+light_w] = light_img
                    
        except Exception as e:
            # 如果柔光处理失败，使用简单的alpha混合
            bg_region = background[y:y+light_h, x:x+light_w]
            
            if light_img.shape[2] == 4:  # 有alpha通道
                alpha = light_img[:,:,3] / 255.0
                alpha = np.expand_dims(alpha, axis=2)
                
                light_rgb = light_img[:,:,:3]
                blended = bg_region * (1 - alpha) + light_rgb * alpha
                background[y:y+light_h, x:x+light_w] = blended.astype(np.uint8)
            else:
                # 直接覆盖（如果没有透明通道）
                background[y:y+light_h, x:x+light_w] = light_img
        
        return background

    def generate_negative_samples(self, num_samples=1000, lights_per_image_range=(1, 3)):
        """
        生成负样本数据集
        
        Args:
            num_samples: 生成的样本数量
            lights_per_image_range: 每张图片上光源数量的范围
        """
        print(f"开始生成 {num_samples} 个负样本...")
        
        for i in range(num_samples):
            try:
                # 随机选择背景图片
                bg_path = random.choice(self.background_images)
                background = cv2.imread(str(bg_path))
                if background is None:
                    print(f"无法加载背景图片: {bg_path}")
                    continue
                
                bg_h, bg_w = background.shape[:2]
                
                # 随机确定这张图片上要放置的光源数量
                num_lights = random.randint(*lights_per_image_range)
                
                # 记录放置的光源位置，避免重叠
                placed_positions = []
                
                for j in range(num_lights):
                    # 随机选择光源图片
                    light_path = random.choice(self.light_images)
                    light_img = self.load_light_image_with_transparency(light_path)
                    
                    if light_img is None:
                        continue
                    
                    # 变换光源图像
                    transformed_light = self.random_transform_light(
                        light_img, (bg_h, bg_w), background
                    )
                    
                    if transformed_light is None:
                        continue
                    
                    light_h, light_w = transformed_light.shape[:2]
                    
                    # 随机选择位置，避免与已放置的光源重叠
                    max_attempts = 50
                    placed = False
                    
                    for attempt in range(max_attempts):
                        # 随机位置，确保光源完全在图像内
                        x = random.randint(0, max(0, bg_w - light_w))
                        y = random.randint(0, max(0, bg_h - light_h))
                        
                        # 检查是否与已放置的光源重叠
                        overlap = False
                        for prev_x, prev_y, prev_w, prev_h in placed_positions:
                            if not (x + light_w < prev_x or x > prev_x + prev_w or
                                   y + light_h < prev_y or y > prev_y + prev_h):
                                overlap = True
                                break
                        
                        if not overlap:
                            # 混合光源到背景
                            background = self.blend_light_with_background(
                                background, transformed_light, (x, y)
                            )
                            placed_positions.append((x, y, light_w, light_h))
                            placed = True
                            break
                    
                    if not placed:
                        print(f"无法为第 {j+1} 个光源找到合适位置")
                
                # 保存生成的图像
                output_filename = f"negative_{i:06d}.jpg"
                output_path = self.images_output_dir / output_filename
                cv2.imwrite(str(output_path), background)
                
                # 创建空的标签文件（负样本没有标注）
                label_filename = f"negative_{i:06d}.txt"
                label_path = self.labels_output_dir / label_filename
                label_path.write_text("")  # 空文件表示没有目标对象
                
                if (i + 1) % 100 == 0:
                    print(f"已生成 {i + 1}/{num_samples} 个负样本")
                    
            except Exception as e:
                print(f"生成第 {i} 个样本时出错: {e}")
                continue
        
        print(f"负样本生成完成！共生成 {num_samples} 个样本")
        print(f"图像保存在: {self.images_output_dir}")
        print(f"标签保存在: {self.labels_output_dir}")

# 使用示例
if __name__ == "__main__":
    # 配置路径
    light_images_dir = "light"  # light文件夹路径
    background_dir = "middle_photo"  # middle_photo文件夹路径
    output_dir = "light_negative_dataset"  # 输出文件夹路径
    
    # 创建生成器
    generator = LightNegativeDatasetGenerator(
        light_images_dir=light_images_dir,
        background_dir=background_dir, 
        output_dir=output_dir
    )
    
    # 生成1000个负样本，每张图片放置1-2个光源
    generator.generate_negative_samples(
        num_samples=1000,
        lights_per_image_range=(1, 2)
    )