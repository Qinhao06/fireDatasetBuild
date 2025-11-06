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
        
        # 获取文件列表 - 递归搜索所有子文件夹
        self.fire_images = (list(self.fire_images_dir.rglob("*.png")) + 
                           list(self.fire_images_dir.rglob("*.jpg")) +
                           list(self.fire_images_dir.rglob("*.jpeg")) +
                           list(self.fire_images_dir.rglob("*.PNG")) +
                           list(self.fire_images_dir.rglob("*.JPG")) +
                           list(self.fire_images_dir.rglob("*.JPEG")))
        
        # 递归获取background_dir及其所有子文件夹中的图片
        self.background_images = (list(self.background_dir.rglob("*.jpg")) + 
                                 list(self.background_dir.rglob("*.png")) +
                                 list(self.background_dir.rglob("*.jpeg")) +
                                 list(self.background_dir.rglob("*.JPG")) +
                                 list(self.background_dir.rglob("*.PNG")) +
                                 list(self.background_dir.rglob("*.JPEG")))
        
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
        
        # 计算局部亮度的平均值和标准差
        local_avg_brightness = np.mean(local_brightness_values)
        brightness_std = np.std(local_brightness_values)
        
        # 综合评估亮度（权重：整体70%，局部30%）
        final_brightness = 0.7 * overall_brightness + 0.3 * local_avg_brightness
        
        return {
            'brightness': final_brightness,  # 0-255
            'brightness_std': brightness_std,  # 亮度变化程度
            'brightness_ratio': final_brightness / 255.0  # 归一化亮度 0-1
        }

    def analyze_fire_brightness(self, fire_img):
        """分析火焰图像的亮度特征"""
        # 分离RGB通道
        if fire_img.shape[2] == 4:
            fire_rgb = fire_img[:,:,:3]
            alpha = fire_img[:,:,3]
            # 只分析有效像素（alpha > 0的区域）
            valid_mask = alpha > 50  # 忽略几乎透明的像素
        else:
            fire_rgb = fire_img
            valid_mask = np.ones(fire_rgb.shape[:2], dtype=bool)
        
        if not np.any(valid_mask):
            return {'brightness': 128, 'brightness_ratio': 0.5}
        
        # 转换为灰度并计算有效区域亮度
        gray_fire = cv2.cvtColor(fire_rgb, cv2.COLOR_BGR2GRAY)
        valid_pixels = gray_fire[valid_mask]
        
        # 计算火焰的整体亮度
        fire_brightness = np.mean(valid_pixels)
        
        # 分析火焰的亮度分布
        fire_brightness_std = np.std(valid_pixels)
        
        # 计算火焰的"热度"（红色和黄色成分）
        red_component = np.mean(fire_rgb[:,:,2][valid_mask])  # Red channel
        green_component = np.mean(fire_rgb[:,:,1][valid_mask])  # Green channel
        blue_component = np.mean(fire_rgb[:,:,0][valid_mask])  # Blue channel
        
        # 计算火焰的色温特征（红+绿相对于蓝的比例）
        warmth_ratio = (red_component + green_component) / max(blue_component, 1)
        
        return {
            'brightness': fire_brightness,
            'brightness_ratio': fire_brightness / 255.0,
            'brightness_std': fire_brightness_std,
            'warmth_ratio': warmth_ratio,
            'red_intensity': red_component,
            'green_intensity': green_component,
            'blue_intensity': blue_component
        }

    def adaptive_fire_processing(self, fire_img, bg_brightness_info):
        """根据火焰和背景的亮度对比自适应处理火焰图像"""
        # 分析火焰本身的亮度特征
        fire_info = self.analyze_fire_brightness(fire_img)
        
        bg_brightness_ratio = bg_brightness_info['brightness_ratio']
        fire_brightness_ratio = fire_info['brightness_ratio']
        
        # 计算亮度对比度（火焰相对于背景的亮度比）
        brightness_contrast = fire_brightness_ratio / max(bg_brightness_ratio, 0.1)
        
        # 分离RGB通道和Alpha通道
        if fire_img.shape[2] == 4:
            fire_rgb = fire_img[:,:,:3].astype(np.float32)
            alpha = fire_img[:,:,3]
        else:
            fire_rgb = fire_img.astype(np.float32)
            alpha = None
        
        # 计算理想的火焰亮度（基于环境亮度的适应性调整）
        # 火焰应该比背景亮，但整体亮度要与环境匹配
        
        if bg_brightness_ratio < 0.2:  # 很暗的背景（如夜晚）
            # 在很暗的环境中，火焰也应该相对较暗，但要保持足够的对比度
            target_brightness_ratio = random.uniform(0.4, 0.7)  # 目标亮度比背景高2-3倍
            brightness_factor = target_brightness_ratio / max(fire_brightness_ratio, 0.1)
            brightness_factor = np.clip(brightness_factor, 0.6, 1.4)  # 限制调整范围
            
            # 增强对比度和暖色调，让火焰在暗环境中更突出
            contrast_factor = random.uniform(1.3, 1.6)
            fire_rgb[:,:,2] = np.clip(fire_rgb[:,:,2] * random.uniform(1.1, 1.2), 0, 255)  # Red
            fire_rgb[:,:,1] = np.clip(fire_rgb[:,:,1] * random.uniform(1.05, 1.1), 0, 255)  # Green
            
        elif bg_brightness_ratio < 0.4:  # 较暗的背景（如黄昏）
            # 火焰亮度适中，保持与环境的协调
            target_brightness_ratio = random.uniform(0.5, 0.8)
            brightness_factor = target_brightness_ratio / max(fire_brightness_ratio, 0.1)
            brightness_factor = np.clip(brightness_factor, 0.7, 1.3)
            
            contrast_factor = random.uniform(1.2, 1.4)
            fire_rgb[:,:,2] = np.clip(fire_rgb[:,:,2] * random.uniform(1.05, 1.15), 0, 255)  # Red
            
        elif bg_brightness_ratio < 0.6:  # 中等亮度背景（如室内光线）
            # 保持火焰的自然亮度
            target_brightness_ratio = random.uniform(0.6, 0.85)
            brightness_factor = target_brightness_ratio / max(fire_brightness_ratio, 0.1)
            brightness_factor = np.clip(brightness_factor, 0.8, 1.2)
            
            contrast_factor = random.uniform(1.1, 1.3)
            
        else:  # 较亮的背景（如白天室外）
            # 在明亮环境中，火焰应该显得相对暗淡但仍可见
            if bg_brightness_ratio > 0.8:  # 非常亮的背景
                target_brightness_ratio = random.uniform(0.6, 0.8)  # 火焰不能太亮
            else:
                target_brightness_ratio = random.uniform(0.7, 0.9)
            
            brightness_factor = target_brightness_ratio / max(fire_brightness_ratio, 0.1)
            brightness_factor = np.clip(brightness_factor, 0.6, 1.1)
            
            contrast_factor = random.uniform(0.9, 1.2)
            
            # 在明亮背景中降低饱和度，让火焰更自然
            hsv = cv2.cvtColor(fire_rgb.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:,:,1] *= random.uniform(0.85, 0.95)  # 降低饱和度
            fire_rgb = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
        
        # 应用亮度调整
        fire_rgb = fire_rgb * brightness_factor
        
        # 应用对比度调整
        fire_rgb = np.clip((fire_rgb - 128) * contrast_factor + 128, 0, 255)
        
        # 根据环境复杂度添加细微变化
        if bg_brightness_info['brightness_std'] > 25:  # 背景明暗变化较大
            # 添加轻微的亮度变化，模拟环境光影响
            noise_strength = random.uniform(0.01, 0.03)
            brightness_noise = np.random.normal(1.0, noise_strength, fire_rgb.shape[:2])
            brightness_noise = np.expand_dims(brightness_noise, axis=2)
            fire_rgb = np.clip(fire_rgb * brightness_noise, 0, 255)
        
        # 重新组合图像
        if alpha is not None:
            result = np.zeros((fire_rgb.shape[0], fire_rgb.shape[1], 4), dtype=np.uint8)
            result[:,:,:3] = fire_rgb.astype(np.uint8)
            result[:,:,3] = alpha
        else:
            result = fire_rgb.astype(np.uint8)
        
        return result

    def random_transform_fire(self, fire_img, bg_size=None, background_img=None):
        """随机变换火焰图像：缩放、旋转、根据背景自适应调整亮度等"""
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
        
        # 根据背景图像自适应调整火焰外观
        if background_img is not None:
            bg_brightness_info = self.analyze_background_brightness(background_img)
            fire_img = self.adaptive_fire_processing(fire_img, bg_brightness_info)
        else:
            # 如果没有背景信息，使用原来的随机调整方式
            brightness = random.uniform(0.7, 1.3)
            fire_img[:,:,:3] = np.clip(fire_img[:,:,:3] * brightness, 0, 255)
        
        return fire_img
    
    def get_adaptive_fire_size_range(self, bg_size, fire_count):
        """根据背景大小和火焰数量，自适应调整火焰尺寸范围"""
        bg_h, bg_w = bg_size
        
        # 根据火焰数量调整尺寸 - 火焰越多，单个火焰应该越小
        if fire_count == 1:
            # 单个火焰可以稍大一些
            max_w_ratio, max_h_ratio = 0.3, 0.45
            min_w_ratio, min_h_ratio = 0.01, 0.02
        elif fire_count == 2:
            # 两个火焰，中等大小
            max_w_ratio, max_h_ratio = 0.30, 0.45
            min_w_ratio, min_h_ratio = 0.01, 0.02
        else:
            # 多个火焰，每个都应该较小
            max_w_ratio, max_h_ratio = 0.25, 0.35
            min_w_ratio, min_h_ratio = 0.01, 0.02
        
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
        max_width_ratio = 0.4  # 火焰最大宽度不超过背景的30%
        max_height_ratio = 0.5  # 火焰最大高度不超过背景的40%
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
                    
                    # 随机变换火焰图像，传入背景尺寸和背景图像进行自适应处理
                    transformed_fire = self.random_transform_fire(fire_img, bg_size=(bg_h, bg_w), background_img=background)
                    
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
    background_dir = "middle_photo"
    output_dir = "fire_yolo_dataset"
    
    # 创建数据集生成器
    generator = FireDatasetGenerator(fire_images_dir, background_dir, output_dir)
    
    # 生成数据集
    generator.generate_dataset(
        num_images=10000,  # 生成500张图像
        fires_per_image_range=(1, 2)  # 每张图像1-2个火焰
    )

if __name__ == "__main__":
    main()